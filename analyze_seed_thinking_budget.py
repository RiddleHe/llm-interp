"""
This script analyzes how ByteDance-Seed/Seed-OSS-36B-Instruct controls its thinking budget.

To start, run CUDA_VISIBLE_DEVICES=<at least three GPUS> \
    python analyze_seed_thinking_budget.py \
        --token_budget <token_budget> \
        --max_tokens <max_tokens> \
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, TopPLogitsWarper, TemperatureLogitsWarper
import os
import re
import argparse
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
import json
import math
from contextlib import contextmanager

from utils import (
    plot_budget_token_ranks, 
    plot_hidden_token_ranks,
    plot_attention_weights,
    plot_key_norms,
    plot_heatmap_generic
)
from modules.rope import *

parser = argparse.ArgumentParser()
parser.add_argument("--token_budget", type=int, default=1024)
parser.add_argument("--max_tokens", type=int, default=2048)
parser.add_argument("--temperature", type=float, default=1.1)
parser.add_argument("--top_p", type=float, default=0.95)
parser.add_argument("--prompt_type", type=str, default="livecodebench")
# Token swapping experiments
parser.add_argument("--swap_token", action="store_true", default=False)
parser.add_argument("--truncate_token", action="store_true", default=False)
parser.add_argument("--truncate_offset", type=int, default=1)
# Logit tracing experiments 
parser.add_argument("--output_ranks", action="store_true", default=False)
parser.add_argument("--token_rank_plot_interval", type=int, default=16)
# Attention experiments
parser.add_argument("--output_hidden_states", action="store_true", default=False)
parser.add_argument("--output_attentions", action="store_true", default=False)
parser.add_argument("--num_layers_plot", type=int, default=8)
parser.add_argument("--rope", action="store_true", default=False)
parser.add_argument("--normalize", action="store_true", default=False)
parser.add_argument("--rerope_current_key", type=int, default=0)

def get_prompt(prompt_type):
    with open(f"prompts.json", "r") as f:
        prompts = json.load(f)
    prompt = prompts[prompt_type][0]
    return prompt

def get_logit_rank(logits, token_id):
    sorted_indices = torch.argsort(logits, dim=-1, descending=True)
    token_rank = (sorted_indices == token_id).nonzero(as_tuple=True)[-1].item()
    return token_rank

def capture_attention_hooks(attention_list):
    def hook(module, input, output):
        # capture attn weights from output
        _, attn_weights = output
        attn = attn_weights[0, :, -1, -min(8, attn_weights.shape[-1]):] # (bs, n_heads, q_len, k_len)
        attn = attn.mean(dim=0).float().detach().cpu().numpy() # (k_len,)
        last_two_attns = attention_list[1:]
        attention_list.clear() # overwrite at every step
        attention_list.extend(last_two_attns)
        attention_list.append(attn)
        
    return hook

def calculate_key_norms(keys, n_layers, num_layers_plot): # (bs=1, n_heads, k_len, head_dim)
    key_norms = {}
    for i, key in enumerate(keys):
        key_norm = key[0, ...].norm(dim=-1).norm(dim=0).float().detach().cpu().numpy() # (k_len,)
        key_norms[n_layers - num_layers_plot - 1 + i] = [key_norm]
    return key_norms

def compute_query_vectors(last_n_hidden_states, layer_indices, model):
    query_vectors = []
    for i, layer_idx in enumerate(layer_indices):
        layer = model.model.layers[layer_idx]
        hidden_state = last_n_hidden_states[i]
        hidden_state = layer.input_layernorm(hidden_state)
        attn_module = layer.self_attn
        query_vector = attn_module.q_proj(hidden_state)

        n_q_heads = attn_module.num_attention_heads # (attn_dim,)
        head_dim = query_vector.numel() // n_q_heads
        query_vector = query_vector.view(n_q_heads, head_dim) # (n_q_heads, head_dim)
        query_vectors.append(query_vector)
    return query_vectors

def conpute_scores_from_scratch(last_n_query_vectors, keys, total_steps=None, model=None, normalize=False, rerope_current_key=0):
    keys = torch.stack(keys, dim=0) # (n_layers, bs=1, n_heads, k_len, head_dim=128)
    keys = keys.squeeze(1) # (n_layers, n_heads=10, k_len, head_dim) 

    queries = torch.stack(last_n_query_vectors, dim=0) # (n_layers, n_q_heads, head_dim)

    head_dim = keys.shape[-1]
    k_len = keys.shape[-2]
    
    if total_steps is not None:
        position_ids = total_steps - 1
        cos, sin = model.model.rotary_emb(
            queries.new_zeros(1, 1, head_dim), position_ids=torch.tensor([[position_ids]], device=queries.device)
        )
        q_rot, _ = apply_rotary_pos_emb(queries.unsqueeze(2), queries.unsqueeze(2), cos, sin)
        queries = q_rot.squeeze(2)

    if rerope_current_key != 0:
        rerope_single_key_inplace(keys, model, rerope_current_key, p_old=total_steps + rerope_current_key, p_new=total_steps - 256)

    if normalize:
        keys_norm, queries_norm = F.normalize(keys, dim=-1), F.normalize(queries, dim=-1)
    else:
        keys_norm, queries_norm = keys, queries

    n_layers, n_heads_q, n_heads_k = queries.shape[0], queries.shape[1], keys.shape[1]
    group_size = n_heads_q // n_heads_k
    assert n_heads_q % n_heads_k == 0, f"GQA, q: {n_heads_q}, k: {n_heads_k}"

    # we group the query, not the key, for efficiency :)
    queries_grouped = queries_norm.view(n_layers, n_heads_k, group_size, -1) # (n_layers, n_heads_k=10, group_size=8, head_dim)
    
    logits = torch.einsum(
        'lhgd,lhsd->lhgs', queries_grouped, keys_norm # (n_layers, n_heads_k, group_size, k_len)
    )
    logits = logits.view(n_layers, n_heads_q, -1) # (n_layers, n_heads_q, k_len)
    if not normalize: logits = logits / math.sqrt(head_dim)
    scores = F.softmax(logits, dim=-1)
    scores = scores.mean(dim=1) # compute mean after softmax!!

    scores = scores[:, -8:] # only take last 8 tokens
    return scores

@torch.no_grad()
def rerope_single_key_inplace(keys, model, key_index, p_old=None, p_new=None):    
    target_keys = keys[:, :, key_index, :] # (n_layers, n_heads_k, head_dim)
    target_keys = target_keys.to(torch.float32).unsqueeze(2) # (n_layers, n_heads_k, 1, head_dim)

    cos_old, sin_old = model.model.rotary_emb(
        target_keys.new_zeros(1, 1, target_keys.shape[-1]),
        position_ids=torch.tensor([[p_old]], device=keys.device)
    )
    cos_old, sin_old = cos_old.to(keys.device), sin_old.to(keys.device)

    k_content, _ = apply_rotary_pos_emb(target_keys, target_keys, cos_old, -sin_old)
    
    cos_new, sin_new = model.model.rotary_emb(
        target_keys.new_zeros(1, 1, target_keys.shape[-1]),
        position_ids=torch.tensor([[p_new]], device=keys.device)
    )
    cos_new, sin_new = cos_new.to(keys.device), sin_new.to(keys.device)

    k_content_new, _ = apply_rotary_pos_emb(k_content, k_content, cos_new, sin_new)

    keys[:, :, key_index, :] = k_content_new.squeeze(2).to(keys.dtype)

@contextmanager
def rope_shift(model, delta_steps):
    rotary = model.model.rotary_emb
    orig_forward = rotary.forward

    def shifted_forward(x, position_ids=None, **kw):
        if position_ids is not None:
            position_ids = position_ids + delta_steps
        return orig_forward(x, position_ids=position_ids, **kw)

    rotary.forward = shifted_forward
    try:
        yield
    finally:
        rotary.forward = orig_forward

def main():
    args = parser.parse_args()

    token_budget = args.token_budget
    max_tokens = args.max_tokens
    temperature = args.temperature
    top_p = args.top_p
    prompt_type = args.prompt_type
    truncate_offset = args.truncate_offset
    output_ranks = args.output_ranks
    token_rank_plot_interval = args.token_rank_plot_interval
    output_hidden_states = args.output_hidden_states
    output_attentions = args.output_attentions
    num_layers_plot = args.num_layers_plot
    rope = args.rope
    normalize = args.normalize
    rerope_current_key = args.rerope_current_key

    prompt = get_prompt(prompt_type)

    model_name_or_path = "ByteDance-Seed/Seed-OSS-36B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, device_map="auto",
        attn_implementation="eager" if output_attentions else "sdpa",
        torch_dtype=torch.bfloat16
    )  # You may want to use bfloat16 and/or move to GPU here
    messages = [
        {"role": "user", "content": prompt},
    ]
    tokenized_messages = tokenizer.apply_chat_template(
        messages, 
        tokenize=False,
        add_generation_prompt=True, 
        thinking_budget=token_budget # control the thinking budget
    )
    # print(f"Tokenized messages: {tokenized_messages}")

    generated_ids = tokenizer(tokenized_messages, return_tensors="pt").input_ids.to(model.device)
    prompt_len = generated_ids.shape[1]
    past_key_values = None

    temp_warper = TemperatureLogitsWarper(temperature)
    top_p_warper = TopPLogitsWarper(top_p)

    budget_reflection_start = -1
    budget_reflection_stop = -1
    perform_swap = False
    truncate_token = False

    budget_token_start = 5
    budget_token_stop = 6

    budget_token_ranks = []
    budget_token_actual_indices = []

    if output_attentions:
        hooks = [] 
        layer_indices = list(
            range(model.config.num_hidden_layers - num_layers_plot, model.config.num_hidden_layers)
        )
        attention_dict = {k: [None, None, None] for k in layer_indices}
        for layer_idx in layer_indices:
            hooks.append(model.model.layers[layer_idx].self_attn.register_forward_hook(
                capture_attention_hooks(attention_dict[layer_idx])
            ))

    for step in tqdm(range(max_tokens), desc="Generating tokens"):
        with torch.no_grad():
            output = model(
                generated_ids if step == 0 else generated_ids[:, -1:],
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
            )
        logits = output.logits[:, -1, :]

        if output_ranks:
            budget_token_rank = get_logit_rank(logits, budget_token_start)
            budget_token_ranks.append(budget_token_rank if budget_token_rank < 64 else 64)

        logits = temp_warper(generated_ids, logits)
        logits = top_p_warper(generated_ids, logits)

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        if next_token.item() == budget_token_start: # <seed:cot_budget_reflect>
            budget_reflection_start = step
            budget_token_actual_indices.append(step)
            print(f"COT budget reflection starts at step {step} with total tokens {step + prompt_len + 1}")
            if step > token_budget:
                print("COT budget reflection exceeds the budget")

            if output_hidden_states:
                hidden_states = output.hidden_states
                hidden_token_ranks = []
                for i in range(len(hidden_states)):
                    hidden_logits = model.lm_head(hidden_states[i][0, -1, :])
                    hidden_token_rank = get_logit_rank(hidden_logits, budget_token_start)
                    hidden_token_ranks.append(hidden_token_rank if hidden_token_rank < 64 else 64)

                plot_hidden_token_ranks(hidden_token_ranks, step)

            if output_attentions:
                keys = [
                    output.past_key_values.layers[layer_idx].keys[:, :, -8:, :] # up to current token
                    for layer_idx in layer_indices
                ] # for each element, (bs=1, n_heads, k_len, head_dim)
                key_norms = calculate_key_norms(keys, len(past_key_values.layers), num_layers_plot)
                plot_key_norms(key_norms, step)
                plot_attention_weights(attention_dict, step)

                if output_hidden_states: # hidden state has embedding output at index 0
                    last_n_hidden_states = [
                        hidden_states[layer_idx][0, -1, :] # (hidden_dim,)
                        for layer_idx in layer_indices
                    ] # layer_idx shifts -1, which is what we expect
                    last_n_query_vectors = compute_query_vectors(last_n_hidden_states, layer_indices, model)
                    keys_all = [
                            output.past_key_values.layers[layer_idx].keys
                            for layer_idx in layer_indices
                        ]
                    if rope:
                        scores = conpute_scores_from_scratch(last_n_query_vectors, keys_all, step + prompt_len, model, rerope_current_key=rerope_current_key)
                    else:
                        scores = conpute_scores_from_scratch(last_n_query_vectors, keys_all, normalize=normalize)
                    scores = scores.float().detach().cpu().numpy() # (n_layers, k_len)
                    plot_heatmap_generic(scores, step, layer_indices, "cosine similarity between query and key")

            if args.swap_token:
                perform_swap = not perform_swap
                if perform_swap: # new token, same kv
                    next_token = tokenizer.encode(" ")
                    next_token = torch.tensor(next_token).unsqueeze(0).to(generated_ids.device)
                    print(f"Swapped token to empty space token")

            elif args.truncate_token:
                truncate_token = not truncate_token
                if truncate_token: # new kv, recomputed
                    print(f"Truncate 0 tokens, regenerating at current step {step}")
                    delta = -256
                    if rerope_current_key == -2:
                        abs_prev = prompt_len + step - 1
                        for layer in range(len(past_key_values.layers)):
                            k = past_key_values.layers[layer].keys
                            rerope_single_key_inplace(k, model, key_index=-1, p_old=abs_prev, p_new=abs_prev + delta)

                        with torch.no_grad():
                            output = model(
                                generated_ids if step == 0 else generated_ids[:, -1:],
                                past_key_values=past_key_values,
                                use_cache=True,
                                output_hidden_states=False,
                                output_attentions=False
                            )

                    elif rerope_current_key == -1:
                        past_key_values.crop(prompt_len + step)
                        with torch.no_grad():
                            with rope_shift(model, delta_steps=-256):
                                output = model(
                                    generated_ids if step == 0 else generated_ids[:, -1:],
                                    past_key_values=past_key_values,
                                    use_cache=True,
                                    output_hidden_states=False,
                                    output_attentions=False
                                )
                    else:
                        raise ValueError(f"Invalid rerope_current_key")

                    logits = output.logits[:, -1, :]
                    logits = temp_warper(generated_ids, logits)
                    logits = top_p_warper(generated_ids, logits)
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    past_key_values = output.past_key_values

        elif next_token.item() == budget_token_stop: # </seed:cot_budget_reflect>
            budget_reflection_stop = step
            print(f"COT budget reflection stops at step {step} with total tokens {step + prompt_len + 1}")
            if budget_reflection_start != -1:
                reflection_text = tokenizer.decode(generated_ids[0, prompt_len + budget_reflection_start :])
                print(f"COT budget reflection text: {reflection_text}")
            budget_reflection_start = -1
            budget_reflection_stop = -1

        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        past_key_values = output.past_key_values

        if next_token.item() == tokenizer.eos_token_id:
            break

    output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
    # print(output_text)

    if output_ranks:
        for idx in budget_token_actual_indices:
            start_idx = idx - token_rank_plot_interval // 2
            end_idx = idx + token_rank_plot_interval // 2
            plot_budget_token_ranks(budget_token_ranks, start_idx, end_idx)

    if output_attentions:
        for hook in hooks: hook.remove()

if __name__ == "__main__":
    main()