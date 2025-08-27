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
import json

from utils import plot_budget_token_ranks, plot_hidden_token_ranks

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
parser.add_argument("--token_rank_plot_interval", type=int, default=16)
parser.add_argument("--output_hidden_states", action="store_true", default=False)
args = parser.parse_args()

token_budget = args.token_budget
max_tokens = args.max_tokens
temperature = args.temperature
top_p = args.top_p
prompt_type = args.prompt_type
truncate_offset = args.truncate_offset
token_rank_plot_interval = args.token_rank_plot_interval
output_hidden_states = args.output_hidden_states

with open(f"prompts.json", "r") as f:
    prompts = json.load(f)
prompt = prompts[prompt_type][0]

def get_logit_rank(logits, token_id):
    sorted_indices = torch.argsort(logits, dim=-1, descending=True)
    token_rank = (sorted_indices == token_id).nonzero(as_tuple=True)[-1].item()
    return token_rank

model_name_or_path = "ByteDance-Seed/Seed-OSS-36B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto")  # You may want to use bfloat16 and/or move to GPU here
messages = [
    {"role": "user", "content": prompt},
]
tokenized_messages = tokenizer.apply_chat_template(
  messages, 
  tokenize=False,
  add_generation_prompt=True, 
  thinking_budget=token_budget # control the thinking budget
)
print(f"Tokenized messages: {tokenized_messages}")

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

for step in tqdm(range(max_tokens), desc="Generating tokens"):
    with torch.no_grad():
        output = model(
            generated_ids if step == 0 else generated_ids[:, -1:],
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=output_hidden_states,
        )
    logits = output.logits[:, -1, :]

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

        if args.swap_token:
            perform_swap = not perform_swap
            if perform_swap: # new token, same kv
                next_token = tokenizer.encode(" ")
                next_token = torch.tensor(next_token).unsqueeze(0).to(generated_ids.device)
                print(f"Swapped token to empty space token")

        elif args.truncate_token:
            truncate_token = not truncate_token
            if truncate_token: # new kv, recomputed
                if truncate_offset == 0:
                    print(f"Truncate 0 tokens, regenerating at current step {step}")
                    continue
                generated_ids = generated_ids[:, :-truncate_offset]
                # Only choose -truncate_offset from past_key_values
                past_key_values.crop(prompt_len + step - truncate_offset)

                print(f"Truncated {truncate_offset} tokens, resetting to step {step - truncate_offset}")
                continue

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
print(output_text)

for idx in budget_token_actual_indices:
    start_idx = idx - token_rank_plot_interval // 2
    end_idx = idx + token_rank_plot_interval // 2
    plot_budget_token_ranks(budget_token_ranks, start_idx, end_idx)