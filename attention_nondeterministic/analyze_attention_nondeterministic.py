"""
python analyze_attention_nondeterministic.py \
    --mode [split_kv, prefill] \
    --attn_implementation [flash_attention_2, sdpa] \
    --seq_lens [4096, 6144] \
    --batch_sizes [1, 2, 4, 8] \
    --use_fa2_repo --fa2_deterministic --fa2_split_size 1024
"""

import torch
import random
import argparse
import os
import copy

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3 import modeling_qwen3 as qwen_mod
import flash_attn
import flash_attn_2_cuda as fa2_cuda

def add_common_determinism_flags():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

def print_env_summary():
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"[Env] torch={torch.__version__} | cuda={torch.version.cuda} | device={gpu}")
    try:
        import flash_attn
        ver = getattr(flash_attn, "__version__", "unknown")
        print(f"[Env] flash_attn={ver}")
    except Exception as e:
        print(f"[Env] flash_attn import failed: {e}")

def _enable_fa2():
    orig = qwen_mod.ALL_ATTENTION_FUNCTIONS.get("flash_attention_2")
    if orig is None:
        print(f"[Patch] flash_attention_2 not found for qwen3")
        return
    
    def fa2_wrapper(self, q, k, v, attn_mask, *, dropout, scaling, sliding_window, **kwargs):
        try:
            is_decode = q.dim() == 4 and q.size(2) == 1
            no_mask = attn_mask is None
            if dropout == 0.0 and (sliding_window is None) and is_decode and no_mask:
                k_cache = k[:, :, :-1, :]
                v_cache = v[:, :, :-1, :]
                k_new = k[:, :, -1:, :]
                v_new = v[:, :, -1:, :]

                q_ = q.transpose(1, 2).contiguous() # (B, S, H, D) for fa2
                k_cache = k_cache.transpose(1, 2).contiguous()
                v_cache = v_cache.transpose(1, 2).contiguous()
                k_new = k_new.transpose(1, 2).contiguous()
                v_new = v_new.transpose(1, 2).contiguous()

                # print(f"[Patch] with_kvcache used: B={q.size(0)} S_cache={k_cache.size(1)}")

                out = flash_attn.flash_attn_with_kvcache(
                    q=q_, k_cache=k_cache, v_cache=v_cache, k=k_new, v=v_new,
                    cache_seqlens=k_cache.size(1), softmax_scale=scaling, causal=True,
                    num_splits=0
                )
                return out, None
            else:
                return orig(
                    self, q, k, v, attn_mask, dropout=dropout, scaling=scaling, sliding_window=sliding_window, **kwargs
                )
        except Exception as e:
            print(f"[Patch] error in patched flash_attention_2: {e}")
            return orig(
                self, q, k, v, attn_mask, dropout=dropout, scaling=scaling, sliding_window=sliding_window, **kwargs
            )

    qwen_mod.ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = fa2_wrapper

def configure_fa2_deterministic(enabled, split_size):
    os.environ["FA2_DETERMINISTIC"] = "1" if enabled else "0"
    if split_size is not None:
        os.environ["FA2_SPLIT_SIZE"] = str(int(split_size))
    
    try:
        import flash_attn
        print(f"[Env] flash_attn file ={flash_attn.__file__}")
        flash_attn.set_deterministic_mode(enabled=enabled, split_size=split_size)
    except Exception as e:
        print(f"[Warning] Could not call flash_attn.set_deterministic_mode: {e} (falling back to stock implementation)")

def load_model(model_id, attn_implementation, dtype):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    torch_dtype = getattr(torch, dtype)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch_dtype,
        device_map="auto",
        attn_implementation=attn_implementation,
    )
    model.eval()
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    return tokenizer, model

def make_tokens(tokenizer, target_len):
    text = "Mitochondria is the powerhouse of the cell."
    text = (text * (target_len // len(tokenizer.encode(text)) + 50))[:]
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    if ids.numel() >= target_len:
        return ids[:target_len].unsqueeze(0)
    pad = tokenizer.pad_token_id
    ids = torch.cat([ids, ids.new_full((target_len - ids.numel(),), pad)])
    return ids.unsqueeze(0)

def diff_stats(a, b):
    a = a.detach().float().reshape(-1)
    b = b.detach().float().reshape(-1)
    d = (a-b).abs()
    denom = a.norm().item() or 1.0
    rel_l2 = (a-b).norm().item() / denom
    return {
        "max_abs": float(d.max().item()),
        "mean_abs": float(d.mean().item()),
        "rel_l2": float(rel_l2),
    }

@torch.no_grad()
def last_logits_oneshot_prefill(model, ids):
    m = torch.ones_like(ids, dtype=torch.long, device=model.device)
    out = model(ids.to(model.device), attention_mask=m.to(model.device), use_cache=True)
    return out.logits[:, -1, :]

@torch.no_grad()
def last_logits_full(model, seqs):
    pad_id = model.config.pad_token_id
    lengths = [t.shape[1] for t in seqs] # precompute to get token pos
    max_len = max(lengths)

    padded = []
    masks = []
    for t, l in zip(seqs, lengths):
        pad = max_len - l
        if pad:
            t = torch.cat([t, t.new_full((1, pad), pad_id)], dim=1)
        padded.append(t)
        mask = torch.zeros_like(t, dtype=torch.long, device=model.device)
        mask[:, :l] = 1
        masks.append(mask)
    X = torch.cat(padded, dim=0).to(model.device)
    M = torch.cat(masks, dim=0).to(model.device)
    out = model(X, attention_mask=M).logits

    B = len(seqs)
    idx = torch.tensor([l-1 for l in lengths], device=out.device)
    rows = torch.arange(B, device=out.device)
    last_logits = out[rows, idx, :]
    return last_logits

@torch.no_grad()
def last_logits_chunked_prefill(model, ids, chunk_size):
    ids = ids.to(model.device)
    B, L = ids.shape
    
    past = None
    pos = 0
    last_logits = None

    while pos < L:
        cur = ids[:, pos: min(pos + chunk_size, L)]
        cur_len = cur.size(1)
        cum_len = pos + cur_len
        attn = torch.ones((B, cum_len), dtype=torch.long, device=model.device)
        
        out = model(
            input_ids=cur,
            attention_mask=attn,
            use_cache=True,
            past_key_values=past,
        )
        past = out.past_key_values
        last_logits = out.logits[:, -1, :]
        pos += cur_len
    return last_logits

def experiment_split_kv(model_id, seq_lens, batch_sizes, attn_implementation, dtype, layer_idx=-1):
    add_common_determinism_flags()
    tokenizer, model = load_model(model_id, attn_implementation, dtype)
    device = model.device

    for L in seq_lens:
        target = make_tokens(tokenizer, L)
        pre_t = target[:, :-1].to(device)
        attn_t = torch.ones_like(pre_t, dtype=torch.long, device=device)
        out_t = model(pre_t, attention_mask=attn_t, use_cache=True)
        past_t_cache = out_t.past_key_values
        frozen_past_t = past_t_cache.to_legacy_cache()
        k_cache_t, v_cache_t = frozen_past_t[layer_idx]

        q_cap = {"q": None, "k_new": None, "v_new": None, "scale": None}
        attn_mod = model.model.layers[layer_idx].self_attn

        def pre_hook(module, args, kwargs):
            """A hook to capture the q, k_new, and v_new at the single decoding step"""
            hidden_states = kwargs.get("hidden_states")
            cos, sin = kwargs.get("position_embeddings")
            B, S, _ = hidden_states.shape
            Hq, Hkv, D = module.config.num_attention_heads, module.config.num_key_value_heads, module.head_dim
            q = module.q_proj(hidden_states).view(B, S, Hq, D)
            k = module.k_proj(hidden_states).view(B, S, Hkv, D)
            v = module.v_proj(hidden_states).view(B, S, Hkv, D)

            q = module.q_norm(q).transpose(1, 2) # (B, Hq, S, D)
            k = module.k_norm(k).transpose(1, 2) # (B, Hkv, S, D)
            v = v.transpose(1, 2)

            q, k = qwen_mod.apply_rotary_pos_emb(q, k, cos, sin)
            q_cap["q"] = q.detach().contiguous()
            q_cap["k_new"] = k.detach().contiguous()
            q_cap["v_new"] = v.detach().contiguous()
            q_cap["scale"] = float(module.scaling)

        h = attn_mod.register_forward_pre_hook(pre_hook, with_kwargs=True)
        try:
            cur_t = target[:, -1:].to(device)
            _ = model(cur_t, attention_mask=None, past_key_values=past_t_cache, use_cache=True)
        finally:
            h.remove()

        # Transpose for FA
        q_t = q_cap["q"].transpose(1, 2).contiguous()
        k_new_t = q_cap["k_new"].transpose(1, 2).contiguous()
        v_new_t = q_cap["v_new"].transpose(1, 2).contiguous()
        scale = q_cap["scale"]
        cache_len_1 = torch.tensor([k_cache_t.size(2)], dtype=torch.int32, device=device)
        assert q_t.size(1) == k_new_t.size(1) == v_new_t.size(1) == 1, f"q_t, k_new_t, v_new_t must have seq_len=1, but get: q_t={q_t.size(1)}, k_new_t={k_new_t.size(1)}, v_new_t={v_new_t.size(1)}"

        out_ref = flash_attn.flash_attn_with_kvcache(
            q=q_t, 
            k_cache=k_cache_t.transpose(1, 2).contiguous(),
            v_cache=v_cache_t.transpose(1, 2).contiguous(),
            k=k_new_t,
            v=v_new_t,
            cache_seqlens=cache_len_1, softmax_scale=scale, causal=True, 
            num_splits=0
        )
        ref_ctx = out_ref[0:1, -1, :, :].reshape(1, -1).detach() # (1, Hq * D)

        for B in batch_sizes:
            if B == 1:
                stats = diff_stats(ref_ctx, ref_ctx)
                print(f"[AttnKernel] B={B:2d} seq_len={L:4d} -> "
                    f"max_abs={stats['max_abs']:.8e} | mean_abs={stats['mean_abs']:.8e} | rel_l2={stats['rel_l2']:.8e}")
                continue

            q_b = q_t.expand(B, -1, -1, -1).contiguous().clone()
            k_b = k_new_t.expand(B, -1, -1, -1).contiguous().clone()
            v_b = v_new_t.expand(B, -1, -1, -1).contiguous().clone()
            k_cache_b = k_cache_t.transpose(1, 2).contiguous().expand(B, -1, -1, -1).contiguous().clone()
            v_cache_b = v_cache_t.transpose(1, 2).contiguous().expand(B, -1, -1, -1).contiguous().clone()
            cache_lens = torch.full((B,), k_cache_t.size(2), dtype=torch.int32, device=device)

            out_b = flash_attn.flash_attn_with_kvcache(
                q=q_b, k_cache=k_cache_b, v_cache=v_cache_b, k=k_b, v=v_b,
                cache_seqlens=cache_lens, softmax_scale=scale, causal=True,
                num_splits=0
            )
            # assert torch.allclose(out_b[0], out_b[1], atol=0, rtol=0), f"Rows differ at B={B}; inputs are not identical"
            cur_ctx = out_b[0:1, -1, :, :].reshape(1, -1).detach()
            stats = diff_stats(ref_ctx, cur_ctx)
            print(f"[AttnKernel] B={B:2d} seq_len={L:4d} -> "
                    f"max_abs={stats['max_abs']:.8e} | mean_abs={stats['mean_abs']:.8e} | rel_l2={stats['rel_l2']:.8e}")

def experiment_prefill(model_id, seq_len, chunk_sizes, attn_implementation, dtype):
    add_common_determinism_flags()

    tokenizer, model = load_model(model_id, attn_implementation, dtype)
    target = make_tokens(tokenizer, seq_len)
    ref = last_logits_oneshot_prefill(model, target)

    for chunk_size in chunk_sizes:
        cur = last_logits_chunked_prefill(model, target, chunk_size)
        stats = diff_stats(ref, cur)
        print(f"[Prefill] chunk_size={chunk_size:4d}, max_abs={stats['max_abs']:.8e} | "
            f"mean_abs={stats['mean_abs']:.8e} | rel_l2={stats['rel_l2']:.8e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--attn_implementation", choices=['flash_attention_2', 'sdpa'], default="flash_attention_2")
    parser.add_argument("--mode", choices=["split_kv", "prefill"], default="split_kv")
    parser.add_argument("--dtype", choices=['float16', 'float32', 'bfloat16'], default='float16')
    # prefill
    parser.add_argument("--chunk_sizes", default="512,256,128,32")
    parser.add_argument("--seq_len", type=int, default=512)
    # split_kv
    parser.add_argument("--seq_lens", default="4096,6144")
    parser.add_argument("--batch_sizes", default="1,2,4,8,16")
    # fa2 deterministic
    parser.add_argument("--use_fa2_repo", action="store_true")
    parser.add_argument("--fa2_deterministic", action="store_true")
    parser.add_argument("--fa2_split_size", type=int, default=None)
    parser.add_argument("--fa2_verbose", action="store_true")
    args = parser.parse_args()

    if args.use_fa2_repo:
        assert args.attn_implementation == "flash_attention_2", f"[Error] --use_fa2_repo requested but attn_implementation is not flash_attention_2"

        configure_fa2_deterministic(
            enabled=args.fa2_deterministic,
            split_size=args.fa2_split_size,
        )
        _enable_fa2()

        if args.fa2_verbose:
            print_env_summary()

    with torch.no_grad():
        if args.mode == "prefill":
            chunk_sizes = [int(s) for s in args.chunk_sizes.split(",") if s.strip()]
            experiment_prefill(args.model, args.seq_len, chunk_sizes, args.attn_implementation, args.dtype)
        elif args.mode == "split_kv":
            seq_lens = [int(x) for x in args.seq_lens.split(",") if x.strip()]
            batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
            experiment_split_kv(args.model, seq_lens, batch_sizes, args.attn_implementation, args.dtype)

if __name__ == "__main__":
    main()