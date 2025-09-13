from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import random
import argparse

def add_common_determinism_flags():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

def load_model(model_id, attn_implementation, dtype):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    torch_dtype = getattr(torch, dtype)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
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
        attn = torch.ones((1, cum_len), dtype=torch.long, device=model.device)
        
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

def experiment_batch(model_id, seq_len, runs, attn_implementation, dtype):
    add_common_determinism_flags()

    tokenizer, model = load_model(model_id, attn_implementation, dtype)
    target = make_tokens(tokenizer, seq_len)
    ref = last_logits_oneshot_prefill(model, target)

    lengths = [63, 127, 255, 511,1023]
    batch_sizes = [1, 2, 4, 8, 16, 32]

    for i in range(runs):
        B = batch_sizes[i % len(batch_sizes)]
        distractors = [
            make_tokens(tokenizer, max(4, lengths[(i+j) % len(lengths)]))
            for j in range(max(0, B-1))
        ]
        pos = (i * 2) % max(1, B)
        seqs = distractors[:pos] + [target] + distractors[pos:]
        logits = last_logits_full(model, seqs)
        cur = logits[pos:pos+1, :]
        
        stats = diff_stats(ref, cur)
        print(f"[Batch] Run {i+1:02d}: B={len(seqs):2d} target_pos={pos:2d} -> max_abs={stats['max_abs']:.8e} | mean_abs={stats['mean_abs']:.8e} | rel_l2={stats['rel_l2']:.8e}")

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
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--runs", type=int, default=6)
    parser.add_argument("--attn_implementation", choices=['flash_attention_2', 'sdpa'], default="flash_attention_2")
    parser.add_argument("--mode", choices=["batch", "prefill"], default="batch")
    parser.add_argument("--chunk_sizes", default="512,256,128,32")
    parser.add_argument("--dtype", choices=['float16', 'float32', 'bfloat16'], default='float16')
    args = parser.parse_args()

    with torch.no_grad():
        if args.mode == "batch":
            experiment_batch(args.model, args.seq_len, args.runs, args.attn_implementation, args.dtype)
        else:
            chunk_sizes = [int(s) for s in args.chunk_sizes.split(",") if s.strip()]
            experiment_prefill(args.model, args.seq_len, chunk_sizes, args.attn_implementation, args.dtype)

if __name__ == "__main__":
    main()