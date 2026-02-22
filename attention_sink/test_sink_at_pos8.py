#!/usr/bin/env python3
"""
Test: Place top-sink tokens at position 8 instead of position 0.
Question: Does the sink follow the token or stay at position 0?

For each top-K token:
  - Replace position 8 with that token (keep position 0 natural)
  - Measure attention to position 0 vs position 8 across all layers/heads
  - Plot comparison: attn_to_pos0 vs attn_to_pos8

Usage:
  CUDA_VISIBLE_DEVICES=4 python test_sink_at_pos8.py \
    --model Qwen/Qwen3-8B \
    --prompt-file prompt_sets_v512_t32/natural_mixed.txt \
    --outdir results/exp_top20_at_pos8
"""

import argparse, os, sys, math, random, time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

EPS = 1e-12

def _disable_packed_sequence_splitting():
    try:
        import transformers.masking_utils as _mu
        _mu.find_packed_sequence_indices = lambda pos, *a, **k: torch.zeros_like(pos, dtype=torch.long)
    except Exception:
        pass

@torch.no_grad()
def _forward(model, input_ids, position_ids):
    device = next(model.parameters()).device
    out = model(
        input_ids=input_ids.to(device),
        position_ids=position_ids.to(device),
        use_cache=False,
        output_attentions=True,
        output_hidden_states=False,
    )
    return out.attentions  # tuple of [B, H, Q, K]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--prompt-file", required=True)
    p.add_argument("--n-prompts", type=int, default=32)
    p.add_argument("--target-pos", type=int, default=8, help="Position to place the top-sink token")
    p.add_argument("--q-start", type=int, default=4)
    p.add_argument("--top-token-ids", type=int, nargs="+",
                   default=[2, 1722, 565, 33975, 14374, 3838, 279, 3014, 40, 322,
                            4340, 27, 32, 888],
                   help="Token IDs of top-sink tokens to test at target-pos")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", default="results/exp_top20_at_pos8")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    _disable_packed_sequence_splitting()

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dt, device_map="auto", attn_implementation="eager"
    )
    model.eval()

    num_layers = model.config.num_hidden_layers

    # Load prompts
    with open(args.prompt_file, "r", encoding="utf-8") as f:
        all_lines = [ln.strip() for ln in f if ln.strip()]
    n = min(args.n_prompts, len(all_lines))
    prompts = random.sample(all_lines, n)
    print(f"[Init] Using {n} prompts, target_pos={args.target_pos}")

    # Tokenize
    tokenized = []
    for text in prompts:
        ids = tok(text, return_tensors="pt", add_special_tokens=True)["input_ids"]
        S = ids.shape[1]
        if S < args.target_pos + 1:
            continue  # skip if too short
        pos = torch.arange(S, dtype=torch.long).unsqueeze(0)
        tokenized.append((ids, pos))
    print(f"[Init] {len(tokenized)} prompts after filtering")

    target_pos = args.target_pos

    # ── Baseline: measure attention to pos0 and pos8 with NO replacement ──
    print("\n[Baseline] Measuring natural attention to pos0 and pos8...")
    baseline_pos0 = []  # per-layer mean attn to pos0
    baseline_pos8 = []  # per-layer mean attn to pos8

    layer_attn_pos0 = {L: [] for L in range(num_layers)}
    layer_attn_pos8 = {L: [] for L in range(num_layers)}

    for ids, pos in tqdm(tokenized, desc="Baseline"):
        attns = _forward(model, ids, pos)
        for L in range(num_layers):
            a = attns[L][0].float()  # [H, Q, K]
            # mean attn from q>=q_start to k=0 and k=target_pos
            attn_to_0 = a[:, args.q_start:, 0].mean().item()
            attn_to_8 = a[:, args.q_start:, target_pos].mean().item()
            layer_attn_pos0[L].append(attn_to_0)
            layer_attn_pos8[L].append(attn_to_8)

    baseline = {
        "pos0": [float(np.mean(layer_attn_pos0[L])) for L in range(num_layers)],
        "pos8": [float(np.mean(layer_attn_pos8[L])) for L in range(num_layers)],
    }

    # ── For each top token: replace pos8 with it, measure attention ──
    results = {}  # token_id -> {"pos0": [...], "pos8": [...]}

    for tid in args.top_token_ids:
        tok_str = tok.decode([tid])
        print(f"\n[Token {tid}] Replacing pos{target_pos} with {repr(tok_str)}...")

        layer_attn_pos0_t = {L: [] for L in range(num_layers)}
        layer_attn_pos8_t = {L: [] for L in range(num_layers)}

        for ids, pos in tqdm(tokenized, desc=f"tok={repr(tok_str)}", leave=False):
            ids2 = ids.clone()
            ids2[0, target_pos] = tid  # replace position 8

            attns = _forward(model, ids2, pos)
            for L in range(num_layers):
                a = attns[L][0].float()
                attn_to_0 = a[:, args.q_start:, 0].mean().item()
                attn_to_8 = a[:, args.q_start:, target_pos].mean().item()
                layer_attn_pos0_t[L].append(attn_to_0)
                layer_attn_pos8_t[L].append(attn_to_8)

        results[tid] = {
            "token": tok_str,
            "pos0": [float(np.mean(layer_attn_pos0_t[L])) for L in range(num_layers)],
            "pos8": [float(np.mean(layer_attn_pos8_t[L])) for L in range(num_layers)],
        }

    # ── Plot ──
    layers = list(range(num_layers))

    # Figure 1: Baseline attn to pos0 vs pos8
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(layers, baseline["pos0"], "b-o", markersize=4, label="Baseline: attn to pos0")
    ax.plot(layers, baseline["pos8"], "r--s", markersize=4, label=f"Baseline: attn to pos{target_pos}")
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Attn(Q[q>=4] -> K[pos])", fontsize=12)
    ax.set_title(f"Baseline: attention to pos0 vs pos{target_pos} (no replacement)", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "baseline_pos0_vs_pos8.png"), dpi=300)
    plt.close()

    # Figure 2: For each top token at pos8, does sink move?
    fig, axes = plt.subplots(2, 1, figsize=(16, 12), sharex=True)

    ax_pos0 = axes[0]
    ax_pos8 = axes[1]

    # Plot baseline as reference
    ax_pos0.plot(layers, baseline["pos0"], "k--", linewidth=2, alpha=0.5, label="baseline (no replace)")
    ax_pos8.plot(layers, baseline["pos8"], "k--", linewidth=2, alpha=0.5, label="baseline (no replace)")

    cmap = plt.cm.tab20
    for i, tid in enumerate(args.top_token_ids):
        r = results[tid]
        color = cmap(i / len(args.top_token_ids))
        label = f"{repr(r['token'])} (id={tid})"
        ax_pos0.plot(layers, r["pos0"], marker="o", markersize=3, linewidth=0.8, color=color, label=label)
        ax_pos8.plot(layers, r["pos8"], marker="o", markersize=3, linewidth=0.8, color=color, label=label)

    ax_pos0.set_ylabel("Attn to pos0", fontsize=12)
    ax_pos0.set_title(f"Attention to pos0 when top-sink tokens are placed at pos{target_pos}", fontsize=13)
    ax_pos0.legend(fontsize=7, ncol=3, loc="upper left")
    ax_pos0.set_ylim(0, 1)
    ax_pos0.grid(True, alpha=0.3)

    ax_pos8.set_ylabel(f"Attn to pos{target_pos}", fontsize=12)
    ax_pos8.set_xlabel("Layer", fontsize=12)
    ax_pos8.set_title(f"Attention to pos{target_pos} (where top-sink token was placed)", fontsize=13)
    ax_pos8.legend(fontsize=7, ncol=3, loc="upper left")
    ax_pos8.set_ylim(0, 1)
    ax_pos8.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "top_tokens_at_pos8.png"), dpi=300)
    plt.close()

    # Figure 3: Summary bar chart - mean attn to pos0 vs pos8 (averaged over all layers)
    fig, ax = plt.subplots(figsize=(14, 7))
    token_labels = [repr(results[tid]["token"]) for tid in args.top_token_ids]
    mean_pos0 = [float(np.mean(results[tid]["pos0"])) for tid in args.top_token_ids]
    mean_pos8 = [float(np.mean(results[tid]["pos8"])) for tid in args.top_token_ids]

    x = np.arange(len(args.top_token_ids))
    w = 0.35
    ax.bar(x - w/2, mean_pos0, w, label="Attn to pos0 (original sink)", color="steelblue", alpha=0.8)
    ax.bar(x + w/2, mean_pos8, w, label=f"Attn to pos{target_pos} (where token placed)", color="darkorange", alpha=0.8)

    # Baseline reference lines
    ax.axhline(float(np.mean(baseline["pos0"])), color="blue", linestyle="--", alpha=0.4, label="baseline pos0")
    ax.axhline(float(np.mean(baseline["pos8"])), color="red", linestyle="--", alpha=0.4, label=f"baseline pos{target_pos}")

    ax.set_xticks(x)
    ax.set_xticklabels(token_labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Mean attn (over all layers)", fontsize=12)
    ax.set_title(f"Does sink follow the token to pos{target_pos}?", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "summary_bar_pos0_vs_pos8.png"), dpi=300)
    plt.close()

    print(f"\n[Done] Results in {args.outdir}/")
    print(f"  - baseline_pos0_vs_pos8.png")
    print(f"  - top_tokens_at_pos8.png")
    print(f"  - summary_bar_pos0_vs_pos8.png")


if __name__ == "__main__":
    main()

