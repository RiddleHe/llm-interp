#!/usr/bin/env python3
"""
Vocab Sink Sweep: For every token in the vocabulary, place it at position 0
of a fixed context and measure the attention-sink score.

Sink score = mean attention(q>=4 -> k=0) averaged over all layers and heads.

Produces 3 figures:
  1. Histogram of sink scores (with special-token markers)
  2. Rank curve (sorted sink scores, annotated top/bottom tokens)
  3. Category box plot (special / CJK / Latin / digit / punct / code / other)

Usage:
  # Quick validation (1 prompt, single GPU):
  CUDA_VISIBLE_DEVICES=0 python vocab_sink_sweep.py --model Qwen/Qwen3-8B \
      --prompt-file prompt_sets_v512_t32/natural_mixed.txt --n-prompts 1 --outdir results_vocab

  # Full run (20 prompts, multi-GPU data-parallel):
  python vocab_sink_sweep.py --model Qwen/Qwen3-8B \
      --prompt-file prompt_sets_v512_t32/natural_mixed.txt --n-prompts 20 \
      --gpus 0,1,2,3 --outdir results_vocab
"""

import argparse, os, sys, json, math, random, time
import numpy as np
import torch
import torch.multiprocessing as mp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import unicodedata
import re

# ─────────────────────────── helpers ───────────────────────────

def _disable_packed_sequence_splitting():
    try:
        import transformers.masking_utils as _mu
        _mu.find_packed_sequence_indices = lambda pos, *a, **k: torch.zeros_like(pos, dtype=torch.long)
        print("[transformers] Disabled packed-sequence split.")
    except Exception as e:
        print(f"[transformers] patch skipped: {e}")


def _load_model(model_name, device, dtype="bf16"):
    dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dt, device_map={"": device}, attn_implementation="eager"
    )
    model.eval()
    return model


@torch.no_grad()
def _forward_batch(model, input_ids_batch, position_ids_batch):
    """Run forward, return attentions tuple. input_ids_batch: [B, S]"""
    device = next(model.parameters()).device
    out = model(
        input_ids=input_ids_batch.to(device),
        position_ids=position_ids_batch.to(device),
        use_cache=False,
        output_attentions=True,
        output_hidden_states=False,
    )
    return out.attentions  # tuple of [B, H, S, S] per layer


def _compute_sink_scores_from_attns(attns, q_start=4):
    """
    attns: tuple of L tensors, each [B, H, S, S].
    Returns: [B] tensor of sink scores = mean attn(q>=q_start -> k=0) over layers & heads.
    """
    # Stack: [L, B, H, S, S]
    A = torch.stack([a.detach().float() for a in attns], dim=0)
    # A[:, :, :, q_start:, 0] -> [L, B, H, Q']
    sink_attn = A[:, :, :, q_start:, 0]
    # mean over L, H, Q' -> [B]
    return sink_attn.mean(dim=(0, 2, 3)).cpu()


# ─────────────────────────── token categorization ───────────────────────────

def _categorize_token(tok_str, token_id, special_ids):
    """Assign a category string to a token."""
    if token_id in special_ids:
        return "special"
    if not tok_str or tok_str.isspace():
        return "whitespace"

    # strip leading special char (Ġ for space, etc.)
    s = tok_str.replace("Ġ", "").replace("▁", "").strip()
    if not s:
        return "whitespace"

    # Check dominant character type
    cats = [unicodedata.category(c) for c in s if not c.isspace()]
    if not cats:
        return "whitespace"

    # CJK
    if any(("\u4e00" <= c <= "\u9fff" or "\u3400" <= c <= "\u4dbf" or
            "\U00020000" <= c <= "\U0002a6df" or "\u3000" <= c <= "\u303f" or
            "\u3040" <= c <= "\u309f" or "\u30a0" <= c <= "\u30ff" or
            "\uac00" <= c <= "\ud7af") for c in s):
        return "CJK"

    if all(c.isdigit() for c in s):
        return "digit"

    if all(c.isalpha() for c in s):
        return "Latin"

    if all(c in "+-*/=<>!@#$%^&()[]{}|\\:;'\",.<>?/~`_" for c in s):
        return "punct/sym"

    # Math / LaTeX
    if any(c == '\\' for c in s) or any(c in "∑∏∫√∂∇" for c in s):
        return "math/LaTeX"

    # Code-like
    if re.search(r'[{}();=]', s) and any(c.isalpha() for c in s):
        return "code"

    return "other"


# ─────────────────────────── worker (per-GPU) ───────────────────────────

def _worker_run(gpu_id, model_name, token_ids_chunk, context_ids_list, q_start, result_dict, worker_idx, dtype):
    """
    Each worker loads its own model on a specific GPU,
    sweeps through its chunk of token IDs, and writes results to result_dict.
    """
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)
    _disable_packed_sequence_splitting()
    model = _load_model(model_name, device, dtype)
    print(f"[Worker {worker_idx}] Model loaded on {device}, chunk size={len(token_ids_chunk)}")

    n_contexts = len(context_ids_list)
    seq_len = context_ids_list[0].shape[1]  # [1, S]
    batch_size = 64  # sequences per batch

    # For each token, accumulate sink scores across contexts
    all_scores = np.zeros((len(token_ids_chunk), n_contexts), dtype=np.float32)

    for ci, ctx_ids in enumerate(context_ids_list):
        # ctx_ids: [1, S], we replace position 0 with each vocab token
        base = ctx_ids[0].clone()  # [S]

        # Process in batches
        for start in tqdm(
            range(0, len(token_ids_chunk), batch_size),
            desc=f"[GPU{gpu_id}] ctx={ci+1}/{n_contexts}",
            leave=False,
            position=worker_idx,
        ):
            end = min(start + batch_size, len(token_ids_chunk))
            bsz = end - start
            batch_ids = base.unsqueeze(0).expand(bsz, -1).clone()  # [bsz, S]
            for i, tid in enumerate(token_ids_chunk[start:end]):
                batch_ids[i, 0] = tid

            pos_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(bsz, -1)
            attns = _forward_batch(model, batch_ids, pos_ids)
            scores = _compute_sink_scores_from_attns(attns, q_start=q_start)  # [bsz]
            all_scores[start:end, ci] = scores.numpy()

    # Mean and std across contexts
    means = all_scores.mean(axis=1)  # [N_chunk]
    stds = all_scores.std(axis=1)    # [N_chunk]

    result_dict[worker_idx] = {
        "token_ids": token_ids_chunk,
        "means": means,
        "stds": stds,
    }
    print(f"[Worker {worker_idx}] Done. scores range: [{means.min():.4f}, {means.max():.4f}]")


# ─────────────────────────── plotting ───────────────────────────

def _plot_histogram(scores, special_mask, special_labels, outdir):
    """Figure 1: Histogram zoomed to data range + special token table."""
    # Compute data range for zoom
    p01, p99 = float(np.percentile(scores, 0.5)), float(np.percentile(scores, 99.5))
    margin = (p99 - p01) * 0.15
    xlo, xhi = p01 - margin, p99 + margin

    fig, (ax_main, ax_full) = plt.subplots(1, 2, figsize=(16, 6),
                                            gridspec_kw={"width_ratios": [3, 1]})

    # ── Left: zoomed histogram ──
    n_bins = 150
    ax_main.hist(scores, bins=n_bins, range=(xlo, xhi), alpha=0.8, color="steelblue", edgecolor="white", linewidth=0.3)

    # Overall mean line
    mu = float(np.mean(scores))
    ax_main.axvline(mu, color="red", linestyle="-", linewidth=1.5, alpha=0.8, label=f"mean = {mu:.4f}")

    # Special token score lines (only a few, with distinct colors)
    sp_scores = [(tid, scores[tid], lbl) for tid, lbl in special_labels.items()]
    sp_scores.sort(key=lambda x: -x[1])
    for i, (tid, sc, lbl) in enumerate(sp_scores[:5]):  # only top 5 special
        ax_main.axvline(sc, color=f"C{i+1}", linestyle="--", linewidth=1.2, alpha=0.7)

    ax_main.set_xlabel("Sink score", fontsize=12)
    ax_main.set_ylabel("Number of tokens", fontsize=12)
    ax_main.set_title("Distribution of sink scores (zoomed to data range)", fontsize=13)
    ax_main.legend(fontsize=10, loc="upper left")
    ax_main.grid(True, alpha=0.3)
    ax_main.set_xlim(xlo, xhi)

    # ── Right: special token table ──
    ax_full.axis("off")
    table_data = []
    for tid, sc, lbl in sp_scores:
        table_data.append([lbl, f"{sc:.4f}"])
    table_data.append(["─── stats ───", ""])
    table_data.append(["all mean", f"{float(np.mean(scores)):.4f}"])
    table_data.append(["all std", f"{float(np.std(scores)):.4f}"])
    table_data.append(["special mean", f"{float(np.mean(scores[special_mask])):.4f}"])
    table_data.append(["normal mean", f"{float(np.mean(scores[~special_mask])):.4f}"])

    tbl = ax_full.table(cellText=table_data, colLabels=["Token", "Score"],
                        loc="center", cellLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.2)
    ax_full.set_title("Special tokens & stats", fontsize=11, pad=10)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "vocab_sink_histogram.png"), dpi=300)
    plt.close(fig)
    print(f"[Plot] Saved vocab_sink_histogram.png")


def _plot_rank_curve(scores, token_strs, special_mask, outdir, top_k=20, bottom_k=10):
    """Figure 2: Rank curve with inset zooms + side tables for top/bottom tokens."""
    order = np.argsort(-scores)  # descending
    sorted_scores = scores[order]
    N = len(scores)

    fig = plt.figure(figsize=(18, 7))
    # Layout: main plot (left 60%), top table (right-top 40%), bottom table (right-bottom 40%)
    gs = fig.add_gridspec(2, 2, width_ratios=[3, 1.3], hspace=0.35, wspace=0.25)
    ax_main = fig.add_subplot(gs[:, 0])
    ax_top = fig.add_subplot(gs[0, 1])
    ax_bot = fig.add_subplot(gs[1, 1])

    x = np.arange(N)

    # ── Main rank curve ──
    ax_main.plot(x, sorted_scores, linewidth=0.6, color="steelblue", alpha=0.9)

    # Highlight special tokens on the curve
    for i in range(N):
        tid = int(order[i])
        if special_mask[tid]:
            ax_main.plot(i, sorted_scores[i], "r*", markersize=8, zorder=5)

    # Mean line
    mu = float(np.mean(scores))
    ax_main.axhline(mu, color="red", linestyle="--", linewidth=1.0, alpha=0.6, label=f"mean = {mu:.4f}")

    ax_main.set_xlabel("Rank (by sink score, descending)", fontsize=12)
    ax_main.set_ylabel("Sink score", fontsize=12)
    ax_main.set_title(f"Rank curve: {N:,} vocab tokens sorted by sink score", fontsize=13)
    ax_main.legend(fontsize=10, loc="upper right")
    ax_main.grid(True, alpha=0.3)
    ax_main.set_xlim(-N * 0.01, N * 1.01)

    # ── Top-K table ──
    ax_top.axis("off")
    top_data = []
    for i in range(min(top_k, N)):
        tid = int(order[i])
        sp = "★" if special_mask[tid] else ""
        tok = repr(token_strs[tid])
        if len(tok) > 18:
            tok = tok[:15] + "..."
        top_data.append([f"{i+1}", tok, f"{sorted_scores[i]:.4f}", sp])

    tbl_top = ax_top.table(cellText=top_data,
                           colLabels=["#", "Token", "Score", ""],
                           loc="center", cellLoc="left")
    tbl_top.auto_set_font_size(False)
    tbl_top.set_fontsize(7)
    tbl_top.scale(1.0, 1.05)
    ax_top.set_title(f"Top {top_k} tokens (highest sink)", fontsize=10, color="darkorange", fontweight="bold")

    # ── Bottom-K table ──
    ax_bot.axis("off")
    bot_data = []
    for i in range(min(bottom_k, N)):
        idx = N - bottom_k + i
        tid = int(order[idx])
        sp = "★" if special_mask[tid] else ""
        tok = repr(token_strs[tid])
        if len(tok) > 18:
            tok = tok[:15] + "..."
        bot_data.append([f"{idx+1}", tok, f"{sorted_scores[idx]:.4f}", sp])

    tbl_bot = ax_bot.table(cellText=bot_data,
                           colLabels=["#", "Token", "Score", ""],
                           loc="center", cellLoc="left")
    tbl_bot.auto_set_font_size(False)
    tbl_bot.set_fontsize(7)
    tbl_bot.scale(1.0, 1.05)
    ax_bot.set_title(f"Bottom {bottom_k} tokens (lowest sink)", fontsize=10, color="gray", fontweight="bold")

    fig.savefig(os.path.join(outdir, "vocab_sink_rank_curve.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved vocab_sink_rank_curve.png")


def _plot_category_boxplot(scores, categories, outdir):
    """Figure 3: Box plot + strip (jittered dots for small categories) + global mean line."""
    cat_names = sorted(set(categories))
    data = [scores[np.array(categories) == c] for c in cat_names]
    counts = [len(d) for d in data]

    fig, ax = plt.subplots(figsize=(14, 7))

    # Global mean
    mu = float(np.mean(scores))
    ax.axhline(mu, color="red", linestyle="--", linewidth=1.2, alpha=0.6,
               label=f"global mean = {mu:.4f}", zorder=1)

    # Boxplot
    bp = ax.boxplot(data, labels=[f"{c}\n(n={n:,})" for c, n in zip(cat_names, counts)],
                    patch_artist=True, showfliers=False, widths=0.55, zorder=2)

    colors = plt.cm.Set2(np.linspace(0, 1, len(cat_names)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # For small categories (n < 200), overlay jittered strip dots
    for i, (d, n) in enumerate(zip(data, counts)):
        if n <= 200:
            jitter = np.random.default_rng(42).normal(0, 0.08, size=n)
            ax.scatter(np.full(n, i + 1) + jitter, d, s=12, alpha=0.6,
                       color="black", zorder=3, edgecolors="none")

    # Annotate medians
    for i, d in enumerate(data):
        med = float(np.median(d))
        ax.text(i + 1, med + 0.001, f"{med:.4f}", ha="center", va="bottom", fontsize=7, color="black")

    ax.set_ylabel("Sink score", fontsize=12)
    ax.set_title("Sink score by token category\n(dots shown for categories with n ≤ 200)", fontsize=13)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(fontsize=9, rotation=0)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "vocab_sink_category_boxplot.png"), dpi=300)
    plt.close(fig)
    print(f"[Plot] Saved vocab_sink_category_boxplot.png")


# ─────────────────────────── main ───────────────────────────

def main():
    p = argparse.ArgumentParser(description="Vocab Sink Sweep")
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--prompt-file", required=True, help="Path to prompt file (one prompt per line)")
    p.add_argument("--n-prompts", type=int, default=1, help="Number of background prompts to average over")
    p.add_argument("--gpus", default=None, help="Comma-separated GPU ids, e.g. '0,1,2,3'. Default: all visible.")
    p.add_argument("--q-start", type=int, default=4, help="Only use queries q >= q_start")
    p.add_argument("--seq-len", type=int, default=32, help="Sequence length (context + token 0)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", default="results_vocab_sweep")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Tokenizer & vocab ──
    print(f"[Init] Loading tokenizer: {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    vocab_size = len(tok.get_vocab())
    all_token_ids = list(range(vocab_size))
    print(f"[Init] Vocab size: {vocab_size:,}")

    special_ids = set(tok.all_special_ids)
    special_tokens_map = {}
    for sid in sorted(special_ids):
        try:
            special_tokens_map[sid] = tok.decode([sid])
        except Exception:
            special_tokens_map[sid] = f"<id={sid}>"
    print(f"[Init] Special tokens ({len(special_ids)}): {special_tokens_map}")

    # ── Load & sample background prompts ──
    with open(args.prompt_file, "r", encoding="utf-8") as f:
        all_prompts = [ln.strip() for ln in f if ln.strip()]
    print(f"[Init] Loaded {len(all_prompts)} prompts from {args.prompt_file}")

    n_ctx = min(args.n_prompts, len(all_prompts))
    sampled_prompts = random.sample(all_prompts, n_ctx)
    print(f"[Init] Using {n_ctx} background prompt(s)")

    # Tokenize contexts: each [1, seq_len]
    context_ids_list = []
    for text in sampled_prompts:
        ids = tok(text, return_tensors="pt", add_special_tokens=True)["input_ids"]
        # Truncate or pad to seq_len
        S = ids.shape[1]
        if S >= args.seq_len:
            ids = ids[:, :args.seq_len]
        else:
            pad = torch.full((1, args.seq_len - S), tok.pad_token_id or 0, dtype=torch.long)
            ids = torch.cat([ids, pad], dim=1)
        context_ids_list.append(ids)
    print(f"[Init] Context seq_len = {args.seq_len}")

    # ── Determine GPUs ──
    if args.gpus:
        gpu_ids = [int(g) for g in args.gpus.split(",")]
    else:
        gpu_ids = list(range(torch.cuda.device_count()))
    n_gpus = len(gpu_ids)
    print(f"[Init] Using {n_gpus} GPU(s): {gpu_ids}")

    # ── Split vocab across GPUs ──
    chunks = np.array_split(np.array(all_token_ids), n_gpus)
    chunks = [c.tolist() for c in chunks]

    t0 = time.time()

    if n_gpus == 1:
        # Single GPU: run in main process
        _disable_packed_sequence_splitting()
        model = _load_model(args.model, f"cuda:{gpu_ids[0]}", args.dtype)
        result_dict = {}

        n_contexts = len(context_ids_list)
        seq_len = args.seq_len
        batch_size = 64
        all_scores = np.zeros((vocab_size, n_contexts), dtype=np.float32)

        for ci, ctx_ids in enumerate(context_ids_list):
            base = ctx_ids[0].clone()
            for start in tqdm(range(0, vocab_size, batch_size), desc=f"ctx={ci+1}/{n_contexts}"):
                end = min(start + batch_size, vocab_size)
                bsz = end - start
                batch_ids = base.unsqueeze(0).expand(bsz, -1).clone()
                for i, tid in enumerate(range(start, end)):
                    batch_ids[i, 0] = tid
                pos_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(bsz, -1)
                attns = _forward_batch(model, batch_ids, pos_ids)
                scores = _compute_sink_scores_from_attns(attns, q_start=args.q_start)
                all_scores[start:end, ci] = scores.numpy()

        sink_means = all_scores.mean(axis=1)
        sink_stds = all_scores.std(axis=1)
        del model
        torch.cuda.empty_cache()

    else:
        # Multi-GPU: spawn workers
        mp.set_start_method("spawn", force=True)
        manager = mp.Manager()
        result_dict = manager.dict()

        processes = []
        for wi, (gpu_id, chunk) in enumerate(zip(gpu_ids, chunks)):
            p_proc = mp.Process(
                target=_worker_run,
                args=(gpu_id, args.model, chunk, context_ids_list, args.q_start, result_dict, wi, args.dtype),
            )
            processes.append(p_proc)
            p_proc.start()

        for p_proc in processes:
            p_proc.join()

        # Reassemble results
        sink_means = np.zeros(vocab_size, dtype=np.float32)
        sink_stds = np.zeros(vocab_size, dtype=np.float32)
        for wi in range(n_gpus):
            res = result_dict[wi]
            for tid, mu, sd in zip(res["token_ids"], res["means"], res["stds"]):
                sink_means[tid] = mu
                sink_stds[tid] = sd

    elapsed = time.time() - t0
    print(f"\n[Done] Sweep completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # ── Save raw data ──
    raw_path = os.path.join(args.outdir, "vocab_sink_scores.npz")
    np.savez_compressed(raw_path, means=sink_means, stds=sink_stds, vocab_size=vocab_size)
    print(f"[Save] Raw scores -> {raw_path}")

    # ── Token strings & categories ──
    print("[Post] Building token strings and categories...")
    token_strs = []
    categories = []
    special_mask = np.zeros(vocab_size, dtype=bool)

    for tid in range(vocab_size):
        try:
            s = tok.decode([tid])
        except Exception:
            s = f"<id={tid}>"
        token_strs.append(s)
        cat = _categorize_token(s, tid, special_ids)
        categories.append(cat)
        if tid in special_ids:
            special_mask[tid] = True

    categories = np.array(categories)

    # ── Print top / bottom tokens ──
    order = np.argsort(-sink_means)
    print(f"\n{'='*60}")
    print(f"TOP 30 tokens by sink score:")
    print(f"{'='*60}")
    for i in range(min(30, vocab_size)):
        tid = int(order[i])
        sp = " ★SPECIAL" if special_mask[tid] else ""
        print(f"  rank={i+1:>5d}  id={tid:>6d}  score={sink_means[tid]:.6f} ± {sink_stds[tid]:.6f}  "
              f"cat={categories[tid]:<12s}  token={repr(token_strs[tid])}{sp}")

    print(f"\n{'='*60}")
    print(f"BOTTOM 30 tokens by sink score:")
    print(f"{'='*60}")
    for i in range(max(0, vocab_size - 30), vocab_size):
        tid = int(order[i])
        sp = " ★SPECIAL" if special_mask[tid] else ""
        print(f"  rank={i+1:>5d}  id={tid:>6d}  score={sink_means[tid]:.6f} ± {sink_stds[tid]:.6f}  "
              f"cat={categories[tid]:<12s}  token={repr(token_strs[tid])}{sp}")

    # ── Summary stats ──
    special_scores = sink_means[special_mask]
    normal_scores = sink_means[~special_mask]
    print(f"\n[Stats] Special tokens:  mean={special_scores.mean():.6f}, std={special_scores.std():.6f}, n={special_scores.size}")
    print(f"[Stats] Normal tokens:   mean={normal_scores.mean():.6f}, std={normal_scores.std():.6f}, n={normal_scores.size}")
    print(f"[Stats] All tokens:      mean={sink_means.mean():.6f}, std={sink_means.std():.6f}, n={sink_means.size}")
    print(f"[Stats] Max sink score:  {sink_means.max():.6f} (token={repr(token_strs[int(np.argmax(sink_means))])})")
    print(f"[Stats] Min sink score:  {sink_means.min():.6f} (token={repr(token_strs[int(np.argmin(sink_means))])})")

    # ── Plot ──
    special_labels = {tid: tok.decode([tid]) for tid in sorted(special_ids)}
    _plot_histogram(sink_means, special_mask, special_labels, args.outdir)
    _plot_rank_curve(sink_means, token_strs, special_mask, args.outdir)
    _plot_category_boxplot(sink_means, categories, args.outdir)

    # Save top/bottom as JSON
    topbot = {
        "top_50": [{"rank": i+1, "id": int(order[i]), "token": token_strs[int(order[i])],
                     "score": float(sink_means[int(order[i])]), "category": categories[int(order[i])],
                     "is_special": bool(special_mask[int(order[i])])} for i in range(min(50, vocab_size))],
        "bottom_50": [{"rank": vocab_size-49+i, "id": int(order[vocab_size-50+i]),
                       "token": token_strs[int(order[vocab_size-50+i])],
                       "score": float(sink_means[int(order[vocab_size-50+i])]),
                       "category": categories[int(order[vocab_size-50+i])],
                       "is_special": bool(special_mask[int(order[vocab_size-50+i])])} for i in range(50)],
        "stats": {
            "special_mean": float(special_scores.mean()),
            "normal_mean": float(normal_scores.mean()),
            "all_mean": float(sink_means.mean()),
            "all_std": float(sink_means.std()),
            "n_prompts": n_ctx,
            "vocab_size": vocab_size,
        }
    }
    with open(os.path.join(args.outdir, "vocab_sink_topbot.json"), "w") as f:
        json.dump(topbot, f, indent=2, ensure_ascii=False)

    print(f"\n[All done] Results in {args.outdir}/")


if __name__ == "__main__":
    main()

