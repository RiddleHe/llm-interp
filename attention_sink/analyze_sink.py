"""
python analyze_sink.py --scan --print heatmap --rope 0=12,1=13,2=14,3=15
python analyze_sink.py --scan --print log
python analyze_sink.py --scan --print heatmap --random-init
python analyze_sink.py --scan --print heatmap --mask upper
"""

import argparse, os, numpy as np, torch, matplotlib.pyplot as plt, math
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch.nn.functional as F

EPS = 1e-8
PRINT_CHOICES = ["heatmap", "log"]

# Perturbation & loading functions

def load_model(model_name, device, dtype, random_init=False):
    torch_dtype = {"auto": "auto", "bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dmap = "auto" if device == "auto" else None

    if random_init:
        print(f"[init] Creating random-init model: {model_name}")
        cfg = AutoConfig.from_pretrained(model_name)
        torch.set_default_device("cuda") # faster init
        model = AutoModelForCausalLM.from_config(cfg, attn_implementation="eager")
        torch.set_default_device("cpu") # compatible with rest of the code
        print(f"[init] Random init complete,")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch_dtype[dtype], device_map=dmap, attn_implementation="eager")
    
    model.eval()
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    return tok, model

def tokenize(tok, text):
    enc = tok(text, return_tensors="pt", add_special_tokens=True)
    return enc["input_ids"]

def make_position_ids(seq_len):
    return torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

def apply_rope_overrides(position_ids, overrides):
    if not overrides:
        return position_ids
    
    pos = position_ids.clone()
    seq_len = pos.shape[1]
    for ti, rp in overrides:
        if 0 <= ti < seq_len:
            pos[0, ti] = int(rp)
        else:
            raise ValueError(f"token idx {ti} out of range (seq_len={seq_len})")
    return pos

def apply_perturbations(base_pos, rope_overrides=None, mask=None):
    pos_ids = base_pos
    applied = {}
    if rope_overrides:
        pos_ids = apply_rope_overrides(base_pos, rope_overrides)
        applied["rope"] = True
    if mask:
        applied["mask"] = True
    return pos_ids, applied

# Parsing functions

def parse_overrides(s):
    """'12=0,42=1' -> [(12,0),(42,1)]"""
    if not s: return []
    out = []
    for part in s.split(","):
        if not part: continue
        ti, rp = part.split("=")
        out.append((int(ti), int(rp)))
    return out

def _parse_qpos(arg, seq_len):
    if arg:
        return [max(0, min(seq_len - 1, int(x))) for x in arg.split(",")]
    last = seq_len - 1
    two_thirds = max(0, (2 * seq_len) // 3 - 1)
    one_third = max(0, seq_len // 3 - 1)
    q_positions = sorted({one_third, two_thirds, last})
    return q_positions

def _append_suffix(path, suffix):
    if not suffix:
        return path
    root, ext = os.path.splitext(path)
    return f"{root}{suffix}{ext}"

# Forward & capture functions

def _forward_attn_and_qkv(model, input_ids, position_ids, layer, need_qkv=False):
    cache = {}
    if need_qkv:
        _capture_qkv(model, layer, cache)
    attns = prefill(model, input_ids, position_ids)
    hook = cache.pop("_hook", None)
    if hook is not None:
        hook.remove()

    q, k, v = cache.get("q", None), cache.get("k", None), cache.get("v", None)
    return attns, q, k, v

@torch.no_grad()
def prefill(model, input_ids, position_ids):
    device = next(model.parameters()).device
    out = model(
        input_ids=input_ids.to(device),
        position_ids=position_ids.to(device),
        use_cache=False,
        output_attentions=True,
        output_hidden_states=True,
    )
    return out.attentions # tuple(len=layers) of [B, H, Q, K]

def _capture_qkv(model, layer_idx, cache):
    attn = model.model.layers[layer_idx].self_attn
    num_heads = model.config.num_attention_heads
    num_kv = model.config.num_key_value_heads
    num_groups = max(1, num_heads // max(1, num_kv))
    head_dim = model.config.head_dim

    def hook(_module, args, kwargs):
        hidden_states = kwargs["hidden_states"]
        cos, sin = kwargs["position_embeddings"]
        B, S, _ = hidden_states.shape

        q = attn.q_proj(hidden_states).view(B, S, num_heads, head_dim)
        k = attn.k_proj(hidden_states).view(B, S, num_kv, head_dim)
        v = attn.v_proj(hidden_states).view(B, S, num_kv, head_dim)
        q = attn.q_norm(q).transpose(1, 2).contiguous()
        k = attn.k_norm(k).transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if num_kv != num_heads:
            k = k.repeat_interleave(num_groups, dim=1)
            v = v.repeat_interleave(num_groups, dim=1)
        cache["q"] = q.detach().float().cpu()
        cache["k"] = k.detach().float().cpu()
        cache["v"] = v.detach().float().cpu()

    h = attn.register_forward_pre_hook(hook, with_kwargs=True)
    cache["_hook"] = h

def compute_cosine_series(q, k, head_idx, k_sink_idx, q_positions):
    kv = k[0, head_idx, k_sink_idx]
    kn = F.normalize(kv, dim=0)

    series = []
    for qi in q_positions:
        qv = q[0, head_idx, qi]
        qn = F.normalize(qv, dim=0)
        cos = float(torch.dot(qn, kn))
        series.append((qi, cos))
    return series, float(kv.norm().item())

# Run functions

def run_cosine(model, input_ids, position_ids, layer, head, qpos_arg):
    attns, q, k, v = _forward_attn_and_qkv(model, input_ids, position_ids, layer, need_qkv=True)
    if q is None or k is None:
        raise RuntimeError("Failed to capture Q/K.")

    seq_len = q.shape[2]
    q_positions = list(_parse_qpos(qpos_arg, seq_len))
    series, k_norm = compute_cosine_series(q, k, head, k_sink_idx=0, q_positions=q_positions)
    v_norm = float(v[0, head, 0].norm().item())
    attn_mat = attns[layer][0, head].detach().float().cpu()

    print(f"Cosine (Q[q_idx], k[0]) layer={layer} head={head}")
    cos_vals = []
    for qi, cos in series:
        attn_to_sink = float(attn_mat[qi, 0].item())
        print(f"  q={qi}\tcos={cos:.6f}\tattn={attn_to_sink:.6f}")
        cos_vals.append(cos)
    return {
        "layer": layer, "head": head, "q_positions": q_positions,
        "cos": cos_vals, "k_norm": float(k_norm), "v_norm": float(v_norm)
    }

def run_heatmap(model, input_ids, position_ids, layer, head, outdir, tag="BASE", return_map=False, suffix=""):
    attns, _, _, _ = _forward_attn_and_qkv(model, input_ids, position_ids, layer, need_qkv=False)
    head_attn = pick_head(attns, layer, head)
    title = f"Layer {layer} Head {head}" if return_map else f"{tag}Layer {layer} head {head}" 
    if return_map:
        return head_attn, title
    tag_safe = str(tag).lower().replace(" ", "_")
    _plot_attn(
        head_attn, 
        title, 
        os.path.join(outdir, f"{tag_safe}_layer_{layer}_head_{head}.png"),
        suffix=suffix
    )
    return None, None

def pick_head(attentions, layer_idx, head_idx):
    attn_l = attentions[layer_idx]
    return attn_l[0, head_idx].detach().float().cpu().numpy()

# Plotting functions

def _plot_attn(attn, title, out_path, suffix=""):
    plt.figure(figsize=(6, 4.5))
    im = plt.imshow(attn, aspect="auto", origin="lower", interpolation="nearest")
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel("K")
    plt.ylabel("Q")
    cbar = plt.colorbar(im)
    cbar.set_label("Attention prob")
    plt.savefig(_append_suffix(out_path, suffix), bbox_inches="tight", dpi=300)
    plt.close()

def _plot_norm_progression(stats, outdir, key, ylabel, title, fname, suffix=""):
    layers = sorted({s["layer"] for s in stats})
    by_layer = {s["layer"]: s for s in stats}
    y = [by_layer[L][key] for L in layers]
    plt.figure(figsize=(7, 3.5))
    plt.plot(layers, y, marker="o")
    plt.xlabel("Layer")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(_append_suffix(os.path.join(outdir, fname), suffix), dpi=300)
    plt.close()

def _plot_cosine_stack(stats, outdir, suffix=""):
    layers = sorted({s["layer"] for s in stats})
    by_layer = {s["layer"]: s for s in stats}
    q_positions = stats[0]["q_positions"]
    nrows = len(q_positions)
    fig, axes = plt.subplots(nrows, 1, figsize=(7, 1.8*nrows), sharex=True)
    if nrows == 1:
        axes = [axes]
    
    for i, qi in enumerate(q_positions):
        y = [by_layer[L]["cos"][i] for L in layers]
        axes[i].plot(layers, y, marker="o")
        axes[i].set_ylabel(f"cos@q={qi}")
    
    axes[-1].set_xlabel("Layer")
    fig.suptitle(f"Cosine to K[0] across layers")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(_append_suffix(os.path.join(outdir, f"scan_cosine.png"), suffix), dpi=300)
    plt.close(fig)

def _plot_heatmap_grid(attn_maps, titles, out_path, rows=4, suffix="", suptitle="Attention heatmaps"):
    n = len(attn_maps)
    if n == 0:
        return
    cols = math.ceil(n / rows)
    fig, axes = plt.subplots(rows, cols, figsize=(3.0*cols, 2.6*rows), squeeze=False)

    vmin = min(float(np.min(m)) for m in attn_maps)
    vmax = max(float(np.max(m)) for m in attn_maps)
    for i in range(rows*cols):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        if i < n:
            im = ax.imshow(
                attn_maps[i], aspect="auto", origin="lower", interpolation="nearest",
                vmin=vmin, vmax=vmax
            )
            ax.set_title(titles[i], fontsize=9)
            ax.set_xlabel("K")
            ax.set_ylabel("Q")
            ax.invert_yaxis()
        else:
            ax.axis("off")
    
    fig.subplots_adjust(right=0.88, top=0.92, bottom=0.02, hspace=0.25, wspace=0.25)
    cbar_ax = fig.add_axes([0.90, 0.22, 0.02, 0.65])
    fig.colorbar(im, cax=cbar_ax, label="Attention prob")
    fig.suptitle(suptitle, fontsize=12, y=0.985)
    fig.tight_layout(rect=[0, 0.01, 0.88, 0.95])
    fig.savefig(_append_suffix(out_path, suffix), bbox_inches="tight", dpi=300)
    plt.close(fig)

def _pert_suffix(rope_overrides=None, mask=None, random_init=False):
    suffix = ""
    if rope_overrides:
        spec = "-".join(f"{ti}={rp}" for ti, rp in rope_overrides)
        suffix += "__" + f"rope[{spec}]"
    if mask:
        suffix += "__" + f"mask[{mask}]"
    if random_init:
        suffix += "__" + "random_init"
    return suffix

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="bf16", choices=["auto", "bf16", "fp16", "fp32"])
    p.add_argument("--prompt", default=None)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--head", type=int, default=0)
    # scanning mode
    p.add_argument("--scan", action="store_true")
    # print mode
    p.add_argument("--print", dest="print_mode", choices=PRINT_CHOICES, default="log")
    p.add_argument("--qpos", default=None, help="Comma-separated list of query positions for --print log, eg. '256,512,768")
    # perturbations
    p.add_argument("--rope", default=None, help="Comma-separated list of token_idx=rope_pos, eg. '12=0,42=1")
    p.add_argument("--mask", default=None, choices=["upper"], help="Causal mask type")
    p.add_argument("--random-init", action="store_true", help="DO NOT load pretrained weights")
    p.add_argument("--outdir", default="results")
    args = p.parse_args()

    tok, model = load_model(args.model, args.device, args.dtype, random_init=args.random_init)

    if args.prompt:
        text = args.prompt
    else:
        text = "To understand the failure of window attention, we find an interesting phenomenon of autoregressive LLMs: a surprisingly large amount of attention score is allocated to the initial tokens, irrespective of their relevance to the language modeling task"

    input_ids = tokenize(tok, text)
    base_pos = make_position_ids(input_ids.shape[1])
    os.makedirs(args.outdir, exist_ok=True)

    rope_str = args.rope
    rope_overrides = parse_overrides(rope_str) if rope_str else None
    pert_suffix = _pert_suffix(rope_overrides=rope_overrides, mask=args.mask, random_init=args.random_init)

    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    scan_stats = []
    scan_maps, scan_titles = [], []

    def handle_one(layer, head):
        pos_ids_perturbed, applied = apply_perturbations(base_pos, rope_overrides=rope_overrides)

        if args.print_mode == "log":
            stats = run_cosine(model, input_ids, pos_ids_perturbed, layer, head, args.qpos)
            scan_stats.append(stats)

        elif args.print_mode == "heatmap":
            tag = ""
            if rope_overrides:
                ov_str = ",".join([f"{ti}={rp}" for ti, rp in rope_overrides])
                tag = f"PERTURBED rope({ov_str}) "

            if args.scan:
                m, t = run_heatmap(
                    model, input_ids, pos_ids_perturbed, layer, head, 
                    args.outdir, tag=tag, return_map=True, suffix=pert_suffix
                )
                if m is not None and t is not None:
                    scan_maps.append(m)
                    scan_titles.append(t)
            else: 
                run_heatmap(
                    model, input_ids, pos_ids_perturbed, layer, head, 
                    args.outdir, tag=tag, return_map=False, suffix=pert_suffix
                )

    if args.scan:
        for L in range(0, num_layers, 4):
            handle_one(L, args.head)

        if args.print_mode == "log" and len(scan_stats) > 0:
            scan_stats = sorted(scan_stats, key=lambda s: s["layer"])
            _plot_norm_progression(
                scan_stats, args.outdir, key="k_norm", ylabel="||K[0]||",
                title="K-norm progression", fname="scan_knorm.png", suffix=pert_suffix
            )
            _plot_norm_progression(
                scan_stats, args.outdir, key="v_norm", ylabel="||V[0]||",
                title="V-norm progression", fname="scan_vnorm.png", suffix=pert_suffix
            )
            _plot_cosine_stack(scan_stats, args.outdir, suffix=pert_suffix)
        
        elif args.print_mode == "heatmap" and len(scan_maps) > 0:
            grid_title = "Attention heatmaps"
            if rope_overrides:
                spec = ",".join(f"{ti}->{rp}" for ti, rp in rope_overrides)
                grid_title += f" - rope({spec})"
            if args.mask:
                grid_title += f" - mask({args.mask})"
            if args.random_init:
                grid_title += " - [random init]"

            _plot_heatmap_grid(
                scan_maps, scan_titles, os.path.join(args.outdir, "scan_heatmaps.png"), rows=4,
                suffix=pert_suffix, suptitle=grid_title,
            )

    else:
        handle_one(args.layer, args.head)

if __name__ == "__main__":
    main()