"""
python analyze_sink.py --scan \
    --print [heatmap, qkv] \
    --rope 0=12,1=13,2=14,3=15 \
    --random-init \
    --lower-attn \

"""

import argparse, os, numpy as np, torch, matplotlib.pyplot as plt, math
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch.nn.functional as F
from tqdm import tqdm
from matplotlib import ticker

EPS = 1e-12
PRINT_CHOICES = ["heatmap", "qkv"]

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
        print(f"[init] Random init complete.")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch_dtype[dtype], device_map=dmap, attn_implementation="eager")
    
    model.eval()
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    return tok, model

def tokenize(tok, text):
    enc = tok(text, return_tensors="pt", add_special_tokens=True)
    return enc["input_ids"]

def _load_prompts_from_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f.readlines() if ln.strip()]

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

def _install_lower_attn_hooks(model, layers, factor=0.5, sink_k_idx=0):
    handles = []
    num_heads = model.config.num_attention_heads
    num_kv = model.config.num_key_value_heads
    num_groups = max(1, num_heads // max(1, num_kv))

    model._lower_attn_cache = {}

    for L in layers:
        attn = model.model.layers[L].self_attn
        cache = {}

        def _prehook(_module, args, kwargs, _cache=cache):
            hidden_states = kwargs["hidden_states"]
            B, S, _ = hidden_states.shape
            head_dim = model.config.head_dim
            v = attn.v_proj(hidden_states).view(B, S, num_kv, head_dim)
            v = v.transpose(1, 2).contiguous()
            if num_kv != num_heads:
                v = v.repeat_interleave(num_groups, dim=1)
            _cache["v"] = v

        def _fwdhook(_module, args, output, _cache=cache):
            attn_output, attn_probs = output[0], output[1]
            model._lower_attn_cache[L] = attn_probs.detach().float().cpu()
            
            probs = attn_probs.clone()
            probs[..., sink_k_idx] *= factor  # we don't renormalize

            v = _cache["v"]
            ctx = torch.matmul(probs, v)
            B, H, Q, D = ctx.shape
            ctx = ctx.transpose(1, 2).contiguous().view(B, Q, H*D)
            new_out = _module.o_proj(ctx)

            out_list = list(output)
            out_list[0] = new_out
            out_list[1] = probs
            return tuple(out_list)

        h1 = attn.register_forward_pre_hook(_prehook, with_kwargs=True)
        h2 = attn.register_forward_hook(_fwdhook)
        handles.append([h1, h2])
    return handles

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

def pick_head(attentions, layer_idx, head_idx):
    attn_l = attentions[layer_idx]
    return attn_l[0, head_idx].detach().float().cpu().numpy()

# Plotting functions

def _plot_heatmap_with_tag(head_attn, layer, head, outdir, tag="BASE", return_map=False, suffix=""):
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

def _plot_attn(attn, title, out_path, suffix=""):
    plt.figure(figsize=(6, 4.5))
    im = plt.imshow(attn, aspect="auto", origin="lower", interpolation="nearest")
    ax = plt.gca()
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel("K")
    ax.set_ylabel("Q")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
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

    if len(y) > 0:
        plt.axhline(y[-1], color="gray", linestyle="--", linewidth=1.0)

    plt.title(title)
    plt.tight_layout()
    plt.savefig(_append_suffix(os.path.join(outdir, fname), suffix), dpi=300)
    plt.close()

def _plot_cosine_stack(stats, outdir, suffix="", suptitle="Cosine to K[sink] across layers"):
    layers = sorted({s["layer"] for s in stats})
    by_layer = {s["layer"]: s for s in stats}
    q_positions = stats[0]["q_positions"]
    nrows = len(q_positions)
    fig_height = 2.2*nrows
    fig, axes = plt.subplots(nrows, 1, figsize=(7, fig_height), sharex=True)
    if nrows == 1:
        axes = [axes]
    
    for i, qi in enumerate(q_positions):
        y = [by_layer[L]["cos"][i] for L in layers]
        axes[i].plot(layers, y, marker="o")
        axes[i].set_ylabel(f"cos@q={qi}")
        axes[i].set_ylim(0.0, 1.0)
        if len(y) > 0:
            axes[i].axhline(y[-1], color="gray", linestyle="--", linewidth=1.0)
    
    axes[-1].set_xlabel("Layer")
    fig.suptitle(suptitle)
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

            ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
        else:
            ax.axis("off")
    
    fig.subplots_adjust(right=0.88, top=0.92, bottom=0.02, hspace=0.25, wspace=0.25)
    cbar_ax = fig.add_axes([0.90, 0.22, 0.02, 0.65])
    fig.colorbar(im, cax=cbar_ax, label="Attention prob")
    fig.suptitle(suptitle, fontsize=12, y=0.985)
    fig.tight_layout(rect=[0, 0.01, 0.88, 0.95])
    fig.savefig(_append_suffix(out_path, suffix), bbox_inches="tight", dpi=300)
    plt.close(fig)

def _pert_suffix(args):
    suffix = ""
    if args.rope:
        suffix += "__" + f"rope[{args.rope}]"
    if args.mask:
        suffix += "__" + f"mask[{args.mask}]"
    if args.random_init:
        suffix += "__" + "random_init"
    if args.lower_attn:
        suffix += "__" + "lower_attn"
    if args.sink_idx != 0:
        suffix += "__" + f"sink[{args.sink_idx}]"
    return suffix

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="bf16", choices=["auto", "bf16", "fp16", "fp32"])
    p.add_argument("--prompt", default=None)
    p.add_argument("--prompt-file", default=None)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--head", type=int, default=0)
    # scanning mode
    p.add_argument("--scan", action="store_true")
    # print mode
    p.add_argument("--print", dest="print_mode", choices=PRINT_CHOICES, default="qkv")
    p.add_argument("--qpos", default=None, help="Comma-separated list of query positions for --print qkv, eg. '256,512,768")
    p.add_argument("--sink-idx", type=int, default=0, help="Index of the sink in qkv mode")
    p.add_argument("--compare", action="store_true", help="Compare perturbed run vs baseline")
    # perturbations
    p.add_argument("--rope", default=None, help="Comma-separated list of token_idx=rope_pos, eg. '12=0,42=1")
    p.add_argument("--mask", default=None, choices=["upper"], help="Causal mask type")
    p.add_argument("--random-init", action="store_true", help="DO NOT load pretrained weights")
    p.add_argument("--lower-attn", action="store_true", help="Iteratively lower attention on sink key at each layer")
    # output
    p.add_argument("--outdir", default="results")
    args = p.parse_args()

    tok, model = load_model(args.model, args.device, args.dtype, random_init=args.random_init)

    if args.compare and args.print_mode == "heatmap":
        raise NotImplementedError("--compare is only available for --print qkv")

    if args.prompt_file:
        prompts = _load_prompts_from_file(args.prompt_file)
    else:
        if args.prompt:
            prompts = [args.prompt]
        else:
            prompts = ["To understand the failure of window attention, we find an interesting phenomenon of autoregressive LLMs: a surprisingly large amount of attention score is allocated to the initial tokens, irrespective of their relevance to the language modeling task"]

    tokenized = []
    for text in prompts:
        input_ids = tokenize(tok, text)
        base_pos = make_position_ids(input_ids.shape[1])
        tokenized.append((input_ids, base_pos))
    print(f"[Tokenized] Tokens[0, 0] is {tokenized[0][0][0, 0]}")
    print(f"[Tokenized] Tokens[0, 0] text is {tok.decode(tokenized[0][0][0, 0])}")

    os.makedirs(args.outdir, exist_ok=True)

    rope_str = args.rope
    rope_overrides = parse_overrides(rope_str) if rope_str else None
    pert_suffix = _pert_suffix(args)

    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads

    lower_attn_handles = []
    if args.lower_attn:
        lower_attn_handles = _install_lower_attn_hooks(
            model, layers=range(num_layers), factor=0.5, sink_k_idx=0
        )

    scan_stats = [] # aggregated per layer
    scan_stats_base = []
    scan_maps, scan_titles = [], []

    def _aggregate_qkv_for_layer(layer, head):
        cos_list = []
        k_norms, v_norms = [], []
        attn_maps = []
        min_len = min(ids.shape[1] for (ids, _bp) in tokenized)
        q_positions = list(_parse_qpos(args.qpos, min_len)) # clamp to valid seq range
        sink_idx = max(0, min(min_len - 1, args.sink_idx)) 
        
        for input_ids, base_pos in tqdm(tokenized, desc="Aggregating QKV", position=1, leave=False):
            pos_ids_perturbed, _ = apply_perturbations(base_pos, rope_overrides=rope_overrides)
            attns, q, k, v = _forward_attn_and_qkv(model, input_ids, pos_ids_perturbed, layer, need_qkv=True)
            if q is None or k is None:
                raise RuntimeError("Failed to capture Q/K.")
            series, k_norm = compute_cosine_series(q, k, head, k_sink_idx=sink_idx, q_positions=q_positions)
            v_norm = float(v[0, head, sink_idx].norm().item())

            k_norms.append(k_norm)
            v_norms.append(v_norm)
            cos_list.append([c for (_, c) in series])
            if hasattr(model, "_lower_attn_cache") and layer in model._lower_attn_cache:
                hm = pick_head(model._lower_attn_cache, layer, head)
            else:
                hm = pick_head(attns, layer, head)
            attn_maps.append(hm[:min_len, :min_len])

        cos_mean = np.mean(np.array(cos_list), axis=0).tolist()
        k_mean = float(np.mean(k_norms))
        v_mean = float(np.mean(v_norms))
        attn_agg = np.mean(np.stack(attn_maps, axis=0), axis=0)
        return {
            "k_norm": k_mean, "v_norm": v_mean, "cos": cos_mean, 
            "layer": layer, "head": head, "q_positions": q_positions
        }

    def _aggregate_attn_score_for_layer(layer, head, tag):
        attn_maps = []
        min_len = min(ids.shape[1] for (ids, _bp) in tokenized)
        for input_ids, base_pos in tqdm(tokenized, desc="Aggregating attention scores", position=1, leave=False):
            pos_ids_perturbed, _ = apply_perturbations(base_pos, rope_overrides=rope_overrides)
            attns, _, _, _ = _forward_attn_and_qkv(model, input_ids, pos_ids_perturbed, layer, need_qkv=False)
            if hasattr(model, "_lower_attn_cache") and layer in model._lower_attn_cache:
                hm = pick_head(model._lower_attn_cache, layer, head)
            else:
                hm = pick_head(attns, layer, head)
            attn_maps.append(hm[:min_len, :min_len])

        head_attn = np.mean(np.stack(attn_maps, axis=0), axis=0)
        if args.scan:
            m, t = _plot_heatmap_with_tag(head_attn, layer, head, args.outdir, tag=tag, return_map=True, suffix=pert_suffix)
            return m, t
        else:
            _plot_heatmap_with_tag(head_attn, layer, head, args.outdir, tag=tag, return_map=False, suffix=pert_suffix)
            return None, None

    if args.scan:
        # perturbed pass
        for L in tqdm(range(0, num_layers, 4), desc="Scanning layers", position=0, leave=True):
            if args.print_mode == "qkv":
                stats = _aggregate_qkv_for_layer(L, args.head)
                scan_stats.append(stats)
            else:
                tag = ""
                if rope_overrides:
                    ov_str = ",".join([f"{ti}={rp}" for ti, rp in rope_overrides])
                    tag = f"PERTURBED rope({ov_str}) "
                m, t = _aggregate_attn_score_for_layer(L, args.head, tag)
                if m is not None and t is not None:
                    scan_maps.append(m)
                    scan_titles.append(t)
        
        # baseline pass
        if args.compare and args.print_mode == "qkv":
            def _agg_baseline(layer, head):
                nonlocal rope_overrides
                tmp = rope_overrides
                rope_overrides = None
                try:
                    return _aggregate_qkv_for_layer(layer, head)
                finally:
                    rope_overrides = tmp
            for L in range(0, num_layers, 4):
                scan_stats_base.append(_agg_baseline(L, args.head))

        if args.print_mode == "qkv" and len(scan_stats) > 0:
            scan_stats = sorted(scan_stats, key=lambda s: s["layer"])
            _plot_norm_progression(
                scan_stats, args.outdir, key="k_norm", ylabel="||K[0]||",
                title=f"K-norm progression (sink={args.sink_idx})", fname="scan_knorm.png", suffix=pert_suffix
            )
            _plot_norm_progression(
                scan_stats, args.outdir, key="v_norm", ylabel="||V[0]||",
                title=f"V-norm progression (sink={args.sink_idx})", fname="scan_vnorm.png", suffix=pert_suffix
            )
            _plot_cosine_stack(scan_stats, args.outdir, suffix=pert_suffix, suptitle=f"Cosine to K[{args.sink_idx}] across layers")
        
        elif args.print_mode == "heatmap" and len(scan_maps) > 0:
            grid_title = "Attention heatmaps"
            if rope_overrides:
                spec = ",".join(f"{ti}->{rp}" for ti, rp in rope_overrides)
                grid_title += f" - rope({spec})"
            if args.mask:
                grid_title += f" - mask({args.mask})"
            if args.random_init:
                grid_title += " - [random init]"
            if args.lower_attn:
                grid_title += " - lower_attn (x0.5)"

            _plot_heatmap_grid(
                scan_maps, scan_titles, os.path.join(args.outdir, "scan_heatmaps.png"), rows=4,
                suffix=pert_suffix, suptitle=grid_title,
            )

    else:
        if args.print_mode == "qkv":
            _ = _aggregate_qkv_for_layer(args.layer, args.head)
        else:
            tag = ""
            if rope_overrides:
                ov_str = ",".join([f"{ti}={rp}" for ti, rp in rope_overrides])
                tag = f"PERTURBED rope({ov_str}) "
            _, _ = _aggregate_attn_score_for_layer(args.layer, args.head, tag=tag)
        
    for pair in lower_attn_handles:
        for h in pair:
            try:
                h.remove()
            except Exception:
                pass

if __name__ == "__main__":
    main()