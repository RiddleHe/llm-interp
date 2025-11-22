import argparse, os, numpy as np, torch, matplotlib.pyplot as plt, math
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch.nn.functional as F
from tqdm import tqdm
from matplotlib import ticker
from mpl_toolkits.mplot3d import Axes3D

EPS = 1e-12
PRINT_CHOICES = ["heatmap", "qkv"]

# Perturbation & loading functions

def _disable_packed_sequence_splitting():
    try:
        import transformers.masking_utils as _mu
        import torch
        
        def _no_split_find_packed_sequence_indices(position_ids, *args, **kwargs):
            return torch.zeros_like(position_ids, dtype=torch.long)
        
        _mu.find_packed_sequence_indices = _no_split_find_packed_sequence_indices
        print("[transformers] Disabled packed-sequence split.")
    except Exception as e:
        print(f"[transformers] Failed to patch find_packed_sequence_indices: {e}")

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

def _make_v_prehook(attn, cache, num_heads, num_kv, num_groups, model_cfg):
    def _prehook(_module, args, kwargs, _cache=cache):
        pass
    return _prehook

def _install_lower_attn_hooks(model, layers, factor=0.5, sink_k_idx=0, only_return_sink_value=False):
    handles = []
    num_heads = model.config.num_attention_heads
    num_kv = model.config.num_key_value_heads
    num_groups = max(1, num_heads // max(1, num_kv))

    model._lower_attn_cache = {}

    for L in layers:
        attn = model.model.layers[L].self_attn
        cache = {}

        def _prehook(_module, args, kwargs, _cache=cache, 
                    _num_heads=num_heads, _num_kv=num_kv, _num_groups=num_groups, _head_dim=model.config.head_dim):
            hidden_states = kwargs["hidden_states"]
            B, S, _ = hidden_states.shape
            v = _module.v_proj(hidden_states).view(B, S, _num_kv, _head_dim)
            v = v.transpose(1, 2).contiguous()
            if _num_kv != _num_heads:
                v = v.repeat_interleave(_num_groups, dim=1)
            _cache["v"] = v

        def _fwdhook(_module, args, output, _cache=cache, _L=L):
            attn_output, attn_probs = output[0], output[1]
            model._lower_attn_cache[_L] = attn_probs.detach().float().cpu()
            
            probs = attn_probs.clone()
            probs[..., sink_k_idx] *= factor  # we don't renormalize

            v = _cache["v"]
            if only_return_sink_value:
                sink_w = probs[..., sink_k_idx].unsqueeze(-1) # [B, H, Q, 1]
                v_sink = v[:, :, sink_k_idx:sink_k_idx+1, :] # [B, H, 1, D]
                ctx = sink_w * v_sink 
            else:
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

def _install_window_attn_hooks(model, layers, window_size=4):
    handles = []
    num_heads = model.config.num_attention_heads
    num_kv = model.config.num_key_value_heads
    num_groups = max(1, num_heads // max(1, num_kv))
    model._window_attn_cache = {}

    for L in layers:
        attn = model.model.layers[L].self_attn
        cache = {}

        _prehook = _make_v_prehook(attn, cache, num_heads, num_kv, num_groups, model.config)

        def _fwdhook(_module, args, output, _cache=cache):
            attn_output, attn_probs = output[0], output[1]
            B, H, Q, K = attn_probs.shape

            k_idx = torch.arange(K, device=attn_probs.device).view(1, 1, 1, K)
            q_idx = torch.arange(Q, device=attn_probs.device).view(1, 1, Q, 1)
            start = (q_idx - (window_size - 1)).clamp_min(0)
            mask = (k_idx >= start) & (k_idx <= q_idx)
            mask = mask.to(attn_probs.dtype)

            probs = attn_probs * mask
            denom = probs.sum(dim=-1, keepdim=True).clamp_min(EPS)
            probs = probs / denom

            v = _cache["v"]
            ctx = torch.matmul(probs, v)
            B_, H_, Q_, D = ctx.shape
            ctx = ctx.transpose(1, 2).contiguous().view(B_, Q_, H_ * D)
            new_out = _module.o_proj(ctx)

            model._window_attn_cache[L] = probs.detach().float().cpu()
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

# PCA

def _pca_from_rows(X, k):
    X = X - X.mean(dim=0)
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    var = (S ** 2)
    total = var.sum().clamp_min(EPS)
    k = min(k, Vh.shape[0])
    rel = (var[:k] / total).cpu().numpy()
    return U[:, :k], S[:k], Vh[:k, :], rel, var.sum().item()

def _cross_explained_variance(X, B):
    Xc = X - X.mean(dim=0, keepdim=True)
    Xproj = (Xc @ B.T) @ B
    num = (Xproj ** 2).sum()
    den = (Xc ** 2).sum().clamp_min(EPS)
    return float((num / den).item())

def _q_bias_3d_projection(XQ, XK, topk=2):
    mu = XK.mean(dim=0)
    if float((mu ** 2).sum().item()) <= EPS:
        return None
    u = F.normalize(mu, dim=0)
    q_energy = (XQ ** 2).sum(dim=1, keepdim=True).clamp_min(EPS)

    bias_coord = XQ @ u
    E_bias = (bias_coord ** 2).unsqueeze(1)

    q_parallel = bias_coord.unsqueeze(1) * u.unsqueeze(0)
    XQ_res = XQ - q_parallel
    XQ_res_c = XQ_res - XQ_res.mean(dim=0, keepdim=True)

    _U, _S, Vh = torch.linalg.svd(XQ_res_c, full_matrices=False)

    k = min(topk, Vh.shape[0])
    if k < 2:
        return None
    V_res = Vh[:k]
    
    res_coords = XQ_res @ V_res.T
    E_res = res_coords ** 2

    frac_bias = E_bias / q_energy
    frac_res = E_res / q_energy
    frac = torch.cat([frac_bias, frac_res], dim=1)
    basis = torch.cat([u.unsqueeze(0), Vh[:k]], dim=0)
    return frac.cpu().numpy()

def _mean_dir_decomposition(XQ, XK, VhQ, label, k):
    mu = XK.mean(dim=0)
    mu_norm_sq = float((mu ** 2).sum().item())
    if mu_norm_sq <= EPS:
        return 0.0, 0.0
    u = F.normalize(mu, dim=0)
    q_energy = (XQ ** 2).sum(dim=1)
    sink_energy = (XQ @ u) ** 2
    frac_q_along_mu = float((sink_energy.mean() / q_energy.mean()).item())
    coeff = VhQ @ mu
    frac_mu_in_Q_topk = float(((coeff ** 2).sum().item()) / mu_norm_sq)
    print(
        f"[MEAN DIRECTION] {label}: frac of Q energy along mean({label}) = {frac_q_along_mu:.4f} | "
        f"frac of mean({label}) in top-{k} Q PCs = {frac_mu_in_Q_topk:.4f}"
    )
    return frac_q_along_mu, frac_mu_in_Q_topk

def _get_head_k_proj_matrix(model, layer_idx, head_idx):
    attn = model.model.layers[layer_idx].self_attn
    num_heads = model.config.num_attention_heads
    num_kv = model.config.num_key_value_heads
    num_groups = max(1, num_heads // max(1, num_kv))
    head_dim = model.config.head_dim

    kv_idx = head_idx // num_groups
    k_proj = attn.k_proj
    W_full = k_proj.weight.detach() # [num_kv * head_dim, d_model]
    start = kv_idx * head_dim
    end = start + head_dim
    return W_full[start:end, :]

def _compute_residual_sink_direction(model, layer_idx, head_idx, mu0):
    if mu0 is None:
        return None
    mu0 = mu0.to(torch.float32)
    n2 = float((mu0 ** 2).sum().item())
    if n2 <= EPS:
        return None
    u0 = F.normalize(mu0, dim=0)
    Wk = _get_head_k_proj_matrix(model, layer_idx, head_idx).to(mu0.device, dtype=torch.float32)
    Wk_pinv = torch.linalg.pinv(Wk) 
    r_sink = Wk_pinv @ u0
    if float((r_sink ** 2).sum().item()) <= EPS:
        return None
    return F.normalize(r_sink, dim=0)

def _energy_fraction_along_direction(R, direction):
    if R is None or direction is None:
        return None
    if direction.numel() == 0:
        return None
    d = F.normalize(direction, dim=0)
    coords = R @ d
    sub = coords ** 2
    tot = (R ** 2).sum(dim=1).clamp_min(EPS)
    return sub / tot

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

def _append_suffix(path, suffix):
    if not suffix:
        return path
    root, ext = os.path.splitext(path)
    return f"{root}{suffix}{ext}"

# Forward & capture functions

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

def _capture_qkv_multi(model, layer_idxs, cache):
    handles = []
    for L in layer_idxs:
        attn = model.model.layers[L].self_attn
        num_heads = model.config.num_attention_heads
        num_kv = model.config.num_key_value_heads
        num_groups = max(1, num_heads // max(1, num_kv))
        head_dim = model.config.head_dim

        def _mk_hook(L, attn_layer):
            def hook(_module, args, kwargs):
                hidden_states = kwargs["hidden_states"]
                cos, sin = kwargs["position_embeddings"]
                B, S, _ = hidden_states.shape

                q = attn_layer.q_proj(hidden_states).view(B, S, num_heads, head_dim)
                k = attn_layer.k_proj(hidden_states).view(B, S, num_kv, head_dim)
                v = attn_layer.v_proj(hidden_states).view(B, S, num_kv, head_dim)
                q = attn_layer.q_norm(q).transpose(1, 2).contiguous()
                k = attn_layer.k_norm(k).transpose(1, 2).contiguous()
                v = v.transpose(1, 2).contiguous()

                from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
                if num_kv != num_heads:
                    k = repeat_kv(k, num_groups)
                    v = repeat_kv(v, num_groups)
                cache[L] = {
                    "q": q.detach().float().cpu(),
                    "k": k.detach().float().cpu(),
                    "v": v.detach().float().cpu(),
                    "residual": hidden_states.detach().float().cpu(),
                }
            return hook

        h = attn.register_forward_pre_hook(_mk_hook(L, attn), with_kwargs=True)
        handles.append(h)
    return handles

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

def _pick_head_with_caches(model, attns, layer_idx, head_idx):
    if hasattr(model, "_lower_attn_cache") and (layer_idx in model._lower_attn_cache):
        return pick_head(model._lower_attn_cache, layer_idx, head_idx)
    if hasattr(model, "_window_attn_cache") and (layer_idx in model._window_attn_cache):
        return pick_head(model._window_attn_cache, layer_idx, head_idx)
    return pick_head(attns, layer_idx, head_idx)

# Plotting functions

def _plot_progression(stats_a, outdir, key, ylabel, title, fname, suffix="", stats_b=None, is_cos=False):
    layers = sorted({s["layer"] for s in stats_a})
    by_a = {s["layer"]: s for s in stats_a}
    y_a = [by_a[L][key] for L in layers]
    plt.figure(figsize=(7, 3.5))
    ax = plt.gca()
    ax.plot(layers, y_a, marker="o", label="Perturbed" if stats_b else "Norm of K")

    if stats_b:
        by_b = {s["layer"]: s for s in stats_b}
        y_b = [by_b[L][key] for L in layers]
        ax.plot(layers, y_b, marker="s", label="Baseline")

    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    if len(y_a) > 0:
        ax.axhline(y_a[-1], color="gray", linestyle="--", linewidth=1.0)
    if is_cos:
        ax.set_ylim(0, 1)
    else:
        ax.set_ylim(0, 100)

    if len(layers) > 0:
        ax.set_xlim(min(layers), max(layers))
        ax.set_xticks(layers)
        ax.set_xticklabels(["baseline" if L == -1 else str(L) for L in layers], rotation=0)
        if -1 not in layers:
            ax.xaxis.set_major_locator(ticker.MultipleLocator(2))

    plt.title(title)

    has_kvar = bool(stats_a) and ("k_total_var" in stats_a[0])
    if has_kvar:
        y2 = [by_a[L]["k_total_var"] for L in layers]
        ax2 = ax.twinx()
        ax2.plot(layers, y2, marker="^", linestyle="--", color="tab:red", label="Total variance of K")
        ax2.set_ylim(0, 20000)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="best")
    else:
        if stats_b:
            ax.legend()
    plt.tight_layout()
    plt.savefig(_append_suffix(os.path.join(outdir, fname), suffix), dpi=300)
    plt.close()

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
    
    fig.tight_layout(rect=[0, 0.01, 0.88, 0.95])
    cbar_ax = fig.add_axes([0.90, 0.10, 0.02, 0.80])
    fig.colorbar(im, cax=cbar_ax, label="Attention prob")
    fig.suptitle(suptitle, fontsize=12, y=0.985)
    fig.savefig(_append_suffix(out_path, suffix), bbox_inches="tight", dpi=300)
    plt.close(fig)

def _pert_suffix(args, include_lower_attn=True):
    suffix = ""
    if args.rope:
        suffix += "__" + f"rope[{args.rope}]"
    if args.mask:
        suffix += "__" + f"mask[{args.mask}]"
    if args.random_init:
        suffix += "__" + "random_init"
    if include_lower_attn and args.lower_attn:
        segment = "lower_attn"
        cur_stop = getattr(args, "_current_stop_layer", None)
        if args.only_stop_layer is not None:
            segment += f"_only_stop[{args.only_stop_layer}]"
        elif cur_stop is not None:
            segment += f"_stop[{cur_stop}]"
        elif getattr(args, "stop_layers", None) is not None:
            b, e = args.stop_layers
            segment += f"_stops[{b}-{int(e)-1}]" 
        if args.lower_factor != 0.5:
            segment += f"_x{args.lower_factor}"
        suffix += "__" + segment
    if args.window_attn:
        segment = "window_attn"
        segment += f"_stop[{int(args.stop_window_layer)}]"
        segment += f"_size[{int(args.window_size)}]"
        suffix += "__" + segment
    if args.sink_idx != 0:
        suffix += "__" + f"sink[{args.sink_idx}]"
    if args.target_idx is not None:
        suffix += "__" + f"target[{args.target_idx}]"
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
    p.add_argument("--scan-interval", type=int, default=4, help="Interval to scan layers")
    # print mode
    p.add_argument("--print", dest="print_mode", choices=PRINT_CHOICES, default="qkv")
    p.add_argument("--qpos", default=None, help="Comma-separated list of query positions for --print qkv, eg. '256,512,768")
    p.add_argument("--sink-idx", type=int, default=0, help="Index of the sink in qkv mode")
    p.add_argument("--target-idx", type=int, default=None, help="Index of the target token for --print qkv")
    p.add_argument("--compare", action="store_true", help="Compare perturbed run vs baseline")
    # perturbations
    p.add_argument("--rope", default=None, help="Comma-separated list of token_idx=rope_pos, eg. '12=0,42=1")
    p.add_argument("--mask", default=None, choices=["upper"], help="Causal mask type")
    p.add_argument("--random-init", action="store_true", help="DO NOT load pretrained weights")
    # lower attn metrics
    p.add_argument("--lower-attn", action="store_true", help="Iteratively lower attention on sink key at each layer")
    p.add_argument("--stop-layers", type=int, nargs=2, default=None, help="Iteratively apply lower attention to all layers until the idx in [BEGIN, END) (0-indexed)")
    p.add_argument("--only-stop-layer", type=int, nargs="+", default=None, help="Only apply lower attention to these layers (0-indexed)")
    p.add_argument("--lower-factor", type=float, default=0.5, help="Factor to lower attention by")
    p.add_argument("--only-return-sink-value", action="store_true", help="When lowering attention, replace attention sum with only attn_to_sink * V_sink")
    # window attn metrics
    p.add_argument("--window-attn", action="store_true", help="Constrain attention to a local window for early layers")
    p.add_argument("--stop-window-layer", type=int, default=4, help="Apply windowed attention to layers [0, stop] inclusive.")
    p.add_argument("--window-size", type=int, default=4, help="Local attention window size")
    # subspaces
    p.add_argument("--find-value-subspace", action="store_true", help="Collect V at token 0 and non-sink positions and run PCA")
    p.add_argument("--find-key-subspace", action="store_true", help="Collect K at token 0 and non-sink positions and run PCA")
    p.add_argument("--pca-topk", type=int, default=6, help="Top-k PCA for groups of vectors")
    # output
    p.add_argument("--outdir", default="results")
    args = p.parse_args()

    _disable_packed_sequence_splitting()
    tok, model = load_model(args.model, args.device, args.dtype, random_init=args.random_init)

    if args.compare and (args.print_mode == "heatmap"):
        raise NotImplementedError("--compare is only available for --print qkv with limited perturbation options")

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

    min_len = min(ids.shape[1] for (ids, _bp) in tokenized)
    last_q = min_len - 1
    target_q = last_q if args.target_idx is None else max(0, min(last_q, args.target_idx))

    rope_str = args.rope
    rope_overrides = parse_overrides(rope_str) if rope_str else None
    base_suffix = _pert_suffix(args, include_lower_attn=False)

    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    job_summary_stats = []

    def _run_scan_pass(scan_layers, rope_overrides_local, need_qkv):
        per_layer_acc = {
            L: {"k_norm": [], "v_norm": [], "cos": [], "sink_attn": []}
            for L in scan_layers
        }
        per_layer_kvecs = {L: [] for L in scan_layers}
        per_layer_maps = {
            L: [] for L in scan_layers
        }
        scan_maps, scan_titles = [], []
        min_len = min(ids.shape[1] for (ids, _bp) in tokenized)
        last_q = min_len - 1
        target_q = last_q if args.target_idx is None else max(0, min(last_q, args.target_idx))
        sink_idx = max(0, min(last_q, args.sink_idx))

        store = {}
        handles = _capture_qkv_multi(model, scan_layers, store) if need_qkv else []
        try:
            for input_ids, base_pos in tqdm(tokenized, desc="Scanning prompts", position=0, leave=False):
                if need_qkv:
                    for L in list(store.keys()):
                        store[L].clear()
                pos_ids_perturbed, _ = apply_perturbations(base_pos, rope_overrides=rope_overrides_local)
                attns = prefill(model, input_ids, pos_ids_perturbed)

                for L in scan_layers:
                    if args.print_mode == "qkv":
                        qkv = store[L]
                        q, k, v = qkv["q"], qkv["k"], qkv["v"]
                        series, k_norm = compute_cosine_series(q, k, args.head, k_sink_idx=sink_idx, q_positions=[target_q])
                        v_norm = float(v[0, args.head, sink_idx].norm().item())
                        hm = _pick_head_with_caches(model, attns, L, args.head)
                        sink_slice = hm[4:, sink_idx]
                        per_layer_acc[L]["k_norm"].append(k_norm)
                        per_layer_acc[L]["v_norm"].append(v_norm)
                        per_layer_acc[L]["cos"].append(series[0][1])
                        per_layer_acc[L]["sink_attn"].append(float(sink_slice.mean().item()))
                        per_layer_kvecs[L].append(k[0, args.head, sink_idx].detach().cpu())
                    else:
                        hm = _pick_head_with_caches(model, attns, L, args.head)
                        per_layer_maps[L].append(hm[:min_len, :min_len])

        finally:
            for h in handles:
                try:
                    h.remove()
                except Exception:
                    pass
        
        if args.print_mode == "qkv":
            stats = []
            for L in sorted(per_layer_acc.keys()):
                acc = per_layer_acc[L]
                kvar = 0.0
                if per_layer_kvecs[L]:
                    Xk = torch.stack(per_layer_kvecs[L], dim=0)
                    _U, S_k, _Vh, _var, k_total_var = _pca_from_rows(Xk, k=1)
                    kvar = float(k_total_var)
                stats.append({
                    "k_norm": float(np.mean(acc["k_norm"])) if acc["k_norm"] else 0.0,
                    "v_norm": float(np.mean(acc["v_norm"])) if acc["v_norm"] else 0.0,
                    "cos": float(np.mean(acc["cos"])) if acc["cos"] else 0.0,
                    "layer": L, "head": args.head,
                    "target_q": target_q, 
                    "sink_attn": float(np.mean(acc["sink_attn"])) if acc["sink_attn"] else 0.0,
                    "k_total_var": kvar
                })
            return stats, scan_maps, scan_titles
        else:
            for L in sorted(per_layer_maps.keys()):
                if per_layer_maps[L]:
                    head_attn = np.mean(np.stack(per_layer_maps[L], axis=0), axis=0)
                    scan_maps.append(head_attn)
                    # DEBUG
                    src = "CACHE" if (hasattr(model, "_lower_attn_cache") and L in model._lower_attn_cache) else "ATTNS"
                    scan_titles.append(f"Layer {L} Head {args.head} [{src}]")
            return [], scan_maps, scan_titles

    stop_jobs = [("baseline", None)]
    if args.lower_attn:
        if args.stop_layers and args.only_stop_layer:
            raise ValueError("--stop-layers and --only-stop-layer cannot be used together.")
        if args.only_stop_layer:
            stop_jobs = [("lower", int(L)) for L in args.only_stop_layer]
        elif args.stop_layers:
            b, e = args.stop_layers
            b = max(0, int(b))
            e = min(num_layers, int(e))
            stop_jobs += [("lower", L) for L in range(b, e)]
        else:
            stop_jobs = [("lower", num_layers - 1)]
    elif args.window_attn:
        stop_jobs = [("window", int(args.stop_window_layer))]

    for _job_kind, _job_stop in stop_jobs:
        args._current_stop_layer = _job_stop
        pert_suffix = _pert_suffix(args, include_lower_attn=True)
        baseline_run = (_job_kind == "baseline")
        cur_suffix = base_suffix if baseline_run else pert_suffix

        lower_attn_handles = []
        window_attn_handles = []
        if args.lower_attn and not baseline_run:
            if args.only_stop_layer is not None:
                target_layers = [int(L) for L in args.only_stop_layer]
            else:
                target_layers = [L for L in range(num_layers) if L <= _job_stop]
            if target_layers:
                print(f"[lower_attn] target_layers: {target_layers}")
                lower_attn_handles = _install_lower_attn_hooks(
                    model, layers=target_layers, factor=args.lower_factor, sink_k_idx=args.sink_idx,
                    only_return_sink_value=args.only_return_sink_value
                )

        elif args.window_attn and not baseline_run:
            w_stop = int(getattr(args, "stop_window_layer", 4))
            w_layers = [L for L in range(num_layers) if L <= w_stop]
            if w_layers:
                window_attn_handles = _install_window_attn_hooks(
                    model, layers=w_layers, window_size=args.window_size
                )

        scan_stats = [] # aggregated per layer
        scan_stats_base = []
        scan_maps, scan_titles = [], []
        summary = None

        if args.scan:
            # perturbed pass
            scan_layers = list(range(0, num_layers, args.scan_interval))
            if not scan_layers or scan_layers[-1] != num_layers - 1:
                scan_layers.append(num_layers - 1)
            per_layer_acc = {
                L: {"k_norm": [], "v_norm": [], "cos": [], "sink_attn": []}
                for L in scan_layers
            }
            need_qkv = (args.print_mode == "qkv")
            scan_stats, scan_maps, scan_titles = _run_scan_pass(scan_layers, rope_overrides, need_qkv)
            
            # baseline pass
            if args.compare and args.print_mode == "qkv":
                for pair in lower_attn_handles + window_attn_handles:
                    for h in pair:
                        try:
                            h.remove()
                        except Exception:
                            pass
                lower_attn_handles = []
                window_attn_handles = []
                if hasattr(model, "_lower_attn_cache"):
                    model._lower_attn_cache = {}

                scan_stats_base, _m, _t = _run_scan_pass(scan_layers, None, True)

            if args.print_mode == "qkv" and len(scan_stats) > 0:
                scan_stats = sorted(scan_stats, key=lambda s: s["layer"])
                q_label = f"Q[{args.target_idx}]" if args.target_idx is not None else "Q[last]"
                if args.compare and len(scan_stats_base) > 0:
                    scan_stats_base = sorted(scan_stats_base, key=lambda s: s["layer"])
                    _plot_progression(
                        scan_stats, args.outdir, key="k_norm", ylabel=f"||K[{args.sink_idx}]||",
                        title=f"K-norm progression (token={args.sink_idx}) (compare)", fname="scan_knorm_compare.png", suffix=cur_suffix,
                        stats_b=scan_stats_base
                    )
                    _plot_progression(
                        scan_stats, args.outdir, key="v_norm", ylabel=f"||V[{args.sink_idx}]||",
                        title=f"V-norm progression (token={args.sink_idx}) (compare)", fname="scan_vnorm_compare.png", suffix=cur_suffix,
                        stats_b=scan_stats_base
                    )
                    _plot_progression(
                        scan_stats, args.outdir, key="cos", ylabel=f"cos({q_label}, K[{args.sink_idx}])",
                        title=f"Cosine to K[{args.sink_idx}] across layers (compare)", fname="scan_cosine_compare.png", suffix=cur_suffix,
                        stats_b=scan_stats_base, is_cos=True
                    )
                    # _log_row_summary("[Perturbed] ", scan_stats)
                    # _log_row_summary("[Baseline] ", scan_stats_base)
                else:
                    _plot_progression(
                        scan_stats, args.outdir, key="k_norm", ylabel=f"||K[{args.sink_idx}]||",
                        title=f"K-norm progression (token={args.sink_idx})", fname="scan_knorm.png", suffix=cur_suffix
                    )
                    _plot_progression(
                        scan_stats, args.outdir, key="v_norm", ylabel=f"||V[{args.sink_idx}]||",
                        title=f"V-norm progression (token={args.sink_idx})", fname="scan_vnorm.png", suffix=cur_suffix
                    )
                    _plot_progression(
                        scan_stats, args.outdir, key="cos", ylabel=f"cos({q_label}, K[{args.sink_idx}])",
                        title=f"Cosine to K[{args.sink_idx}] across layers", fname="scan_cosine.png", suffix=cur_suffix, is_cos=True
                    )
                    # _log_row_summary("[Perturbed] ", scan_stats)
            
            elif args.print_mode == "heatmap" and len(scan_maps) > 0:
                grid_title = "Attention heatmaps"
                if rope_overrides:
                    spec = ",".join(f"{ti}->{rp}" for ti, rp in rope_overrides)
                    grid_title += f" - rope({spec})"
                if args.mask:
                    grid_title += f" - mask({args.mask})"
                if args.random_init:
                    grid_title += " - [random init]"
                if args.lower_attn and not baseline_run:
                    segment = "lower_attn"
                    if args.only_stop_layer is not None:
                        segment += f" only_stop[{args.only_stop_layer}]"
                    elif args._current_stop_layer is not None:
                        segment += f" stop[{args._current_stop_layer}]"
                    grid_title += f" - {segment} (x{args.lower_factor})"
                if args.window_attn and not baseline_run:
                    grid_title += f" - window_attn stop[{int(args.stop_window_layer)}] w{int(args.window_size)}"

                _plot_heatmap_grid(
                    scan_maps, scan_titles, os.path.join(args.outdir, "scan_heatmaps.png"), rows=4,
                    suffix=cur_suffix, suptitle=grid_title,
                )
            
            if summary is not None:
                print(f"[Summary] sink attention at layer {num_layers - 1}: {summary:.4f} | stop at {args._current_stop_layer}")
                if args.lower_attn:
                    if baseline_run:
                        job_summary_stats.append({"layer": -1, "sink_attn": float(summary)})
                    elif args._current_stop_layer is not None:
                        job_summary_stats.append({"layer": int(args._current_stop_layer), "sink_attn": float(summary)})

        else:
            if args.find_value_subspace or args.find_key_subspace:
                target_layer = int(args.layer)
                store = {}
                handles = _capture_qkv_multi(model, [target_layer], store)
                field = "v" if args.find_value_subspace else "k"
                VAR = "V" if field == "v" else "K"
                try:
                    x_tok0, x_tok1, x_tok8, x_tokq = [], [], [], []
                    cos0, cos1, cos8 = [], [], []
                    r_tok0, r_tok1, r_tok8 = [], [], []
                    for input_ids, base_pos in tqdm(tokenized, desc=f"Collecting {VAR} for PCA", position=0, leave=False):
                        _ = prefill(model, input_ids, base_pos)
                        qkv = store.get(target_layer, None)
                        if not qkv:
                            continue
                        Q = qkv["q"]
                        X = qkv[field]
                        H = qkv["residual"]
                        assert X.shape[2] >= 9, "There must be at least 8 tokens in each sequence"
                        x0 = X[0, args.head, 0]
                        x1 = X[0, args.head, 1]
                        x8 = X[0, args.head, 8]
                        xq = Q[0, args.head, target_q]
                        x_tok0.append(x0.clone())
                        x_tok1.append(x1.clone())
                        x_tok8.append(x8.clone())
                        x_tokq.append(xq.clone())
                        if args.find_key_subspace:
                            # Gather residuals
                            assert H.shape[1] >= 9, "There must be at least 8 tokens in each sequence for residuals"
                            r_tok0.append(H[0, 0].clone()); r_tok1.append(H[0, 1].clone()); r_tok8.append(H[0, 8].clone())
                            # Gather cosines
                            for ks, bucket in [(0, cos0), (1, cos1), (8, cos8)]:
                                series, _ = compute_cosine_series(
                                    Q, X, args.head, k_sink_idx=ks, q_positions=[target_q]
                                )
                                bucket.append(series[0][1])
                finally:
                    for h in handles:
                        try: h.remove()
                        except Exception: pass
                
                assert len(x_tok0) == len(x_tok1) == len(x_tok8) > 0, f"No {VAR} vectors found"
                X0 = torch.stack(x_tok0, dim=0); X1 = torch.stack(x_tok1, dim=0); X8 = torch.stack(x_tok8, dim=0)
                XQ = torch.stack(x_tokq, dim=0)
                
                if args.find_key_subspace and r_tok0:
                    R0 = torch.stack(r_tok0, dim=0); R1 = torch.stack(r_tok1, dim=0); R8 = torch.stack(r_tok8, dim=0)
                else:
                    R0 = R1 = R8 = None

                k = int(args.pca_topk)

                U0, S0, Vh0, var0, K0_total_var = _pca_from_rows(X0, k)
                U1, S1, Vh1, var1, K1_total_var = _pca_from_rows(X1, k)
                U8, S8, Vh8, var8, K8_total_var = _pca_from_rows(X8, k)
                UQ, SQ, VhQ, varQ, Q_total_var = _pca_from_rows(XQ, k)

                print(f"[PCA: {VAR}] token 0 (sink) total variance: {K0_total_var:.4f}")
                print(f"[PCA: {VAR}] token 1 (non-sink) total variance: {K1_total_var:.4f}")
                print(f"[PCA: {VAR}] token 8 (non-sink) total variance: {K8_total_var:.4f}")
                print(f"[PCA: Q] targt Q (target_q={target_q}) total variance: {Q_total_var:.4f}")

                if args.find_key_subspace:
                    # raw cosine scores
                    def _mean_or_zero(xs): return float(np.mean(xs)) if xs else 0.0
                    print(
                        f"[COS] mean cos(Q, K0) = {_mean_or_zero(cos0):.4f} | "
                        f"[COS] mean cos(Q, K1) = {_mean_or_zero(cos1):.4f} | "
                        f"[COS] mean cos(Q, K8) = {_mean_or_zero(cos8):.4f}" 
                    )

                    # centered cosine scores
                    XQ_c = XQ - XQ.mean(dim=0, keepdim=True)
                    X0_c = X0 - X0.mean(dim=0, keepdim=True)
                    X1_c = X1 - X1.mean(dim=0, keepdim=True)
                    X8_c = X8 - X8.mean(dim=0, keepdim=True)
                    cos_c0 = F.cosine_similarity(XQ_c, X0_c, dim=1).numpy()
                    cos_c1 = F.cosine_similarity(XQ_c, X1_c, dim=1).numpy()
                    cos_c8 = F.cosine_similarity(XQ_c, X8_c, dim=1).numpy()
                    print(
                        f"[COS_centered] mean cos(Q, K0) = {float(cos_c0.mean().item()):.4f} | "
                        f"[COS_centered] mean cos(Q, K1) = {float(cos_c1.mean().item()):.4f} | "
                        f"[COS_centered] mean cos(Q, K8) = {float(cos_c8.mean().item()):.4f}"
                    )

                    # decomposition of Q along mean(K0)
                    _mean_dir_decomposition(XQ, X0, VhQ, "K0", k)
                    _mean_dir_decomposition(XQ, X1, VhQ, "K1", k)
                    _mean_dir_decomposition(XQ, X8, VhQ, "K8", k)

                    # visualize 3D PCA
                    Y0 = (X0_c @ Vh0[:3].T).numpy()
                    Y1 = (X1_c @ Vh1[:3].T).numpy()
                    Y8 = (X8_c @ Vh8[:3].T).numpy()

                    R = max(np.abs(Y0).max(), np.abs(Y1).max(), np.abs(Y8).max())

                    fig = plt.figure(figsize=(9,3))
                    for idx, (Y, title) in enumerate(
                        zip([Y0, Y1, Y8], ["K0 subspace", 'K1 subspace', 'K8 subspace'])
                    ):
                        ax = fig.add_subplot(1, 3, idx + 1, projection='3d')
                        ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], s=5)
                        ax.set_title(title)
                        ax.set_xlim(-R, R); ax.set_ylim(-R, R); ax.set_zlim(-R, R)
                        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")

                    fig.tight_layout()
                    out_path = os.path.join(args.outdir, f"pca_k_subspaces_L{target_layer}_H{args.head}.png")
                    fig.savefig(out_path, dpi=300)
                    plt.close(fig)

                    # visualize Q in bias direction + residual PC
                    bias_sets = [
                        (_q_bias_3d_projection(XQ, X0, topk=2), "Q energy along dimensions of K0"),
                        (_q_bias_3d_projection(XQ, X1, topk=2), "Q energy along dimensions of K1"),
                        (_q_bias_3d_projection(XQ, X8, topk=2), "Q energy along dimensions of K8"),
                    ]
                    Ys = [Y for (Y, _t) in bias_sets if Y is not None]
                    if Ys:
                        max_fb = max(float(Y[:, 0].max()) for Y in Ys)
                        R_bias = max_fb * 1.05
                        fig = plt.figure(figsize=(12, 3.5))
                        for idx, (Y, title) in enumerate(bias_sets):
                            if Y is None:
                                continue
                            ax = fig.add_subplot(1, 3, idx + 1, projection="3d")
                            frac_bias = Y[:, 0]
                            ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], s=8)

                            local_bias_mean = float(frac_bias.mean())
                            yy, zz = np.meshgrid(
                                np.linspace(0.0, 1.0, 2),
                                np.linspace(0.0, 1.0, 2),
                            )
                            xx = np.full_like(yy, local_bias_mean)
                            ax.plot_surface(xx, yy, zz, alpha=0.12, color="gray")
                            ax.set_title(f"{title}\n(plane: mean bias energy fraction)", fontsize=9)

                            ax.set_xlim(0.0, R_bias); ax.set_ylim(0.0, 1.0); ax.set_zlim(0.0, 1.0)
                            ax.set_xlabel("K bias"); ax.set_ylabel("residual PC1"); ax.set_zlabel("residual PC2")
                        fig.subplots_adjust(wspace=0.35, right=0.97, left=0.10)
                        fig.tight_layout()
                        out_path = os.path.join(args.outdir, f"pca_q_bias_energy_L{target_layer}_H{args.head}.png")
                        fig.savefig(out_path, dpi=300)
                        plt.close(fig)

                    # residual decomposition w.r.t K0 mean direction
                    mu0 = X0.mean(dim=0)
                    r_sink = _compute_residual_sink_direction(
                        model, target_layer, args.head, mu0
                    )
                    frac_R0 = _energy_fraction_along_direction(R0, r_sink); frac_R1 = _energy_fraction_along_direction(R1, r_sink); frac_R8 = _energy_fraction_along_direction(R8, r_sink)
                    m0, s0 = float(frac_R0.mean()), float(frac_R0.std())
                    m1, s1 = float(frac_R1.mean()), float(frac_R1.std())
                    m8, s8 = float(frac_R8.mean()), float(frac_R8.std())
                    print(
                        f"[Residual] layer={target_layer} head={args.head} | "
                        f"frac residual energy along canonical sink direction: "
                        f"tok0={m0:.4f}±{s0:.4f} | tok1={m1:.4f}±{s1:.4f} | tok8={m8:.4f}±{s8:.4f}"
                    )
                
        for pair in lower_attn_handles + window_attn_handles:
            for h in pair:
                try:
                    h.remove()
                except Exception:
                    pass

    if args.lower_attn and len(job_summary_stats) > 0 and (args.stop_layers is not None) and (args.only_stop_layer is None):
        job_summary_stats = sorted(job_summary_stats, key=lambda s: s["layer"])
        _prev_stop = getattr(args, "_current_stop_layer", None)
        args._current_stop_layer = None
        agg_suffix = _pert_suffix(args, include_lower_attn=True)
        args._current_stop_layer = _prev_stop
        title_str = f"Sink attention score progression for the last query token (x{args.lower_factor})"
        _plot_progression(
            job_summary_stats, args.outdir, key="sink_attn",
            ylabel="Sink attention score", title=title_str,
            fname="scan_sink_attn.png", suffix=agg_suffix, is_cos=True
        )

if __name__ == "__main__":
    main()