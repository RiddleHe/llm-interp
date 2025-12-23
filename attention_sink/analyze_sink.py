import argparse, os, numpy as np, torch, matplotlib.pyplot as plt, math, copy
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

def _get_hidden_from_hook_args(args, kwargs):
    if kwargs and "hidden_states" in kwargs:
        return kwargs["hidden_states"]
    return args[0]

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

def apply_perturbations(base_pos, rope_overrides=None, mask=None):
    pos_ids = base_pos
    applied = {}
    if rope_overrides:
        pos_ids = apply_rope_overrides(base_pos, rope_overrides)
        applied["rope"] = True
    if mask:
        applied["mask"] = True
    return pos_ids, applied

def _register_block_raw_captures(model, L, cache):
    handles = []
    h = model.model.layers[L].register_forward_pre_hook(
        lambda m, a, k, _c=cache: _c.setdefault("raw_L", []).append(
            _get_hidden_from_hook_args(a, k).detach().float().cpu(),
        ),
        with_kwargs=True
    )
    handles.append(h)

    return handles

# PCA

def _svd_centered(X, full_matrices=False):
    Xc = X - X.mean(dim=0, keepdim=True)
    U, S, Vh = torch.linalg.svd(Xc, full_matrices=full_matrices)
    var = S ** 2
    total = var.sum().clamp_min(EPS)
    return U, S, Vh, var, float(total.item())

def _pca_from_rows(X, k):
    U, S, Vh, var, total = _svd_centered(X, full_matrices=False)
    k = min(k, Vh.shape[0])
    rel = (var[:k] / total).cpu().numpy()
    return U[:, :k], S[:k], Vh[:k, :], rel, total

def _bias_3d_projection(X, bias_vec, topk=2):
    if X is None or bias_vec is None:
        return None
    b = F.normalize(bias_vec, dim=0)
    if float((b ** 2).sum().item()) <= EPS:
        return None

    total_energy = (X ** 2).sum(dim=1, keepdim=True).clamp_min(EPS)
    bias_coord = X @ b
    E_bias = (bias_coord ** 2).unsqueeze(1)

    X_parallel = bias_coord.unsqueeze(1) * b.unsqueeze(0)
    X_res = X - X_parallel
    X_res_c = X_res - X_res.mean(dim=0, keepdim=True)

    _U, _S, Vh = torch.linalg.svd(X_res_c, full_matrices=False)
    k = min(topk, Vh.shape[0])
    if k < 2:
        return None
    V_res = Vh[:k]
    
    res_coords = X_res @ V_res.T
    E_res = res_coords ** 2

    frac_bias = E_bias / total_energy
    frac_res = E_res / total_energy
    frac = torch.cat([frac_bias, frac_res], dim=1)
    return frac.cpu().numpy()

def _frac_energy_on_direction(X, u):
    if X is None or X.numel() == 0:
        return 0.0
    X = X.to(dtype=torch.float32)
    u = u.to(dtype=torch.float32, device=X.device)
    u = F.normalize(u, dim=0)
    denom = (X ** 2).sum(dim=1).clamp_min(EPS)
    num = (X @ u) ** 2
    return float((num / denom).mean().item())

def _mean_dir_decomposition(XQ, XK, VhQ, label, k):
    mu = XK.mean(dim=0)
    mu_norm_sq = float((mu ** 2).sum().item())
    if mu_norm_sq <= EPS:
        return 0.0
    return _frac_energy_on_direction(XQ, mu)

def _set_overlap_ratio(A, B, k):
    k = int(k)
    if k <= 0:
        return 0.0
    if not A or not B:
        return 0.0
    return float(len(A & B) / float(k))

def _fracmass_on_set(Z, idx_set):
    if Z is None or Z.numel() == 0 or (not idx_set):
        return 0.0
    Z = Z.to(dtype=torch.float32)
    denom = (Z ** 2).sum(dim=-1).clamp_min(EPS)
    idx = torch.tensor(sorted(idx_set), dtype=torch.long, device=Z.device)
    num = (Z[:, idx] ** 2).sum(dim=1)
    return float((num / denom).mean().item())

def _topk_list_from_scores(scores, k):
    if scores is None or scores.numel() == 0:
        return []
    k = int(max(0, min(int(k), int(scores.numel()))))
    if k <= 0:
        return []
    idx = torch.topk(scores, k=k, largest=True).indices.detach().cpu().tolist()
    return [int(i) for i in idx]

def _topk_set_from_scores(scores, k):
    return set(_topk_list_from_scores(scores, k))
    
def _topk_activated_set(Z, k):
    if Z is None or Z.numel() == 0:
        return set()
    scores = (Z.to(dtype=torch.float32) ** 2).mean(dim=0)
    return _topk_set_from_scores(scores, k)

def _topk_activated_list_abs(Z, k):
    if Z is None or Z.numel() == 0:
        return []
    scores = Z.to(dtype=torch.float32).abs().mean(dim=0)
    return _topk_list_from_scores(scores, k)

def _k_for_energy_frac(X, fracs=(0.25, 0.75, 0.99)):
    if X is None or X.numel() == 0:
        return [0 for _ in fracs]
    X = X.to(dtype=torch.float32)

    e = (X ** 2).mean(dim=0) # per dim energy
    total = float(e.sum().clamp_min(EPS).item())

    e_sorted, _ = torch.sort(e, descending=True)
    cdf = torch.cumsum(e_sorted, dim=0) / total

    ks = []
    for f in fracs:
        f = float(f)
        if f <= 0:
            ks.append(0)
            continue
        if f >= 1:
            ks.append(int(cdf.numel()))
            continue
        idx = int(torch.searchsorted(cdf, torch.tensor(f, device=cdf.device), right=False).item())
        ks.append(min(idx + 1, int(cdf.numel())))
    return ks

def _k_for_energy_frac_str(X, fracs=(0.25, 0.75, 0.99)):
    ks = _k_for_energy_frac(X, fracs=fracs)
    return _print_list(str(int(k)) for k in ks)

def _pos_count(Z):
    return float((Z > 0).sum(dim=1).float().mean().item())

def _pos_ratio(Z):
    return float((Z > 0).to(torch.float32).mean().item())

def _mean_vec_norm(X):
    if X is None or X.numel() == 0:
        return 0.0
    return float(X.norm(dim=-1).mean().item())

def _mean_or_zero(xs): return float(np.mean(xs)) if xs else 0.0

def _cloud_radius(Z):
    if Z is None or Z.numel() == 0:
        return 0.0
    Z2 = Z.reshape(-1, Z.shape[-1]) # [N, D]
    if Z2.size(0) < 2 or Z2.size(-1) == 0:
        return 0.0
    Z2 = F.normalize(Z2, dim=-1) # discard influence of length
    Zc = Z2 - Z2.mean(dim=0, keepdim=True)
    D = float(Zc.size(-1)) # D
    return float((Zc.pow(2).sum(dim=-1).mean() / D).sqrt().item())

def _component_direction_stats(R, A, M):
    stats = {
        "spread_residual": _cloud_radius(R),
        "spread_attn": _cloud_radius(A),
        "spread_mlp": _cloud_radius(M),
    }
    return stats

def _activation_entropy(Z):
    Z2 = (Z ** 2).mean(dim=0)
    p = Z2 / Z2.sum().clamp_min(EPS)
    return float(-(p * (p + EPS).log()).sum().item())

def _mean_topk_column_norm(Wd, idx_set):
    if not idx_set:
        return 0.0
    idx = torch.tensor(sorted(idx_set), dtype=torch.long, device=Wd.device)
    return float(Wd[:, idx].norm(dim=0).mean().item())

# decomposition of input functions

def _collect_raw_components_for_layer(model, layer_idx, tok_idxs, tokenized):
    cache_raw = {}
    store = {}
    handles = []
    handles += _register_block_raw_captures(model, layer_idx, cache_raw)
    handles += _capture_qkv_multi(
        model,
        [layer_idx],
        store,
        capture_attn_out=True,
        capture_mlp_out=True,
    )

    R_tok = {t: [] for t in tok_idxs}
    A_tok = copy.deepcopy(R_tok); M_tok = copy.deepcopy(R_tok); Y_tok = copy.deepcopy(R_tok)

    try:
        for input_ids, base_pos in tqdm(
            tokenized,
            desc=f"Collecting raw R/A/M at L={layer_idx}",
            leave=False,
        ):
            for L in list(store.keys()):
                store[L].clear()
            _ = prefill(model, input_ids, base_pos)

            if "raw_L" not in cache_raw or not cache_raw["raw_L"]:
                continue
            raw_L = cache_raw["raw_L"][-1][0]

            entry = store.get(layer_idx, None)
            if entry is None:
                continue
            attn_out = entry.get("attn_out", None)
            mlp_out = entry.get("mlp_out", None)
            if attn_out is None or mlp_out is None:
                continue
            attn_out = attn_out[0]
            mlp_out = mlp_out[0]

            S = raw_L.shape[0]
            for t in tok_idxs:
                if t >= S:
                    continue
                r = raw_L[t].clone()
                a = attn_out[t].clone()
                m = mlp_out[t].clone()
                y = r + a + m
                R_tok[t].append(r)
                A_tok[t].append(a)
                M_tok[t].append(m)
                Y_tok[t].append(y)

    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass

    out = {}
    for name, bucket in [
        ("R", R_tok),
        ("A", A_tok),
        ("M", M_tok),
        ("Y", Y_tok),
    ]:
        out[name] = {}
        for t, vecs in bucket.items():
            if vecs:
                out[name][t] = torch.stack(vecs, dim=0)
    return out

def _collect_mlp_internals_for_layer(model, layer_idx, tok_idxs, tokenized):
    mlp = model.model.layers[layer_idx].mlp
    cache = {}

    def _mlp_hook(_module, inputs, output, _cache=cache):
        x = inputs[0]
        with torch.no_grad():
            g = _module.gate_proj(x)
            u = _module.up_proj(x)
            a = _module.act_fn(g)
            z = a * u
            _cache["x"] = x.detach().float().cpu()
            _cache["g"] = g.detach().float().cpu()
            _cache["u"] = u.detach().float().cpu()
            _cache["a"] = a.detach().float().cpu()
            _cache["z"] = z.detach().float().cpu()
            _cache["y"] = output.detach().float().cpu()

    h = mlp.register_forward_hook(_mlp_hook)

    X_tok = {t: [] for t in tok_idxs}
    G_tok = copy.deepcopy(X_tok)
    U_tok = copy.deepcopy(X_tok)
    A_tok = copy.deepcopy(X_tok)
    Z_tok = copy.deepcopy(X_tok)
    Y_tok = copy.deepcopy(X_tok)

    try:
        for input_ids, base_pos in tqdm(
            tokenized,
            desc=f"Collecting MLP internals at L={layer_idx}",
            leave=False,
        ):
            cache.clear()
            _ = prefill(model, input_ids, base_pos)
            assert all(k in cache for k in ["x", "g", "u", "a", "z", "y"])

            x = cache["x"][0]
            g = cache["g"][0]
            u = cache["u"][0]
            a = cache["a"][0]
            z = cache["z"][0]
            y = cache["y"][0]
            S = int(x.shape[0])
            for t in tok_idxs:
                assert t < S
                X_tok[t].append(x[t].clone())
                G_tok[t].append(g[t].clone())
                U_tok[t].append(u[t].clone())
                A_tok[t].append(a[t].clone())
                Z_tok[t].append(z[t].clone())
                Y_tok[t].append(y[t].clone())

    finally:
        try:
            h.remove()
        except Exception:
            pass

    out = {"X": {}, "G": {}, "U": {}, "A": {}, "Z": {}, "Y": {}}
    for name, bucket in [
        ("X", X_tok),
        ("G", G_tok),
        ("U", U_tok),
        ("A", A_tok),
        ("Z", Z_tok),
        ("Y", Y_tok),
    ]:
        for t, vecs in bucket.items():
            if vecs:
                out[name][t] = torch.stack(vecs, dim=0)

    return out

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

def _capture_qkv_multi(model, layer_idxs, cache, capture_attn_out=False, capture_mlp_out=False):
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

        h_pre = attn.register_forward_pre_hook(_mk_hook(L, attn), with_kwargs=True)
        handles.append(h_pre)

        if capture_attn_out:
            def _mk_fwd_hook(L_):
                def fwd_hook(_module, args, output):
                    attn_out = output[0]
                    cache[L_]["attn_out"] = attn_out.detach().float().cpu()
                return fwd_hook
            h_fwd = attn.register_forward_hook(_mk_fwd_hook(L))
            handles.append(h_fwd)

        if capture_mlp_out:
            mlp = model.model.layers[L].mlp
            def _mk_mlp_fwd_hook(L_):
                def fwd_hook(_module, args, output, _cache=cache):
                    entry = _cache.setdefault(L_, {})
                    entry["mlp_out"] = output.detach().float().cpu()
                return fwd_hook
            h_mlp = mlp.register_forward_hook(_mk_mlp_fwd_hook(L))
            handles.append(h_mlp)

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
    return pick_head(attns, layer_idx, head_idx)

# Plotting functions

def _plot_progression(stats_a, outdir, key, ylabel, title, fname, suffix="", stats_b=None, mode=None):
    layers = sorted({s["layer"] for s in stats_a})
    by_a = {s["layer"]: s for s in stats_a}
    y_a = [by_a[L][key] for L in layers]
    plt.figure(figsize=(7, 3.5))
    ax = plt.gca()
    ax.plot(layers, y_a, marker="o", label="Perturbed" if stats_b else "Norm")

    if stats_b:
        by_b = {s["layer"]: s for s in stats_b}
        y_b = [by_b[L][key] for L in layers]
        ax.plot(layers, y_b, marker="s", label="Baseline")

    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    if mode == "cos":
        ax.set_ylim(0, 1)
    elif mode == "res":
        ax.set_ylim(0, 10)
    else:
        ax.set_ylim(0, 100)

    if len(layers) > 0:
        ax.set_xlim(min(layers), max(layers))
        ax.set_xticks(layers)
        ax.set_xticklabels(["baseline" if L == -1 else str(L) for L in layers], rotation=0)
        if -1 not in layers:
            ax.xaxis.set_major_locator(ticker.MultipleLocator(2))

    plt.title(title)

    base = f"{key.split('_')[0]}" # k_norm -> k, v_norm -> v, out_norm -> out
    var_key = f"{base}_spread"
    has_var = bool(stats_a) and (var_key in stats_a[0])
    if has_var:
        y2 = [by_a[L][var_key] for L in layers]
        ax2 = ax.twinx()
        ax2.plot(
            layers, y2, marker="^", linestyle="--", color="tab:red", 
            label="Normalized spread (radius of normalized vector cloud)"
        )
        ax2.set_ylim(0, 1e-1)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="best")
    else:
        if stats_b:
            ax.legend()
    plt.tight_layout()
    plt.savefig(_append_suffix(os.path.join(outdir, fname), suffix), dpi=300)
    plt.close()

def _plot_bias_energy_3d(bias_sets, out_path, x_label):
    Ys = [Y for (Y, _t) in bias_sets if Y is not None]
    if not Ys:
        return
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
        ax.plot_surface(xx, yy, zz, alpha=0.12, color="green")
        ax.set_title(title)

        ax.set_xlim(0.0, R_bias)
        ax.set_ylim(0.0, 1.0)
        ax.set_zlim(0.0, 1.0)
        ax.set_xlabel(x_label)
        ax.set_ylabel("residual PC1")
        ax.set_zlabel("residual PC2")

    fig.subplots_adjust(wspace=0.35, right=0.97, left=0.10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
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
    if args.sink_idx != 0:
        suffix += "__" + f"sink[{args.sink_idx}]"
    if args.target_idx is not None:
        suffix += "__" + f"target[{args.target_idx}]"
    return suffix

def _print_list(lst):
    return "[" + ", ".join(lst) + "]"

def _print_metrics_table(metrics, col_names, title=None, col_w=20):
    if title:
        print(title)
    if not metrics:
        print("[metrics] (empty)")
        return
    assert all(len(vals) == len(col_names) for _rn, vals, _k in metrics), \
        "Each metrics row must have the same number of values as col_names"
    
    row_name_w = max(len(r[0]) for r in metrics)
    print(" " * (row_name_w + 2) + "".join(name.rjust(col_w) for name in col_names))
    for row_name, vals, kind in metrics:
        print(row_name.ljust(row_name_w) + "  " + "".join(_fmt(v, kind).rjust(col_w) for v in vals))

def _fmt(v, kind):
    if v is None:
        return "N/A"
    if isinstance(v, str):
        return v
    if kind == "sci":
        return f"{float(v):.4e}"
    return f"{float(v):.4f}"

def _get(key, comp):
    return [float(s[key]) for s in comp]

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
    p.add_argument("--scan-interval", type=int, default=2, help="Interval to scan layers")
    # print mode
    p.add_argument("--print", dest="print_mode", choices=PRINT_CHOICES, default="qkv")
    p.add_argument("--qpos", default=None, help="Comma-separated list of query positions for --print qkv, eg. '256,512,768")
    p.add_argument("--sink-idx", type=int, default=0, help="Index of the sink in qkv mode")
    p.add_argument("--target-idx", type=int, default=None, help="Index of the target token for --print qkv")
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
    # subspaces
    p.add_argument("--find-value-subspace", action="store_true", help="Collect V at token 0 and non-sink positions")
    p.add_argument("--find-key-subspace", action="store_true", help="Collect K at token 0 and non-sink positions")
    p.add_argument("--find-mlp-subspace", action="store_true", help="Collect MLP internals at token 0 and non-sink positions")
    p.add_argument("--pca-topk", type=int, default=6, help="Top-k PCA for groups of vectors")
    # decomposition of input
    p.add_argument("--decompose-ln", action="store_true", help="For --layer L, decompose the norm input LN(residual + attn + mlp) at L")
    p.add_argument("--decompose-output", action="store_true", help="For --layer L, decompose the output (residual + attn + mlp) at L")
    # output
    p.add_argument("--outdir", default="results")
    args = p.parse_args()

    _disable_packed_sequence_splitting()
    tok, model = load_model(args.model, args.device, args.dtype, random_init=args.random_init)

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

    def _run_scan_pass(scan_layers, rope_overrides_local):
        per_layer_acc = {
            L: {"k_norm": [], "v_norm": [], "postln_norm": [], "out_norm": [], "cos": [], "sink_attn": []}
            for L in scan_layers
        }
        per_layer_kvecs = {L: [] for L in scan_layers}
        per_layer_vvecs = {L: [] for L in scan_layers}
        per_layer_postlnvecs = {L: [] for L in scan_layers}
        per_layer_outvecs = {L: [] for L in scan_layers}
        per_layer_maps = {L: [] for L in scan_layers}

        scan_maps, scan_titles = [], []
        min_len = min(ids.shape[1] for (ids, _bp) in tokenized)
        last_q = min_len - 1
        target_q = last_q if args.target_idx is None else max(0, min(last_q, args.target_idx))
        sink_idx = max(0, min(last_q, args.sink_idx))

        store = {}
        handles = _capture_qkv_multi(
            model, 
            scan_layers, 
            store,
            capture_mlp_out=True,
        )
        try:
            for input_ids, base_pos in tqdm(tokenized, desc="Scanning prompts", position=0, leave=False):
                for L in list(store.keys()):
                    store[L].clear()
                pos_ids_perturbed, _ = apply_perturbations(base_pos, rope_overrides=rope_overrides_local)
                attns = prefill(model, input_ids, pos_ids_perturbed)

                for L in scan_layers:
                    if args.print_mode == "qkv":
                        qkv = store[L]
                        q, k, v = qkv["q"], qkv["k"], qkv["v"]
                        H = qkv["residual"]
                        mlp_out = qkv["mlp_out"]
                        series, k_norm = compute_cosine_series(q, k, args.head, k_sink_idx=sink_idx, q_positions=[target_q])

                        v_sink = v[0, args.head, sink_idx]
                        h_sink = H[0, sink_idx]
                        out_sink = mlp_out[0, sink_idx]

                        v_norm = float(v_sink.norm().item())
                        postln_norm = float(torch.log(h_sink.norm().clamp_min(EPS)).item())
                        out_norm = float(torch.log(out_sink.norm().clamp_min(EPS)).item())

                        hm = _pick_head_with_caches(model, attns, L, args.head)
                        sink_slice = hm[4:, sink_idx]

                        per_layer_acc[L]["k_norm"].append(k_norm)
                        per_layer_acc[L]["v_norm"].append(v_norm)
                        per_layer_acc[L]["postln_norm"].append(postln_norm)
                        per_layer_acc[L]["out_norm"].append(out_norm)
                        per_layer_acc[L]["cos"].append(series[0][1])
                        per_layer_acc[L]["sink_attn"].append(float(sink_slice.mean().item()))

                        per_layer_kvecs[L].append(k[0, args.head, sink_idx].detach().cpu())
                        per_layer_vvecs[L].append(v_sink.detach().cpu())
                        per_layer_postlnvecs[L].append(h_sink.detach().cpu())
                        per_layer_outvecs[L].append(out_sink.detach().cpu())

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
                vvar = 0.0
                postlnvar = 0.0
                outvar = 0.0

                if per_layer_kvecs[L]:
                    Xk = torch.stack(per_layer_kvecs[L], dim=0)
                    kvar = _cloud_radius(Xk)
                if per_layer_vvecs[L]:
                    Xv = torch.stack(per_layer_vvecs[L], dim=0)
                    vvar = _cloud_radius(Xv)
                if per_layer_postlnvecs[L]:
                    Xh = torch.stack(per_layer_postlnvecs[L], dim=0)
                    postlnvar = _cloud_radius(Xh)
                if per_layer_outvecs[L]:
                    Xo = torch.stack(per_layer_outvecs[L], dim=0)
                    outvar = _cloud_radius(Xo)

                stats.append({
                    "k_norm": float(np.mean(acc["k_norm"])) if acc["k_norm"] else 0.0,
                    "v_norm": float(np.mean(acc["v_norm"])) if acc["v_norm"] else 0.0,
                    "postln_norm": float(np.mean(acc["postln_norm"])) if acc["postln_norm"] else -12.0,
                    "out_norm": float(np.mean(acc["out_norm"])) if acc["out_norm"] else -12.0,
                    "cos": float(np.mean(acc["cos"])) if acc["cos"] else 0.0,
                    "layer": L, "head": args.head,
                    "target_q": target_q, 
                    "sink_attn": float(np.mean(acc["sink_attn"])) if acc["sink_attn"] else 0.0,
                    "k_spread": kvar,
                    "v_spread": vvar,
                    "postln_spread": postlnvar,
                    "out_spread": outvar,
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

    for _job_kind, _job_stop in stop_jobs:
        args._current_stop_layer = _job_stop
        pert_suffix = _pert_suffix(args, include_lower_attn=True)
        baseline_run = (_job_kind == "baseline")
        cur_suffix = base_suffix if baseline_run else pert_suffix

        lower_attn_handles = []
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

        scan_stats = [] # aggregated per layer
        scan_maps, scan_titles = [], []
        summary = None

        if args.scan:
            # perturbed pass
            scan_layers = list(range(0, num_layers, args.scan_interval))
            scan_stats, scan_maps, scan_titles = _run_scan_pass(scan_layers, rope_overrides)

            if args.print_mode == "qkv" and len(scan_stats) > 0:
                scan_stats = sorted(scan_stats, key=lambda s: s["layer"])
                q_label = f"Q[{args.target_idx}]" if args.target_idx is not None else "Q[last]"
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
                    title=f"Cosine to K[{args.sink_idx}] across layers", fname="scan_cosine.png", suffix=cur_suffix,
                    mode="cos"
                )
                _plot_progression(
                    scan_stats, args.outdir, key="postln_norm", ylabel=f"log ||Attention_LN_out[{args.sink_idx}]||",
                    title=f"Layernorm output to attention norm progression (token={args.sink_idx})", fname="scan_postln_norm.png",
                    suffix=cur_suffix, mode="res",
                )
                _plot_progression(
                    scan_stats, args.outdir, key="out_norm", ylabel=f"log ||MLP_out[{args.sink_idx}]||",
                    title=f"MLP output activation norm progression (token={args.sink_idx})", fname="scan_mlp_out_norm.png",
                    suffix=cur_suffix, mode="res",
                )
            
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
            if args.find_key_subspace:
                target_layer = int(args.layer)
                store = {}
                handles = _capture_qkv_multi(
                    model, [target_layer], store,
                    # capture_attn_out=args.find_key_subspace,
                    # capture_mlp_out=args.find_key_subspace
                )
                raw_cache = {}
                handles += _register_block_raw_captures(model, target_layer, raw_cache)
                VAR = "K"
                try:
                    x_tok0, x_tok1, x_tok8, x_tokq = [], [], [], []
                    cos0, cos1, cos8 = [], [], []
                    r_tok0, r_tok1, r_tok8 = [], [], []
                    for input_ids, base_pos in tqdm(tokenized, desc=f"Collecting k for PCA", position=0, leave=False):
                        _ = prefill(model, input_ids, base_pos)
                        qkv = store.get(target_layer, None)
                        if not qkv:
                            continue
                        Q = qkv["q"]
                        X = qkv["k"]
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

                min_len = min(t.shape[1] for t in raw_cache["raw_L"])
                X_raw = torch.stack([t[:, :min_len] for t in raw_cache["raw_L"]], dim=0)[:, 0]
                assert X_raw.shape[1] >= 9, "There must be at least 8 tokens in each sequence"
                
                R0 = torch.stack(r_tok0, dim=0); R1 = torch.stack(r_tok1, dim=0); R8 = torch.stack(r_tok8, dim=0)

                k = int(args.pca_topk)

                UQ, SQ, VhQ, varQ, Q_total_var = _pca_from_rows(XQ, k)

                # centered cosine scores
                XQ_c = XQ - XQ.mean(dim=0, keepdim=True)
                X0_c = X0 - X0.mean(dim=0, keepdim=True)
                X1_c = X1 - X1.mean(dim=0, keepdim=True)
                X8_c = X8 - X8.mean(dim=0, keepdim=True)
                cos_c0 = F.cosine_similarity(XQ_c, X0_c, dim=1).detach().cpu().numpy()
                cos_c1 = F.cosine_similarity(XQ_c, X1_c, dim=1).detach().cpu().numpy()
                cos_c8 = F.cosine_similarity(XQ_c, X8_c, dim=1).detach().cpu().numpy()

                # frac of energy along dim
                frac_q_mu0 = _mean_dir_decomposition(XQ, X0, VhQ, "k0", k)
                frac_q_mu1 = _mean_dir_decomposition(XQ, X1, VhQ, "k1", k)
                frac_q_mu8 = _mean_dir_decomposition(XQ, X8, VhQ, "k8", k)

                metrics = [
                    ("spread_radius", [_cloud_radius(X0), _cloud_radius(X1), _cloud_radius(X8)], "sci"),
                    ("mean_norm", [_mean_vec_norm(X0), _mean_vec_norm(X1), _mean_vec_norm(X8)], "sci"),
                    ("cos(Q, K)", [_mean_or_zero(cos0), _mean_or_zero(cos1), _mean_or_zero(cos8)], "float"),
                    ("cos_centered(Q, K)", [float(np.mean(cos_c0)), float(np.mean(cos_c1)), float(np.mean(cos_c8))], "float"),
                    (f"frac_Q_along_mean_K", [frac_q_mu0, frac_q_mu1, frac_q_mu8], "float"),
                ]
                col_names = ["tok0", "tok1", "tok8"]
                title = f"[K metrics] layer={target_layer} head={args.head} target_q={target_q}"
                _print_metrics_table(metrics, col_names, title=title)

                # visualize Q in bias direction
                bias_sets = [
                    (_bias_3d_projection(XQ, X0.mean(dim=0), topk=2), "Q energy along dimensions of K0"),
                    (_bias_3d_projection(XQ, X1.mean(dim=0), topk=2), "Q energy along dimensions of K1"),
                    (_bias_3d_projection(XQ, X8.mean(dim=0), topk=2), "Q energy along dimensions of K8"),
                ]
                _plot_bias_energy_3d(
                    bias_sets,
                    os.path.join(args.outdir, f"pca_q_bias_energy_L{target_layer}_H{args.head}.png"),
                    x_label="K bias",
                )

            L = int(args.layer)
            Lm1 = max(0, L - 1)
            tok_idxs = [0, 1, 8]

            if args.find_mlp_subspace:
                mlp_data = _collect_mlp_internals_for_layer(model, L, tok_idxs, tokenized)
                assert 0 in mlp_data["Y"]

                # fetch down projection column vectors
                mlp = model.model.layers[L].mlp
                Wd = mlp.down_proj.weight.detach().to(dtype=torch.float32).cpu()
                Wg = mlp.gate_proj.weight.detach().to(dtype=torch.float32).cpu()
                d_model, d_mlp = int(Wd.shape[0]), int(Wd.shape[1])

                # fetch mean output direction
                Y0 = mlp_data["Y"][0].to(dtype=torch.float32)
                mu_y = Y0.mean(dim=0)
                u_hat = F.normalize(mu_y, dim=0)

                # calculate column scores
                alpha = (u_hat.unsqueeze(0) @ Wd).squeeze(0)
                alpha_abs = alpha.abs()

                k = 3
                k_fracs = (0.25, 0.75, 0.99)
                S_proj = _topk_set_from_scores(alpha_abs, k) # topk indices that align with mu_y

                col_names = ["tok0", "tok1", "tok8"]

                Z0 = mlp_data["Z"][0]; Z1 = mlp_data["Z"][1]; Z8 = mlp_data["Z"][8] 
                G0 = mlp_data["G"][0]; G1 = mlp_data["G"][1]; G8 = mlp_data["G"][8]
                U0 = mlp_data["U"][0]; U1 = mlp_data["U"][1]; U8 = mlp_data["U"][8]
                A0 = mlp_data["A"][0]; A1 = mlp_data["A"][1]; A8 = mlp_data["A"][8]
                X0 = mlp_data["X"][0]; X1 = mlp_data["X"][1]; X8 = mlp_data["X"][8]

                idx_sorted = sorted(S_proj)
                z_abs_mean = (Z0[:, idx_sorted].abs().mean(dim=0)).cpu().tolist()
                pairs = sorted(
                    zip(idx_sorted, z_abs_mean),
                    key=lambda x: -x[1],
                )   
                ref_idx = pairs[0][0]
                w_ref = F.normalize(Wd[:, ref_idx], dim=0)

                wdown_cos_to_top1 = []
                for i, _ in pairs:
                    wi = F.normalize(Wd[:, i], dim=0)
                    wdown_cos_to_top1.append(float(torch.dot(wi, w_ref).item()))

                # sanity checks
                diff0 = Z0 - A0 * U0
                z0_rel_l2 = float((
                    diff0.norm(dim=1) / Z0.norm(dim=1).clamp_min(EPS)
                ).mean().item())
                print(f"[Sanity check] reconstruction mean_rel_l2={z0_rel_l2:.4e}")

                S_act0 = _topk_activated_set(Z0, k) # should be the same as S_proj
                S_act1 = _topk_activated_set(Z1, k)
                S_act8 = _topk_activated_set(Z8, k)

                S_u0 = _topk_activated_set(U0, k)
                S_a0 = _topk_activated_set(A0, k)

                z_norm_vals = [
                    float(Z0.norm(dim=1).mean().item()),
                    float(Z1.norm(dim=1).mean().item()),
                    float(Z8.norm(dim=1).mean().item()),
                ]
                z_kfrac_vals = [
                    _k_for_energy_frac_str(Z0, k_fracs),
                    _k_for_energy_frac_str(Z1, k_fracs),
                    _k_for_energy_frac_str(Z8, k_fracs),
                ]
                z_topk_idx_vals = [
                    _print_list(map(str, _topk_activated_list_abs(Z0, k))),
                    _print_list(map(str, _topk_activated_list_abs(Z1, k))),
                    _print_list(map(str, _topk_activated_list_abs(Z8, k))),
                ]
                wd_aligned_topk_idx_vals = [
                    _print_list(map(str, sorted(S_proj))),
                    None,
                    None,
                ]
                wdown_cos_to_top1_vals = [
                    _print_list(f"{v:.2f}" for v in wdown_cos_to_top1),
                    None,
                    None,
                ]
                frac_mass_vals = [
                    _fracmass_on_set(Z0, S_proj),
                    _fracmass_on_set(Z1, S_proj),
                    _fracmass_on_set(Z8, S_proj),
                ]
                entropy_vals = [
                    _activation_entropy(Z0),
                    _activation_entropy(Z1),
                    _activation_entropy(Z8),
                ]
                colnorm_vals = [
                    _mean_topk_column_norm(Wd, S_act0),
                    _mean_topk_column_norm(Wd, S_act1),
                    _mean_topk_column_norm(Wd, S_act8),
                ]
                g_entropy_vals = [
                    _activation_entropy(G0),
                    _activation_entropy(G1),
                    _activation_entropy(G8),
                ]
                a_entropy_vals = [
                    _activation_entropy(A0),
                    _activation_entropy(A1),
                    _activation_entropy(A8),
                ]
                g_pos_ratio_vals = [
                    f"{int(round(_pos_count(G0)))}/{int(G0.size(1))}",
                    f"{int(round(_pos_count(G1)))}/{int(G1.size(1))}",
                    f"{int(round(_pos_count(G8)))}/{int(G8.size(1))}",
                ]
                u_norm_vals = [
                    float(U0.norm(dim=1).mean().item()),
                    float(U1.norm(dim=1).mean().item()),
                    float(U8.norm(dim=1).mean().item()),
                ]
                u_topk_idx_vals = [
                    _print_list(map(str, _topk_activated_list_abs(U0, k))),
                    _print_list(map(str, _topk_activated_list_abs(U1, k))),
                    _print_list(map(str, _topk_activated_list_abs(U8, k))),
                ]
                a_norm_vals = [
                    float(A0.norm(dim=1).mean().item()),
                    float(A1.norm(dim=1).mean().item()),
                    float(A8.norm(dim=1).mean().item()),
                ]
                a_kfrac_vals = [
                    _k_for_energy_frac_str(A0, k_fracs),
                    _k_for_energy_frac_str(A1, k_fracs),
                    _k_for_energy_frac_str(A8, k_fracs),
                ]
                a_topk_idx_vals = [
                    _print_list(map(str, _topk_activated_list_abs(torch.clamp(A0, min=0.0), k))),
                    _print_list(map(str, _topk_activated_list_abs(torch.clamp(A1, min=0.0), k))),
                    _print_list(map(str, _topk_activated_list_abs(torch.clamp(A8, min=0.0), k))),
                ]

                metrics_z = [
                    ("inter_z_norm", z_norm_vals, "sci"),
                    ("inter_z_entropy", entropy_vals, "float"),
                    ("inter_z_k@frac_energy [.25, .75, .99]", z_kfrac_vals, ""),
                    (f"inter_z_topk_act_idx k={k}", z_topk_idx_vals, ""),
                    (f"W_down_topk_aligned_vec_idx k={k}", wd_aligned_topk_idx_vals, ""),
                    (f"frac_act_on_aligned_vec k={k} (sq)", frac_mass_vals, "float"),
                    (f"W_down_cos_to_top1_aligned_vec k={k}", wdown_cos_to_top1_vals, ""),
                    (f"W_down_topk_act_norm k={k}", colnorm_vals, "float"),
                ]
                metrics_ga = [
                    ("gate_g_entropy", g_entropy_vals, "float"),
                    ("gate_g_pos/all", g_pos_ratio_vals, ""),
                    ("act_a_entropy", a_entropy_vals, "float"),
                    ("act_a_norm", a_norm_vals, "sci"),
                    (f"act_a_topk_act_idx k={k}", a_topk_idx_vals, ""),
                    (f"act_a_k@frac_energy [.25, .75, .99]", a_kfrac_vals, ""),
                ]
                metrics_u = [
                    ("up_u_norm", u_norm_vals, "sci"),
                    (f"up_u_tok_act_idx k={k}", u_topk_idx_vals, ""),
                ]

                print()
                _print_metrics_table(metrics_z, col_names, title=f"[MLP subspace: Z/W_d] layer={L}")
                print()
                _print_metrics_table(metrics_ga, col_names, title=f"[MLP subspace: G/A] layer={L}")
                print()
                _print_metrics_table(metrics_u, col_names, title=f"[MLP subspace: U] layer={L}")  

                g_top_idx = _topk_activated_list_abs(G0, k)
                gate_row_norms, frac_x0, frac_x1, frac_x8 = [], [], [], []
                for ii in g_top_idx:
                    w = Wg[ii, :]
                    gate_row_norms.append(float(w.norm().item()))
                    frac_x0.append(_frac_energy_on_direction(X0, w))
                    frac_x1.append(_frac_energy_on_direction(X1, w))
                    frac_x8.append(_frac_energy_on_direction(X8, w))

                col_names_g = [f"top{j+1} row" for j in range(len(g_top_idx))]
                idx_str = [str(i) for i in g_top_idx]
                metrics_g = [
                    ("idx", idx_str, ""),
                    ("||w||", gate_row_norms, "sci"),
                    ("frac_energy_X_tok0_on_vec", frac_x0, "float"),
                    ("frac_energy_X_tok1_on_vec", frac_x1, "float"),
                    ("frac_energy_X_tok8_on_vec", frac_x8, "float"),
                ]
                # _print_metrics_table(
                #     metrics_g,
                #     col_names_g,
                #     title=f"[Gate subspace] layer={L}",
                # )

            if args.decompose_output:
                raw_components = _collect_raw_components_for_layer(
                    model, L, tok_idxs, tokenized,
                )

                col_names = ["tok0", "tok1", "tok8"]
                Ys, Rs, As, Ms = [], [], [], []
                for t in tok_idxs:
                    Y = raw_components["Y"][t]
                    R_raw = raw_components["R"][t]
                    A_raw = raw_components["A"][t]
                    M_raw = raw_components["M"][t]
                    Ys.append(Y); Rs.append(R_raw); As.append(A_raw); Ms.append(M_raw)
                    assert Y is not None and Y.ndim == 2 and Y.shape[0] >= 2

                Y_norm = []; Y_spread = []
                for Y in Ys:
                    Y_norm.append(_mean_vec_norm(Y))
                    Y_spread.append(_cloud_radius(Y))

                comp = [] # component breakdown R/A/M
                for i in range(len(tok_idxs)):
                    comp_stats = _component_direction_stats(Rs[i], As[i], Ms[i])
                    comp_stats.update({
                        "norm_residual": _mean_vec_norm(Rs[i]),
                        "norm_attn": _mean_vec_norm(As[i]),
                        "norm_mlp": _mean_vec_norm(Ms[i]),
                    })
                    comp.append(comp_stats)

                metrics = [
                    ("norm_Y", Y_norm, "sci"),
                    ("spread_Y", Y_spread, "sci"),
                    ("norm_residual", _get("norm_residual", comp), "sci"),
                    ("norm_attn", _get("norm_attn", comp), "sci"),
                    ("norm_mlp", _get("norm_mlp", comp), "sci"),
                    ("spread_residual", _get("spread_residual", comp), "sci"),
                    ("spread_attn", _get("spread_attn", comp), "sci"),
                    ("spread_mlp", _get("spread_mlp", comp), "sci"),
                    # TODO: plot cos and covariance
                ]
                _print_metrics_table(metrics, col_names, title=f"[Block Output L={L}]")
                
        for pair in lower_attn_handles:
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
            fname="scan_sink_attn.png", suffix=agg_suffix,
            mode="cos"
        )

if __name__ == "__main__":
    main()