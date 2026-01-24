import argparse, os, numpy as np, torch, matplotlib.pyplot as plt, math, copy
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch.nn.functional as F
from tqdm import tqdm
from matplotlib import ticker
from mpl_toolkits.mplot3d import Axes3D
from contextlib import contextmanager
import random

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

def _write_hidden_to_hook_args(x_new, args, kwargs):
    if kwargs is not None and "hidden_states" in kwargs:
        kwargs["hidden_states"] = x_new
        return args, kwargs

    args = list(args)
    args[0] = x_new
    return tuple(args), kwargs

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

def _install_mlp_out_ablation_hook(
    model,
    layer_idx,
    mode,
    vec_idx=None,
    tok_ablate=0,
    tok_ref=8,
    clamp_alpha_max=1.0,
):
    handles = []
    mlp = model.model.layers[layer_idx].mlp

    Q = None
    if mode == "direction":
        Wd = mlp.down_proj.weight.detach().to(dtype=torch.float32)
        B = Wd[:, vec_idx]
        Q, _R = torch.linalg.qr(B, mode="reduced") # orthobasis [d_model, n_vec]
        Q = Q.contiguous()

    def _hook(_module, args, output, _Q=Q):
        y = output
        bsz, S, D = int(y.size(0)), int(y.size(1)), int(y.size(2))

        y2 = y.clone()

        if mode == "magnitude":
            v0 = y2[:, tok_ablate, :]
            v8 = y2[:, tok_ref, :]
            n0 = v0.norm(dim=-1).clamp_min(EPS)
            n8 = v8.norm(dim=-1)
            alpha = (n8 / n0).to(dtype=v0.dtype)
            alpha = alpha.clamp(max=clamp_alpha_max)
            y2[:, tok_ablate, :] = v0 * alpha.unsqueeze(-1)
            return y2

        if mode == "direction":
            v0 = y2[:, tok_ablate, :].to(dtype=torch.float32)
            Qd = _Q.to(device=v0.device, dtype=torch.float32)
            coeff = v0 @ Qd # [B, S, n_vec]
            proj = coeff @ Qd.T # [B, S, d_model]
            v_new = v0 - proj
            y2[:, tok_ablate, :] = v_new.to(dtype=v0.dtype)
            return y2

        return output

    h = mlp.register_forward_hook(_hook)
    handles.append(h)
    return handles

def _install_local_input_subspace_ablation_hook(
    model,
    layer_idx,
    Q,
    tok_ablate=0,
):
    handles = []
    blk = model.model.layers[layer_idx]

    def _prehook(_module, args, kwargs):
        x = _get_hidden_from_hook_args(args, kwargs)
        x2 = x.clone()

        v = x2[:, tok_ablate, :].to(dtype=torch.float32) # [B, D]
        Qd = Q.to(device=v.device, dtype=torch.float32) # [D, n_vec]

        coeff = v @ Qd # [B, n_vec]
        proj = coeff @ Qd.T # [B, D]
        v_new = v - proj
        x2[:, tok_ablate, :] = v_new.to(dtype=x.dtype)
        
        return _write_hidden_to_hook_args(x2, args, kwargs)

    h = blk.register_forward_pre_hook(_prehook, with_kwargs=True)
    handles.append(h)
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

# Geometric analysis

def _mean_cos_to_rows(X, W, row_idx_list):
    X = X.to(dtype=torch.float32)
    W = W.to(dtype=torch.float32, device=X.device)
    rows = torch.tensor([int(i) for i in row_idx_list], dtype=torch.long, device=X.device)
    Wsel = W.index_select(0, rows)

    Xn = F.normalize(X, dim=1)
    Wn = F.normalize(Wsel, dim=1)
    cos = Xn @ Wn.t()
    return cos.mean(dim=1)

def _pos_frac(G):
    return (G > 0).to(dtype=torch.float32).mean(dim=1) 

def _frac_energy_on_set_per_sample(Z, idx_list):
    Z = Z.to(dtype=torch.float32) # [B, D]
    idx = torch.tensor([int(i) for i in idx_list], dtype=torch.long, device=Z.device) 
    denom = (Z ** 2).sum(dim=1).clamp_min(EPS) # [B]
    num = (Z.index_select(1, idx) ** 2).sum(dim=1) # [B]
    return num / denom # [B]

def _frac_energy_on_dims_per_sample(Z, idx_list):
    Z = Z.to(dtype=torch.float32) # [B, D]
    idx = torch.tensor([int(i) for i in idx_list], dtype=torch.long, device=Z.device)
    denom = (Z ** 2).sum(dim=1, keepdim=True).clamp_min(EPS) # [B, 1]
    cols = Z.index_select(1, idx)
    num = (cols ** 2)
    return num / denom

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

def _fracmass_on_set(Z, idx_set):
    if Z is None or Z.numel() == 0 or (not idx_set):
        return 0.0
    Z = Z.to(dtype=torch.float32)
    denom = (Z ** 2).sum(dim=-1).clamp_min(EPS)
    idx = torch.tensor(sorted(idx_set), dtype=torch.long, device=Z.device)
    num = (Z[:, idx] ** 2).sum(dim=1)
    return float((num / denom).mean().item())

def _topk_idx_from_np(arr, k=3):
    order = np.argsort(-arr)
    return [int(i) for i in order[:int(k)].tolist()]

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

def _topk_energy_dims(
    Z,
    k=3,
    min_median_mass=0.55,
):
    assert Z is not None and Z.ndim == 2
    Z = Z.to(dtype=torch.float32)
    N, D = int(Z.size(0)), int(Z.size(1))
    kk = int(max(1, min(int(k), D)))

    e = Z.pow(2)
    denom = e.sum(dim=1, keepdim=True).clamp_min(EPS)
    frac = e / denom

    mean_frac = frac.mean(dim=0)
    idx = torch.topk(mean_frac, k=kk, largest=True).indices.detach().cpu().tolist()
    idx = [int(i) for i in idx]

    mass = frac[:, idx].sum(dim=1).detach().cpu().numpy().astype(np.float32)

    stats = {
        "k": kk,
        "mass_mean": float(np.mean(mass)),
        "mass_median": float(np.median(mass)),
        "mass_p10": float(np.quantile(mass, 0.10)),
        "mass_p90": float(np.quantile(mass, 0.90)),
    }

    if stats["mass_median"] < float(min_median_mass):
        return [], stats

    return idx, stats
    
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

def _compute_Wd_subspace(model, layer_idx, vec_idx):
    mlp = model.model.layers[layer_idx].mlp
    Wd = mlp.down_proj.weight.detach().to(dtype=torch.float32)
    B = Wd[:, vec_idx]
    Q, _R = torch.linalg.qr(B, mode="reduced")
    return Q.contiguous()

def _k_for_energy_frac(X, fracs=(0.25, 0.75, 0.99)):
    if X is None or X.numel() == 0:
        return [0 for _ in fracs]
    X = X.to(dtype=torch.float32)
    if X.ndim == 1:
        X = X.unsqueeze(0)

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
    if X is None or X.numel() == 0:
        return "[]"
    if X.ndim == 1:
        X = X.unsqueeze(0)
    ks = []
    for i in range(int(X.size(0))):
        ks.append(_k_for_energy_frac(X[i:i+1], fracs=fracs))
    k = torch.tensor(ks, dtype=torch.float32)
    mu = k.mean(dim=0)
    sd = k.std(dim=0, unbiased=False)
    parts = [_fmt_mu_sd(mu[j].item(), sd[j].item()) for j in range(int(mu.numel()))]
    return _print_list(parts)

def _mean_std_scalar_str(x):
    x = x.to(dtype=torch.float32)
    mu = float(x.mean().item())
    sd = float(x.std(unbiased=False).item())
    return _fmt_mu_sd(mu, sd)

def _mean_std_scalar_float_str(x, mu_digits=2, sd_digits=2):
    x = x.to(dtype=torch.float32)
    mu = float(x.mean().item())
    sd = float(x.std(unbiased=False).item())
    return _fmt_mu_sd_float(mu, sd, mu_digits=mu_digits, sd_digits=sd_digits)

def _fmt_mu_sd_float(mu, sd, mu_digits=2, sd_digits=2):
    mu_f = float(mu)
    sd_f = float(sd)
    return f"{mu_f:.{int(mu_digits)}f}±{sd_f:.{int(sd_digits)}f}"

def _fmt_mu_sd(mu, sd):
    mu_i = int(round(float(mu)))
    sd_f = float(sd)
    if sd_f < 0.05:
        sd_s = "0"
    elif sd_f < 100:
        sd_s = str(int(round(sd_f)))
    else:
        exp = int(math.floor(math.log10(sd_f)))
        mant = int(round(sd_f / 10 ** exp))
        if mant == 10:
            mant, exp = 1, exp + 1
        sd_s = f"{mant}e{exp}"
    return f"{mu_i}±{sd_s}"

def _reduce_headwise(xs):
    if not xs:
        return 0.0, 0.0
    X = torch.stack(xs, dim=0).to(dtype=torch.float32) # [N, H]
    head_mu = X.mean(dim=0) # [H]
    mu = float(head_mu.mean().item())
    var = float(head_mu.var(unbiased=False).item())
    return mu, var

def _reduce_headwise_vec(xh):
    xh = xh.to(dtype=torch.float32) # [H,]
    mu = float(xh.mean().item())
    var = float(xh.var(unbiased=False).item())
    return mu, var

def _headwise_cloud_radius_mu_var(vecs_h):
    X = torch.stack(vecs_h, dim=0) # [N, H, D]
    N, H, D = int(X.size(0)), int(X.size(1)), int(X.size(2))
    spreads = []
    for h in range(H):
        spreads.append(_cloud_radius(X[:, h, :]))
    spreads = torch.tensor(spreads, dtype=torch.float32) # [H]
    return _reduce_headwise_vec(spreads)

def _rank_in_row_abs(w, idx):
    absw = w.abs()
    order = torch.argsort(absw, descending=True)
    return int((order == int(idx)).nonzero(as_tuple=False).item()) + 1

def _rms(x, dim=-1, keepdim=False, eps=EPS):
    return (x.pow(2).mean(dim=dim, keepdim=keepdim) + eps).sqrt()

def _pos_count_stats(Z):
    assert Z.ndim == 2
    cnt = (Z > 0).sum(dim=1).to(dtype=torch.float32)
    mu = float(cnt.mean().item())
    sd = float(cnt.std(unbiased=False).item())
    D = int(Z.size(1))
    return mu, sd, D

def _pos_mu_sd_str(Z):
    mu, sd, D = _pos_count_stats(Z)
    return _fmt_mu_sd(mu, sd)

def _mean_vec_norm(X):
    if X is None or X.numel() == 0:
        return 0.0
    return float(X.norm(dim=-1).mean().item())

def _vec_norm_mu_sd_str(X):
    X = X.to(dtype=torch.float32)
    norms = X.norm(dim=1)
    mu = float(norms.mean().item())
    sd = float(norms.std(unbiased=False).item())
    return _fmt_mu_sd(mu, sd)

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

def _mean_activation_entropy(Z):
    Z = Z.to(dtype=torch.float32)
    assert Z.ndim == 2
    Z2 = (Z ** 2).clamp_min(0.0)
    denom = Z2.sum(dim=1, keepdim=True).clamp_min(EPS)
    p = Z2 / denom
    H = -(p * (p + EPS).log()).sum(dim=1)
    return float(H.mean().item())

def _topk_vals_mu_sd_str(Z, k=3, mode="abs"):
    Z = Z.to(dtype=torch.float32)
    assert Z.ndim == 2
    k = int(max(0, min(int(k), int(Z.size(1)))))

    if mode == "pos":
        score = torch.clamp(Z, min=0.0)
        vals, _ = torch.topk(score, k=k, dim=1, largest=True)
    elif mode == "abs":
        score = Z.abs()
        _, idx = torch.topk(score, k=k, dim=1, largest=True)
        vals = Z.gather(1, idx)
    else:
        raise ValueError

    mu = vals.mean(dim=0)
    sd = vals.std(dim=0, unbiased=False)
    parts = [_fmt_mu_sd(mu[j].item(), sd[j].item()) for j in range(int(mu.numel()))]
    return _print_list(parts)

def _mean_topk_column_norm(Wd, idx_set):
    if not idx_set:
        return 0.0
    idx = torch.tensor(sorted(idx_set), dtype=torch.long, device=Wd.device)
    return float(Wd[:, idx].norm(dim=0).mean().item())

def _row_norm_rank_desc(W):
    W = W.to(dtype=torch.float32)
    norms = W.norm(dim=1)
    order = torch.argsort(norms, descending=True)
    rank = torch.empty_like(order)
    rank[order] = torch.arange(1, order.numel() + 1, device=order.device)
    return norms, rank

def _vec_neg_topk(vec, k=3):
    v = vec.detach().to(dtype=torch.float32)
    neg_mask = v < 0
    if int(neg_mask.sum().item()) == 0:
        return [], []
    neg_idx_all = neg_mask.nonzero(as_tuple=False).view(-1)
    neg_vals_all = v[neg_idx_all]
    kk = min(int(k), int(neg_vals_all.numel()))
    _, order = torch.topk(neg_vals_all.abs(), k=kk, largest=True)
    neg_idx = neg_idx_all[order]
    neg_vals = neg_vals_all[order]
    return [int(i) for i in neg_idx.tolist()], [float(v) for v in neg_vals.tolist()]

def _select_topk_dims_from_rows(X, W_list, probe_idx, topk=2, topk_per_row=3):
    probe_idx = [int(i) for i in probe_idx]
    topk_per_row = int(topk_per_row)
    D = int(X.size(1))

    counts = np.zeros((D,), dtype=np.int64)
    energy_sum = np.zeros((D,), dtype=np.float64)

    for W in W_list:
        for ridx in probe_idx:
            w = W[int(ridx), :].to(dtype=torch.float32)
            e, _what = _dot_contrib_energy_per_dim(X, w, eps=EPS)
            top_idx = _topk_idx_from_np(e, k=topk_per_row)
            for j in top_idx:
                counts[int(j)] += 1
                energy_sum[int(j)] += float(e[int(j)])

    order = np.lexsort((-energy_sum, -counts))
    topk_idx = [int(order[i]) for i in range(int(topk))]
    return topk_idx, counts, energy_sum

def _dot_contrib_energy_per_dim(X, w_row, eps=EPS):
    X = X.to(dtype=torch.float32)
    w = w_row.to(dtype=torch.float32, device=X.device)

    Xhat = F.normalize(X, dim=1)
    what = F.normalize(w, dim=0)

    C = Xhat * what.view(1, -1) # [N, D]
    energy = C.pow(2).mean(dim=0) # [D]
    energy = energy.detach().cpu().numpy()
    what = what.detach().cpu().numpy()

    return energy, what
    
def _k_at_cdf_threshold(mass_sorted, thr):
    cdf = np.cumsum(mass_sorted)
    thr = float(thr)
    thr = max(0.0, min(1.0, thr))
    k = int(np.searchsorted(cdf, thr, side="left")) + 1
    return int(min(k, int(mass_sorted.size)))

def _majority_sign_over_threshold(X, dim_idx, thr=0.75):
    col = X[:, int(dim_idx)]
    frac_pos = float((col > 0).to(dtype=torch.float32).mean().item())
    frac_neg = float((col < 0).to(dtype=torch.float32).mean().item())
    thr = float(thr)
    if frac_pos >= thr:
        return "+"
    if frac_neg >= thr:
        return "-"
    return "∅"

def _squash_rank_axis(ranks, D, thr, tail_frac=0.10):
    D = int(max(1, D))
    r = ranks.astype(np.float32)
    thr = int(max(1, min(int(thr), D - 1)))
    tail_frac = float(max(1e-6, min(float(tail_frac), 0.99)))

    head_w = (1.0 - tail_frac) * float(D)
    tail_w = tail_frac * float(D)

    x = np.empty_like(r, dtype=np.float32)
    head = (r <= thr)
    x[head] = (r[head] / float(thr)) * head_w

    tail = ~head
    denom = float(max(1, (D - 1 - thr)))
    x[tail] = head_w + ((r[tail] - float(thr)) / denom) * tail_w
    return x

# hooks

def _collect_residual_decomp_to_layer(model, layer_idx, tok_idxs, tokenized):
    L = int(layer_idx)
    tok_idxs = [int(t) for t in tok_idxs]
    run_cache = {}

    comps = []
    comps.append(("embed", "embed"))
    for i in range(0, L+1):
        comps.append((f"attn_{i}", f"L{i}.attn"))
        if i <= L - 1:
            comps.append((f"mlp_{i}", f"L{i}.mlp"))
    
    buckets = {key: {t: [] for t in tok_idxs} for key, _lbl in comps}
    comp_labels = {key: lbl for key, lbl in comps}

    handles = []

    # get embed states
    h_emb = model.model.layers[0].register_forward_pre_hook(
        lambda m, a, k, _c=run_cache: _c.__setitem__("embed", _get_hidden_from_hook_args(a, k).detach().float().cpu()),
        with_kwargs=True,
    )
    handles.append(h_emb)

    # get attn out and mlp out
    for i in range(0, L+1):
        attn = model.model.layers[i].self_attn
        mlp = model.model.layers[i].mlp

        def _mk_attn_hook(ii):
            def _hook(_module, args, output, _c=run_cache):
                _c[f"attn_{ii}"] = output[0].detach().float().cpu()
            return _hook

        def _mk_mlp_hook(ii):
            def _hook(_module, args, output, _c=run_cache):
                _c[f"mlp_{ii}"] = output.detach().float().cpu()
            return _hook

        handles.append(attn.register_forward_hook(_mk_attn_hook(i)))
        if i <= L - 1:
            handles.append(mlp.register_forward_hook(_mk_mlp_hook(i)))

    try:
        for input_ids, base_pos in tqdm(
            tokenized,
            desc=f"Collecting residual decomp to Xpre at L={L}",
            leave=False,
        ):
            run_cache.clear()
            _ = prefill(model, input_ids, base_pos)

            for key, _lbl in comps:
                if key not in run_cache:
                    raise RuntimeError(f"[residual] Missing cache key: {key}")

            for key, _lbl in comps:
                X = run_cache[key][0]
                S = int(X.size(0))
                for t in tok_idxs:
                    if t >= S:
                        continue
                    buckets[key][t].append(X[t].clone())

    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass

    out = {
        "labels": comp_labels,
        "data": {},
    }
    for key, _lbl in comps:
        out["data"][key] = {}
        for t in tok_idxs:
            vecs = buckets[key][t]
            if vecs:
                out["data"][key][t] = torch.stack(vecs, dim=0) # [N, D]
    return out

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

def _collect_preln_gradients_for_Qproj(
    model,
    layer_idx,
    tok_idx,
    tokenized,
    Q_basis,
):
    assert Q_basis is not None and Q_basis.ndim == 2

    blk = model.model.layers[layer_idx]
    ln = blk.post_attention_layernorm

    grads = []
    cache = {}

    def _ln_prehook(_module, args):
        s = args[0]
        s2 = s.detach().clone().requires_grad_(True)
        cache["s"] = s2
        return (s2,)

    def _ln_fwdhook(_module, args, output):
        cache["x_post"] = output

    h_pre = ln.register_forward_pre_hook(_ln_prehook)
    h_fwd = ln.register_forward_hook(_ln_fwdhook)

    try:
        with _no_param_grads(model):
            for input_ids, base_pos in tqdm(
                tokenized,
                desc=f"[decompose_preln_Q_proj] grads @ L={layer_idx} tok={tok_idx}",
                leave=False,
            ):
                cache.clear()
                model.zero_grad(set_to_none=True)

                _ = model(
                    input_ids=input_ids.to(next(model.parameters()).device),
                    position_ids=base_pos.to(next(model.parameters()).device),
                    use_cache=False,
                    output_attentions=False,
                    output_hidden_states=False,
                )

                s = cache.get("s", None)
                x_post = cache.get("x_post", None)
                if s is None or x_post is None:
                    continue

                x0 = x_post[:, tok_idx, :].to(dtype=torch.float32)
                Qd = Q_basis.to(device=x0.device, dtype=torch.float32)
                coeff = x0 @ Qd
                f = (coeff.pow(2).sum(dim=1)).mean()
                f.backward()

                g = s.grad[:, tok_idx, :].detach().float().cpu()
                grads.append(g[0].clone())

    finally:
        h_pre.remove()
        h_fwd.remove()

    if not grads:
        return None
    
    return torch.stack(grads, dim=0)

def _collect_mlp_internals_for_layer(model, layer_idx, tok_idxs, tokenized):
    mlp = model.model.layers[layer_idx].mlp
    attn = model.model.layers[layer_idx].self_attn
    ln = model.model.layers[layer_idx].post_attention_layernorm
    cache = {}

    def _ln_prehook(_module, args, _cache=cache):
        _cache["ln_in"] = args[0].detach().float().cpu()
    h_ln = ln.register_forward_pre_hook(_ln_prehook)

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
    LN_in_tok = copy.deepcopy(X_tok)

    try:
        for input_ids, base_pos in tqdm(
            tokenized,
            desc=f"Collecting MLP internals at L={layer_idx}",
            leave=False,
        ):
            cache.clear()
            _ = prefill(model, input_ids, base_pos)
            assert all(k in cache for k in ["x", "g", "u", "a", "z", "y", "ln_in"])

            x = cache["x"][0]
            g = cache["g"][0]
            u = cache["u"][0]
            a = cache["a"][0]
            z = cache["z"][0]
            y = cache["y"][0]
            ln_in = cache["ln_in"][0]
            S = int(x.shape[0])
            for t in tok_idxs:
                assert t < S
                X_tok[t].append(x[t].clone())
                G_tok[t].append(g[t].clone())
                U_tok[t].append(u[t].clone())
                A_tok[t].append(a[t].clone())
                Z_tok[t].append(z[t].clone())
                Y_tok[t].append(y[t].clone())
                LN_in_tok[t].append(ln_in[t].clone())

    finally:
        for _h in [h, h_ln]:
            try:
                _h.remove()
            except Exception:
                pass

    out = {"X": {}, "G": {}, "U": {}, "A": {}, "Z": {}, "Y": {}, "LN_in": {}}
    for name, bucket in [
        ("X", X_tok),
        ("G", G_tok),
        ("U", U_tok),
        ("A", A_tok),
        ("Z", Z_tok),
        ("Y", Y_tok),
        ("LN_in", LN_in_tok),
    ]:
        for t, vecs in bucket.items():
            if vecs:
                out[name][t] = torch.stack(vecs, dim=0)

    return out

def _make_preln_component_ablation_prehook(
    Xc_by_key,
    tok_ablate,
):
    ctr = {"i": 0}
    keys = list(Xc_by_key.keys())

    def _prehook(_module, args, _ctr=ctr, _keys=keys, _Xc_by_key=Xc_by_key):
        x = args[0]
        i = int(_ctr["i"])
        _ctr["i"] = i + 1

        any_X = _Xc_by_key[_keys[0]]
        if i < 0 or i >= int(any_X.size(0)):
            return (x,)

        x2 = x.clone()
        v_sum = None
        for k in _keys:
            v = _Xc_by_key[k][i]
            v_sum = v if (v_sum is None) else (v_sum + v)

        v_sum = v_sum.to(device=x2.device, dtype=x2.dtype)
        x2[:, tok_ablate, :] = x2[:, tok_ablate, :] - v_sum.view(1, -1)
        return (x2,)

    return _prehook, ctr

def _collect_mlp_in_gradients_for_single_zdim(
    model,
    layer_idx,
    tok_idx,
    tokenized,
    z_dim,
):
    mlp = model.model.layers[layer_idx].mlp
    grads = []
    cache = {}
    z_dim = int(z_dim)

    def _mlp_prehook(_module, args):
        x = args[0]
        x2 = x.detach().clone().requires_grad_(True)
        cache["x"] = x2
        return (x2,)

    h_pre = mlp.register_forward_pre_hook(_mlp_prehook)

    try:
        with _no_param_grads(model):
            for input_ids, base_pos in tqdm(
                tokenized,
                desc=f"[decompose_mlp_in] grads @ L={layer_idx} tok={tok_idx} z_dim={z_dim}",
                leave=False,
            ):
                cache.clear()
                model.zero_grad(set_to_none=True)

                _ = model(
                    input_ids=input_ids.to(next(model.parameters()).device),
                    position_ids=base_pos.to(next(model.parameters()).device),
                    use_cache=False,
                    output_attentions=False,
                    output_hidden_states=False,
                )

                x = cache.get("x", None)
                if x is None:
                    continue

                g = mlp.gate_proj(x)
                u = mlp.up_proj(x)
                a = mlp.act_fn(g)
                z = a * u

                z0 = z[:, tok_idx, z_dim].to(dtype=torch.float32)
                f = (z0 ** 2).sum()
                f.backward()

                gx = x.grad[:, tok_idx, :].detach().float().cpu()
                grads.append(gx[0].clone())

    finally:
        try:
            h_pre.remove()
        except Exception:
            pass

    if not grads:
        return None

    return torch.stack(grads, dim=0)

def _scan_sink_and_collect_mlp_scalars(
    model,
    tokenized,
    scan_layers,
    layer_idx,
    tok_idx,
    Wu,
    wu_rows,
    z_idx,
):
    # attention metrics
    per_layer_acc = {L: {"sink_attn_h": []} for L in scan_layers}
    
    mlp = model.model.layers[layer_idx].mlp
    ln = model.model.layers[layer_idx].post_attention_layernorm
    
    scal = {
        "wu_dot": [],
        "z_frac": [],
        "out_log_norm": [],
        "g_norm": [],
        "u_norm": [],
        "z_norm": [],
        "g_frac3": [],
        "u_frac3": [],
        "z_frac3": [],
        "wu_dot_pre": [],
    }

    attn_out_norm = []
    preln_queue = []

    def _ln_capture_prehook(_module, args):
        x_pre = args[0]
        preln_queue.append(x_pre[:, tok_idx, :].detach())

    def _mlp_hook(_module, inputs, output):
        x = inputs[0]
        y = output
        x0 = x[:, tok_idx, :].detach()
        y0 = y[:, tok_idx, :].detach()

        with torch.no_grad():
            g = _module.gate_proj(x).detach()
            u = _module.up_proj(x).detach()
            a = _module.act_fn(g).detach()
            z = (a * u).detach()

            g0 = g[:, tok_idx, :]
            u0 = u[:, tok_idx, :]
            z0 = z[:, tok_idx, :]

            wu_dot = _mean_cos_to_rows(x0.to(dtype=torch.float32), Wu, wu_rows)
            zfrac = _frac_energy_on_set_per_sample(z0, z_idx)
            outln = torch.log(y0.norm(dim=1).clamp_min(EPS))

            scal["wu_dot"].append(wu_dot.detach().float().cpu())
            scal["z_frac"].append(zfrac.detach().float().cpu())
            scal["out_log_norm"].append(outln.detach().float().cpu())

            scal["g_norm"].append(g0.norm(dim=1).detach().float().cpu())
            scal["u_norm"].append(u0.norm(dim=1).detach().float().cpu())
            scal["z_norm"].append(z0.norm(dim=1).detach().float().cpu())

            scal["g_frac3"].append(_frac_energy_on_dims_per_sample(g0, z_idx).detach().float().cpu())
            scal["u_frac3"].append(_frac_energy_on_dims_per_sample(u0, z_idx).detach().float().cpu())
            scal["z_frac3"].append(_frac_energy_on_dims_per_sample(z0, z_idx).detach().float().cpu())

            if preln_queue:
                xpre0 = preln_queue.pop(0)
                wu_dot_pre = _mean_cos_to_rows(xpre0.to(dtype=torch.float32), Wu, wu_rows)
                scal["wu_dot_pre"].append(wu_dot_pre.detach().float().cpu()) 

    attn = model.model.layers[layer_idx].self_attn
    def _attn_hook(_module, args, output):
        aout = output[0]
        v = aout[:, tok_idx, :].detach().to(dtype=torch.float32)
        attn_out_norm.append(v.norm(dim=1).detach().cpu())

    h_ln = ln.register_forward_pre_hook(_ln_capture_prehook)
    h_attn = attn.register_forward_hook(_attn_hook)
    h = mlp.register_forward_hook(_mlp_hook)

    try:
        preln_queue.clear()
        for input_ids, base_pos in tqdm(
            tokenized,
            desc=f"Collecting MLP metrics and sink attention at L={layer_idx} | token={tok_idx}",
            leave=False,
        ):
            attns = prefill(model, input_ids, base_pos)

            for L in scan_layers:
                hm_all = _pick_all_heads_with_caches(model, attns, L)
                sink_qmean_per_head = hm_all[:, 4:, 0].mean(dim=1)
                per_layer_acc[L]["sink_attn_h"].append(sink_qmean_per_head.detach().cpu())

    finally:
        h.remove()
        h_attn.remove()
        h_ln.remove()

    attn_stats = []
    for L in sorted(per_layer_acc.keys()):
        acc = per_layer_acc[L]
        sink_mu, sink_var = _reduce_headwise(acc["sink_attn_h"])
        attn_stats.append({"sink_attn": sink_mu, "sink_attn_var": sink_var, "layer": L})

    mlp_scal_np = {}
    for k, chunks in scal.items():
        v = torch.cat(chunks, dim=0).numpy().astype(np.float32) # some are [N,] and some are [N, K]
        mlp_scal_np[k] = v

    attn_out_norm_mu = float(torch.cat(attn_out_norm, dim=0).mean().item())

    return attn_stats, mlp_scal_np, attn_out_norm_mu

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

def _parse_component_spec(spec):
    if spec is None:
        return None, None
    s = str(spec).strip()
    if not s:
        return None, None
    s_low = s.lower()

    if s_low == "embed":
        return "embed", "embed"

    if "." in s_low:
        a, b = s_low.split(".", 1)
        if a in ("attn", "mlp") and b.isdigit():
            idx = int(b)
            key = f"{a}_{idx}"
            return key, f"{a}.{idx}"
    
    raise ValueError(f"Invalid component spec: {spec}")

def _parse_ablate_layers_spec(spec_list, valid_comp_keys):
    if not spec_list:
        return [], []

    keys = []
    pretty = []
    seen = set()
    valid = set(valid_comp_keys)

    for s in spec_list:
        k, p = _parse_component_spec(s)
        if k is None:
            continue
        if k not in valid:
            raise ValueError(f"'{s}' -> '{k}' not in valid comp keys: {valid}")

        if k in seen:
            continue

        seen.add(k)
        keys.append(k)
        pretty.append(p)
        
    return keys, pretty

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

def _scan_sink_attention_only(model, tokenized, scan_layers):
    per_layer_acc = {
        L: {"sink_attn_h": []}
        for L in scan_layers
    }

    min_len = min(ids.shape[1] for (ids, _bp) in tokenized)
    last_q = min_len - 1

    for input_ids, base_pos in tqdm(tokenized, desc="Scanning (attn only)", position=0, leave=False):
        attns = prefill(model, input_ids, base_pos)

        for L in scan_layers:
            hm_all = _pick_all_heads_with_caches(model, attns, L)
            sink_qmean_per_head = hm_all[:, 4:, 0].mean(dim=1)
            per_layer_acc[L]["sink_attn_h"].append(sink_qmean_per_head.detach().cpu())
            
    stats = []
    for L in sorted(per_layer_acc.keys()):
        acc = per_layer_acc[L]
        sink_mu, sink_var = _reduce_headwise(acc["sink_attn_h"])
        stats.append({
            "sink_attn": sink_mu,
            "sink_attn_var": sink_var,
            "layer": L,
        })

    return stats

@contextmanager
def _no_param_grads(model):
    req = []
    for p in model.parameters():
        req.append(p.requires_grad)
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p, r in zip(model.parameters(), req):
            p.requires_grad_(r)

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

def _pick_all_heads_with_caches(model, attns, layer_idx):
    src = model._lower_attn_cache \
        if (hasattr(model, "_lower_attn_cache") and (layer_idx in model._lower_attn_cache)) \
        else attns
    a = src[layer_idx] # [B, H, Q, K]
    return a[0].detach().float().cpu()

# plots

def _plot_projection_histogram_before_after_stacked(
    p_before_list,
    p_after_list,
    labels,
    out_path,
    title,
    bins=60,
):
    assert len(p_before_list) == len(p_after_list) == len(labels)
    k = int(len(labels))

    fig, axes = plt.subplots(k, 1, figsize=(9.5, max(3.2, 2.6 * k)), sharex=False)
    if k == 1:
        axes = [axes]

    for i, (pbef, paft, lab) in enumerate(zip(p_before_list, p_after_list, labels)):
        ax = axes[i]
        pbef = np.asarray(pbef, dtype=np.float32)
        paft = np.asarray(paft, dtype=np.float32)

        lo = float(min(pbef.min(), paft.min()))
        hi = float(max(pbef.max(), paft.max()))
        edges = np.linspace(lo, hi, int(bins))

        ax.hist(pbef, bins=edges, density=True, alpha=0.55, label="before")
        ax.hist(paft, bins=edges, density=True, alpha=0.55, label="after")
        ax.axvline(0.0, color="black", linewidth=0.8)

        ax.set_ylabel("density")
        ax.set_title(str(lab), loc="left", fontsize=10)
        ax.grid(True, which="major", axis="both", linewidth=0.4, alpha=0.25)
        if i == 0:
            ax.legend(loc="best", frameon=False, fontsize=9)

    axes[-1].set_xlabel("projection value at coordinate")
    fig.suptitle(title, fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0.0, 1, 0.985])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

def _plot_projection_histogram_stacked(
    p0_list, 
    pb_list,
    labels, 
    out_path, 
    title, 
    bins=60
):
    assert len(p0_list) == len(pb_list) == len(labels)
    k = int(len(labels))

    fig, axes = plt.subplots(k, 1, figsize=(9.5, max(3.2, 2.6 * k)), sharex=False)
    if k == 1:
        axes = [axes]

    for i, (p0, pb, lab) in enumerate(zip(p0_list, pb_list, labels)):
        ax = axes[i]
        p0 = np.asarray(p0, dtype=np.float32)
        pb = np.asarray(pb, dtype=np.float32)

        lo = float(min(p0.min(), pb.min()))
        hi = float(max(p0.max(), pb.max()))
        edges = np.linspace(lo, hi, int(bins))

        ax.hist(p0, bins=edges, density=True, alpha=0.55, label="tok0")
        ax.hist(pb, bins=edges, density=True, alpha=0.55, label="not-tok0")
        ax.axvline(0.0, color="black", linewidth=0.8)

        ax.set_ylabel("density")
        ax.set_title(str(lab), loc="left", fontsize=10)
        ax.grid(True, which="major", axis="both", linewidth=0.4, alpha=0.25)
        if i == 0:
            ax.legend(loc="best", frameon=False, fontsize=9)

    axes[-1].set_xlabel(r"$v_1^\top(x-\mu_b)$")
    
    fig.suptitle(title, fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0.0, 1, 0.985])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

def _plot_grad_second_moment_spectrum_stacked(
    lambdas_list, 
    labels,
    out_path, 
    title, 
    max_k=256
):
    assert len(lambdas_list) == len(labels)
    k = int(len(labels))

    fig, axes = plt.subplots(k, 1, figsize=(10.5, max(4.0, 2.8 * k)), sharex=False)
    if k == 1:
        axes = [axes]

    for i, (lam, lab) in enumerate(zip(lambdas_list, labels)):
        ax = axes[i]
        lam = np.asarray(lam, dtype=np.float64)
        lam = lam[np.isfinite(lam)]
        lam = lam[lam > 0]

        K = int(min(int(max_k), int(lam.size)))
        xs = np.arange(1, K+1, dtype=np.int32)
        y = lam[:K]
        cdf = np.cumsum(y) / max(1e-12, float(np.sum(lam)))

        ax.plot(xs, y, marker="o", linewidth=1.0, markersize=3)
        ax.set_yscale("log")
        ax.set_ylabel(r"$\lambda_i$  (log)")
        ax.grid(True, which="both", linewidth=0.4, alpha=0.35)

        axr = ax.twinx()
        axr.plot(xs, cdf, marker="o", linewidth=1.0, markersize=3)
        axr.set_ylim(0.0, 1.02)
        axr.set_ylabel(r"CDF")
        axr.grid(False)

        ax.set_title(str(lab), loc="left", fontsize=10)

        ax.set_xlabel("eigen index (k)")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=10))

    fig.suptitle(title, fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0.0, 1, 0.985])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

def _plot_residual_dim_heatmaps(dim_list, comp_names, tok0_mat, diff_mat, out_path, title):
    assert len(dim_list) == tok0_mat.shape[0] == diff_mat.shape[0]
    C = int(len(comp_names))
    R = int(len(dim_list))

    fig, axes = plt.subplots(R, 1, figsize=(max(10, 0.55 * C), 1.6 * R), sharex=True)
    if R == 1:
        axes = [axes]

    def _row_norm(v):
        v = v.astype(np.float32)
        vmax = float(np.max(v))
        vmin = float(np.min(v))
        return vmin, vmax

    for di, d in enumerate(dim_list):
        ax = axes[di]

        row_tok0 = tok0_mat[di:di+1, :]

        vmin1, vmax1 = _row_norm(row_tok0)
        _ = ax.imshow(
            row_tok0,
            aspect="auto",
            interpolation="nearest",
            vmin=vmin1,
            vmax=vmax1,
            extent=(-0.5, C - 0.5, -0.5, 0.5),
        )

        diff_vec = diff_mat[di, :].astype(np.float32)
        max_abs = float(np.max(np.abs(diff_vec)))
        if max_abs < EPS:
            max_abs = 1e-6

        y_base = 0.70
        y_span = 0.45
        xs = np.arange(C, dtype=np.float32)
        heights = (diff_vec / max_abs) * y_span
        
        ax.bar(
            xs,
            heights,
            width=0.35,
            bottom=y_base,
            alpha=0.9,
            edgecolor="none",
            clip_on=False,
        )

        ax.set_yticks([0])
        ax.set_yticklabels(["tok0"], fontsize=9)

        ax.set_title(
            f"dim={int(d)} | heat=tok0 mean | bars=(tok0-tok8) mean",
            fontsize=10,
            loc="left",
        )
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(left=False)

        ax.set_xlim(-0.5, C - 0.5)
        ax.set_ylim(-0.5, 1.05)

    axes[-1].set_xticks(list(range(C)))
    axes[-1].set_xticklabels(comp_names, rotation=65, ha="right", fontsize=8)

    fig.suptitle(title, fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0.0, 1, 0.985])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

def _plot_per_gate_mean_contrib(Srms, gamma_base, gamma_alt, Wg, out_path, title):
    Srms = Srms.to(dtype=torch.float32)
    gb = gamma_base.to(dtype=torch.float32).view(1, -1)
    ga = gamma_alt.to(dtype=torch.float32).view(1, -1)

    Xb = Srms * gb
    Xa = Srms * ga

    Gb = Xb @ Wg.t()
    Ga = Xa @ Wg.t()

    mu_b = Gb.mean(dim=0).detach().cpu().numpy() # [D]
    sd_b = Gb.std(dim=0, unbiased=False).detach().cpu().numpy()
    mu_a = Ga.mean(dim=0).detach().cpu().numpy()
    sd_a = Ga.std(dim=0, unbiased=False).detach().cpu().numpy()

    idx = np.arange(mu_b.shape[0])
    # idx = np.argsort(mu_b)

    x = np.arange(len(idx))
    plt.figure(figsize=(12, 3.8))
    plt.errorbar(
        x, mu_b[idx], yerr=sd_b[idx],
        fmt=".", elinewidth=0.5, capsize=1, label="baseline (gamma)"
    )
    plt.errorbar(
        x, mu_a[idx], yerr=sd_a[idx],
        fmt=".", elinewidth=0.5, capsize=1, label="no_gamma_top1+2"
    )
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xlabel("gate neurons")
    plt.ylabel("mean contribution ± std")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def _plot_z_topk_energy_stacks(
    Z_list,
    tok_labels,
    topk,
    out_path,
    title,
):
    norms_mu = []
    parts = []
    parts_idx = []

    for Z in Z_list:
        Z = Z.to(dtype=torch.float32)
        nm = float(Z.norm(dim=1).mean().item())
        norms_mu.append(nm)

        e = (Z.pow(2))
        frac = e / e.sum(dim=1, keepdim=True).clamp_min(EPS)
        mass = frac.mean(dim=0).detach().cpu().numpy().astype(np.float32)

        order = np.argsort(-mass)
        k_eff = int(min(topk, order.shape[0]))
        idx = [int(i) for i in order[:k_eff].tolist()]
        top_mass = [float(mass[i]) for i in idx]
        rest = float(max(0.0, 1.0 - sum(top_mass)))

        parts_idx.append(idx)
        parts.append(top_mass + [rest])

    y = np.arange(len(tok_labels), dtype=np.int32)

    fig, axes = plt.subplots(
        2, 1, figsize=(13.0, 6.4), sharey=True,
        gridspec_kw={"height_ratios": [2.0, 1.6]}
    )
    ax_full, ax_zoom = axes

    cmap = plt.get_cmap("viridis")
    top_colors = [cmap(t) for t in np.linspace(0.15, 0.85, topk)]
    rest_face = (0.92, 0.92, 0.92, 1.0)
    rest_edge = (0.45, 0.45, 0.45, 1.0)

    def _draw(ax, xlim=None, show_labels=True):
        for i, (nm, fracs, idxs) in enumerate(zip(norms_mu, parts, parts_idx)):
            left = 0.0

            for j, f in enumerate(fracs[:-1]):
                w = float(nm) * float(f)
                ax.barh(i, w, left=left, height=0.62, color=top_colors[j], alpha=0.92, edgecolor="white", linewidth=0.6)

                if show_labels and (j < 6) and (w > 0.06 * max(1.0, nm)):
                    ax.text(left + 0.5 * w, i, str(int(idxs[j])), ha="center", va="center", fontsize=8, color="black")

                left += w
        
            w_rest = float(nm) * float(fracs[-1])
            ax.barh(i, w_rest, left=left, height=0.62, color=rest_face, alpha=1.0, edgecolor=rest_edge, linewidth=0.9, hatch="///")
            left += w_rest

            ax.text(left + 0.01 * max(1.0, left), i, f"||Z||≈{nm:.1f}", va="center", ha="left", fontsize=9, color="black")

        ax.set_yticks(y)
        ax.set_yticklabels(tok_labels, fontsize=10)
        ax.set_ylabel("token", fontsize=10)
        ax.grid(True, axis="x", linewidth=0.4, alpha=0.18)
        ax.grid(False, axis="y")
        ax.tick_params(axis="y", pad=6)

        if xlim is not None:
            ax.set_xlim(0.0, float(xlim))

    ax_full.set_title(title, fontsize=11, loc="center", pad=10)
    ax_full.set_xlabel("mean ||Z|| (stacked by top-k energy dims + rest)", labelpad=10)
    _draw(ax_full, xlim=None, show_labels=True)

    zoom_ref = max(norms_mu[1:])
    ax_zoom.set_xlabel("mean ||Z|| (zoom to tok1/tok8 scale)", labelpad=10)
    _draw(ax_zoom, xlim=zoom_ref * 1.15, show_labels=True)

    fig.subplots_adjust(left=0.12, right=0.98, top=0.90, bottom=0.10, hspace=0.30)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

def _plot_wg_row_energy_stack(X, Wg, row_idxs, out_path, title, squash_boundary=16, topk=3):
    fig, axes = plt.subplots(len(row_idxs), 1, figsize=(12, 7.5), sharex=False, squeeze=False)
    axes = axes[:, 0]

    for ax, ridx in zip(axes, row_idxs):
        w = Wg[int(ridx), :].to(dtype=torch.float32)
        energy, what = _dot_contrib_energy_per_dim(X, w, eps=EPS)

        order = np.argsort(-energy)
        e_sorted = energy[order]
        what_sorted = what[order]

        D = int(e_sorted.shape[0])
        ranks = np.arange(D)

        denom = float(np.sum(e_sorted))
        mass = (e_sorted / denom).astype(np.float32)

        k25 = _k_at_cdf_threshold(mass, 0.25)
        k50 = _k_at_cdf_threshold(mass, 0.50)
        k75 = _k_at_cdf_threshold(mass, 0.75)
        
        thr = int(max(1, min(int(squash_boundary), D-1)))
        x = _squash_rank_axis(ranks, D=D, thr=thr, tail_frac=0.10)

        b25 = int(min(max(0, k25), D-1))
        b50 = int(min(max(0, k50), D-1))
        b75 = int(min(max(0, k75), D-1))

        cur = 0
        if b25 > 0:
            ax.axvspan(x[0], x[b25], alpha=0.10, color="tab:green", label="top .25 mass")
            cur = b25

        if b50 > 0:
            if b50 == cur and cur > 0:
                ax.axvspan(x[0], x[cur], alpha=0.10, color="tab:orange", label="top .50 mass")
            elif b50 > cur:
                ax.axvspan(x[cur], x[b50], alpha=0.10, color="tab:orange", label="top .50 mass")
                cur = b50

        if b75 > 0:
            if b75 == cur and cur > 0:
                ax.axvspan(x[0], x[cur], alpha=0.10, color="tab:red", label="top .75 mass")
            elif b75 > cur:
                ax.axvspan(x[cur], x[b75], alpha=0.10, color="tab:red", label="top .75 mass")
                cur = b75

        kk = int(min(int(topk), int(e_sorted.shape[0])))
        top_idx = order[:kk]

        parts = []
        for i in top_idx.tolist():
            wsgn = "+" if what[i] >= 0 else "-"
            xsgn = _majority_sign_over_threshold(X, int(i), thr=0.75)
            parts.append(f"{int(i)}({wsgn}/{xsgn})")

        top_str = " | top{} idx: {}".format(
            kk,
            " > ".join(parts)
        )

        ax.plot(x, e_sorted, linewidth=1.0, label=f"row={int(ridx)}{top_str}")

        ax.set_ylabel("E[c^2]")
        # ax.set_yscale("log")
        top_energy = float(e_sorted[0])
        ax.set_yscale("symlog", linthresh=top_energy * 1e-2, linscale=1.0)
        ax.set_xlim(0.0, float(D))

        tick_ranks = [0, 1, 2, 4, 8, thr, b25, b50, b75, D-1]
        tick_ranks = sorted(set(int(r) for r in tick_ranks if 0 <= int(r) < D))
        tick_pos = _squash_rank_axis(np.array(tick_ranks), D=D, thr=thr, tail_frac=0.10)
        ax.set_xticks(tick_pos.tolist())
        ax.set_xticklabels([str(int(r)) for r in tick_ranks], fontsize=9)

        handles, labels = ax.get_legend_handles_labels()
        seen = set()
        h2, l2 = [], []
        for h, l in zip(handles, labels):
            if l in seen:
                continue
            seen.add(l)
            h2.append(h)
            l2.append(l)
        ax.legend(h2, l2, loc="upper right", fontsize=8, frameon=False)
        ax.grid(True, linewidth=0.3, alpha=0.4)

    axes[-1].set_xlabel(f"dim rank by contrib energy (desc)")

    fig.suptitle(title, fontsize=12, y=0.995)
    fig.tight_layout(rect=[0,0.0,1,0.985])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

def _plot_ablation_delta_bars(
    comp_names,
    deltas_mu_sd,
    out_path,
    title,
    out_log_norm_ref=None,
    norm_stack=None,
    stack_idx=None,
):
    metrics = [
        # ("wg_dot", "Δz cos(X_postLN, Wg_rows)"),
        ("wu_dot", "Δz cos(X_postLN, Wu_rows)"),
        ("g_stack", "||G|| with energy split (3 dims + Rest)"),
        ("u_stack", "||U|| with energy split (3 dims + Rest)"),
        # ("z_frac", "Relative Δ (Z frac @ top3)"),
        ("z_stack", "||Z|| with energy split (3 dims + Rest)"),
        ("out_log_norm", "×  (baseline=1)")
    ]
    K = len(comp_names)

    fig, axes = plt.subplots(
        len(metrics),
        1,
        figsize=(13.5, max(12, 1.30 * K)),
        sharey=True,
    )
    if len(metrics) == 1:
        axes = [axes]

    y = np.arange(K, dtype=np.int32)

    def _draw_stack(ax, stack_key, xlabel):
        assert norm_stack is not None and stack_idx is not None
        idx3 = [int(i) for i in stack_idx]
        parts = np.asarray(norm_stack[stack_key], dtype=np.float32)
        left = np.zeros((K,), dtype=np.float32)

        cmap = plt.get_cmap("viridis")
        cols = [cmap(t) for t in np.linspace(0.20, 0.80, 3)]
        rest_face = (0.92, 0.92, 0.92, 1.0)
        rest_edge = (0.45, 0.45, 0.45, 1.0)

        for j in range(3):
            w = parts[:, j]
            ax.barh(y, w, left=left, height=0.68, alpha=0.92, edgecolor="white", linewidth=0.6, color=cols[j])
            for i in range(K):
                if w[i] > 0.10 * max(1e-6, float(parts[i].sum())):
                    ax.text(left[i] + 0.5 * w[i], y[i], str(idx3[j]), ha="center", va="center", fontsize=8, color="black")
            left += w

        w = parts[:, 3]
        ax.barh(y, w, left=left, height=0.68, alpha=1.0, edgecolor=rest_edge, linewidth=0.9, color=rest_face, hatch="//")

        ax.set_xlabel(xlabel)
        ax.grid(True, axis="x", linewidth=0.4, alpha=0.35)
        ax.grid(False, axis="y")
        ax.set_yticks(y)
        ax.set_yticklabels(comp_names, fontsize=8)
        ax.set_ylabel("Residual term", fontsize=9)

    for i, (mkey, ylab) in enumerate(metrics):
        ax = axes[i]

        if mkey in ("g_stack", "u_stack", "z_stack"):
            if mkey == "g_stack":
                _draw_stack(ax, "g", ylab)
            elif mkey == "u_stack":
                _draw_stack(ax, "u", ylab)
            elif mkey == "z_stack":
                _draw_stack(ax, "z", ylab)
            continue

        mu, sd = deltas_mu_sd[mkey]
        mu = np.asarray(mu, dtype=np.float32)
        sd = np.asarray(sd, dtype=np.float32)

        ax.barh(y, mu, height=0.68, alpha=0.90, edgecolor="none")

        ax.errorbar(
            mu,
            y,
            xerr=sd,
            fmt="none",
            ecolor="black",
            elinewidth=0.8,
            capsize=2,
            alpha=0.55,
        )

        if mkey == "out_log_norm":
            ax.axvline(1.0, color="black", linewidth=0.9, linestyle="--")
            if out_log_norm_ref is not None:
                ref = float(out_log_norm_ref)
                ax.axvline(ref, color=(0.35, 0.35, 0.35), linewidth=1.2, linestyle=":")
                xmax = float(np.nanmax(mu + sd))
                if ref < 0.05 * max(1e-6, xmax):
                    ax.text(0.01 * max(1.0, xmax), y[-1] + 0.2, f"attn_ref={ref:.3g}",
                            fontsize=8, ha="left", va="bottom", color=(0.25, 0.25, 0.25))
                else:
                    ax.text(ref, y[-1] + 0.2, f"attn_ref={ref:.3g}", fontsize=8, ha="center",
                            va="bottom", color=(0.25, 0.25, 0.25))
        else:
            ax.axvline(0.0, color="black", linewidth=0.8)

        if mkey in ("wg_dot", "wu_dot"):
            pre_key = mkey + "_pre"
            if pre_key in deltas_mu_sd:
                mu_pre, _ = deltas_mu_sd[pre_key]
                mu_pre = np.asarray(mu_pre, dtype=np.float32)
                ax.barh(y, mu_pre, height=0.34, alpha=0.85, edgecolor="none", color="orange")

            ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))

        ax.set_xlabel(ylab)
        ax.grid(True, axis="x", linewidth=0.4, alpha=0.35)
        ax.grid(False, axis="y")

        ax.set_yticks(y)
        ax.set_yticklabels(comp_names, fontsize=8)
        ax.set_ylabel("Residual term", fontsize=9)

    fig.suptitle(title, fontsize=12, y=0.995)
    fig.subplots_adjust(left=0.38, hspace=0.28)
    fig.tight_layout(rect=[0, 0.0, 1, 0.985])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

def _plot_wg_col_distrib(Wg, top_idx, out_path, title, highlight_rows=None):
    col = Wg[:, int(top_idx)].to(dtype=torch.float32).detach().cpu().numpy()
    plt.figure(figsize=(7, 3.5))
    counts, bins, _ = plt.hist(col, bins=200, alpha=0.6, label="all rows")
    plt.axvline(0.0, color="black", linewidth=0.8)

    if highlight_rows:
        ymax = float(np.max(counts))
        for r in highlight_rows:
            r = int(r)
            if 0 <= r < col.shape[0]:
                x = float(col[r])
                plt.axvline(x, linestyle="--", linewidth=1.0)
                plt.text(x, 0.95 * ymax, f"j={r}", rotation=90, va="top", ha="right", fontsize=8)

    plt.xlabel(f"Wg[:, {int(top_idx)}] value")
    plt.ylabel("count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def _plot_progression_multi(series, outdir, key, ylabel, title, fname, mode=None):
    if not series:
        return

    layers_all = sorted({s["layer"] for _lbl, stats in series for s in stats})
    if not layers_all:
        return

    plt.figure(figsize=(12, 6))
    ax = plt.gca()

    for lbl, stats in series:
        if not stats:
            continue
        
        by = {s["layer"]: s for s in stats}
        y = [by[L][key] for L in layers_all if L in by]
        x = [L for L in layers_all if L in by]

        yerr = None
        if len(stats) > 0 and (key + "_var") in stats[0]:
            yerr = [math.sqrt(max(0.0, float(by[L].get(key + "_var", 0.0)))) for L in x]

        if yerr is not None:
            ax.errorbar(x, y, yerr=yerr, marker="o", linewidth=1.0, elinewidth=0.8, capsize=2, label=lbl)
        else:
            ax.plot(x, y, marker="o", label=lbl)

    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)

    if mode == "cos": 
        ax.set_ylim(-1, 1)
    elif mode == "prob":
        ax.set_ylim(0, 1)
    elif mode == "res":
        ax.set_ylim(0, 10)
    elif mode == "spread":
        vals = []
        for _lbl, _stats in series:
            for s in _stats:
                v = s[key]
                vals.append(float(v))
        vmax = max(vals)
        ax.set_ylim(0.0, vmax * 1.15)
    else:
        pass # autoscale

    ax.set_title(title)
    ax.legend(loc="best", frameon=False, fontsize=8)

    ax.grid(True, which="major", axis="both", linewidth=0.5, alpha=0.35)
    ax.grid(True, which="minor", axis="both", linewidth=0.3, alpha=0.10)

    if mode == "cos":
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    elif mode == "prob":
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))

    if len(layers_all) > 0:
        ax.set_xlim(min(layers_all), max(layers_all))
        ax.set_xticks(layers_all)
        ax.set_xticklabels([str(L) for L in layers_all], rotation=0)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, fname), dpi=300)
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

def _print_list(lst):
    return "[" + ", ".join(lst) + "]"

def _pad_list(xs, k):
    xs = list(xs)
    return xs + [""] * (k - len(xs))

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
    p.add_argument("--prompt-file", default="prompts.txt")
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--head", type=int, default=0)
    # scanning mode
    p.add_argument("--scan", action="store_true")
    p.add_argument("--scan-interval", type=int, default=1, help="Interval to scan layers")
    # print mode
    p.add_argument("--print", dest="print_mode", choices=PRINT_CHOICES, default="qkv")
    p.add_argument("--qpos", default=None, help="Comma-separated list of query positions for --print qkv, eg. '256,512,768")
    p.add_argument("--sink-idx", type=int, nargs="+", default=[0], help="Indices of the sink in qkv mode")
    p.add_argument("--target-idx", type=int, default=None, help="Index of the target token for --print qkv")
    # perturbations
    p.add_argument("--rope", default=None, help="Comma-separated list of token_idx=rope_pos, eg. '12=0,42=1")
    p.add_argument("--mask", default=None, choices=["upper"], help="Causal mask type")
    p.add_argument("--random-init", action="store_true", help="DO NOT load pretrained weights")
    # lower attn metrics
    p.add_argument("--lower-attn", action="store_true", help="Iteratively lower attention on sink key at each layer")
    p.add_argument("--only-stop-layer", type=int, default=None, help="Only apply lower attention to this layer (0-indexed)")
    p.add_argument("--lower-factor", type=float, default=0.5, help="Factor to lower attention by")
    p.add_argument("--only-return-sink-value", action="store_true", help="When lowering attention, replace attention sum with only attn_to_sink * V_sink")
    # subspaces
    p.add_argument("--find-value-subspace", action="store_true", help="Collect V at token 0 and non-sink positions")
    p.add_argument("--find-key-subspace", action="store_true", help="Collect K at token 0 and non-sink positions")
    p.add_argument("--find-mlp-subspace", action="store_true", help="Collect MLP internals at token 0 and non-sink positions")
    p.add_argument("--mlp", nargs="+", choices=["z", "g", "g-row", "g-sign", "u", "residual"], default=None, help="When --find-mlp-subspace, choose which sections to print")
    p.add_argument("--pca-topk", type=int, default=6, help="Top-k PCA for groups of vectors")
    # decomposition of input
    p.add_argument("--decompose-ln", action="store_true", help="For --layer L, decompose the norm input LN(residual + attn + mlp) at L")
    p.add_argument("--decompose-output", action="store_true", help="For --layer L, decompose the output (residual + attn + mlp) at L")
    p.add_argument("--decompose-mlp-in", action="store_true")
    # ablation
    p.add_argument("--ablate-mlp-out", choices=["magnitude", "direction"], default=None, help="Ablate MLP output at token 0.")
    p.add_argument("--vec-idx", type=int, nargs="+", default=None, help="Indices of the vectors to ablate.")
    # output
    p.add_argument("--outdir", default="results")
    args = p.parse_args()

    _disable_packed_sequence_splitting()
    tok, model = load_model(args.model, args.device, args.dtype, random_init=args.random_init)

    prompts = _load_prompts_from_file(args.prompt_file)

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
    sink_idxs = [max(0, min(last_q, int(i))) for i in (args.sink_idx or [0])]

    rope_str = args.rope
    rope_overrides = parse_overrides(rope_str) if rope_str else None

    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads

    def _run_scan_pass(scan_layers, rope_overrides_local, sink_idxs_local):
        per_layer_acc = {
            L: {
                "k_norm_h": {int(s): [] for s in sink_idxs_local}, # list of [H]
                "v_norm_h": {int(s): [] for s in sink_idxs_local},
                "cos_h": {int(s): [] for s in sink_idxs_local},
                "sink_attn_h": {int(s): [] for s in sink_idxs_local},
                "out_norm": {int(s): [] for s in sink_idxs_local}, 
                "k_vecs_h": {int(s): [] for s in sink_idxs_local}, # list of [H, D]
                "v_vecs_h": {int(s): [] for s in sink_idxs_local},
                "out_vecs": {int(s): [] for s in sink_idxs_local}, # list of [D]
            }
            for L in scan_layers
        }
  
        per_layer_maps = {L: [] for L in scan_layers}

        scan_maps, scan_titles = [], []
        min_len = min(ids.shape[1] for (ids, _bp) in tokenized)
        last_q = min_len - 1
        target_q = last_q if args.target_idx is None else max(0, min(last_q, args.target_idx))

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

                        hm_all = _pick_all_heads_with_caches(model, attns, L)
                        q_all = q[0, :, 4:, :].to(dtype=torch.float32) # [H, Q', D]
                        qn = F.normalize(q_all, dim=2)

                        for sink_idx in sink_idxs_local:
                            sink_idx = int(sink_idx)
                            
                            k_sink, v_sink = k[0, :, sink_idx, :], v[0, :, sink_idx, :]  # [H, D]
                            k_norm_h, v_norm_h = k_sink.norm(dim=1), v_sink.norm(dim=1)
                        
                            kn = F.normalize(k_sink.to(dtype=torch.float32), dim=1) # [H, D]
                            cos_hq = (qn * kn[:, None, :]).sum(dim=2) # [H, Q']
                            cos_h = cos_hq.mean(dim=1) # [H]

                            sink_qmean_per_head = hm_all[:, 4:, sink_idx].mean(dim=1) # [H]

                            out_sink = mlp_out[0, sink_idx] 
                            h_sink = H[0, sink_idx]
                        
                            out_norm = float(torch.log(out_sink.norm().clamp_min(EPS)).item())

                            per_layer_acc[L]["k_norm_h"][sink_idx].append(k_norm_h.detach().cpu())
                            per_layer_acc[L]["v_norm_h"][sink_idx].append(v_norm_h.detach().cpu())
                            per_layer_acc[L]["cos_h"][sink_idx].append(cos_h.detach().cpu())
                            per_layer_acc[L]["sink_attn_h"][sink_idx].append(sink_qmean_per_head.detach().cpu())
                            per_layer_acc[L]["out_norm"][sink_idx].append(out_norm)
                            per_layer_acc[L]["k_vecs_h"][sink_idx].append(k_sink.detach().float().cpu())
                            per_layer_acc[L]["v_vecs_h"][sink_idx].append(v_sink.detach().float().cpu())
                            per_layer_acc[L]["out_vecs"][sink_idx].append(out_sink.detach().float().cpu())

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
            stats_by_idx = {int(s): [] for s in sink_idxs_local}
            for L in sorted(per_layer_acc.keys()):
                acc = per_layer_acc[L]

                for sink_idx in sink_idxs_local:
                    sink_idx = int(sink_idx)
                    k_mu, k_var = _reduce_headwise(acc["k_norm_h"][sink_idx])
                    v_mu, v_var = _reduce_headwise(acc["v_norm_h"][sink_idx])
                    cos_mu, cos_var = _reduce_headwise(acc["cos_h"][sink_idx])
                    sink_mu, sink_var = _reduce_headwise(acc["sink_attn_h"][sink_idx])

                    out_list = acc["out_norm"][sink_idx]
                    out_mu = float(np.mean(out_list))
                    out_var = float(np.var(out_list))

                    k_spread_mu, k_spread_var = _headwise_cloud_radius_mu_var(acc["k_vecs_h"][sink_idx])
                    v_spread_mu, v_spread_var = _headwise_cloud_radius_mu_var(acc["v_vecs_h"][sink_idx])
                    o_spread = _cloud_radius(torch.stack(acc["out_vecs"][sink_idx], dim=0))

                    stats_by_idx[sink_idx].append({
                        "k_norm": k_mu,
                        "k_norm_var": k_var,
                        "v_norm": v_mu,
                        "v_norm_var": v_var,
                        "cos": cos_mu,
                        "cos_var": cos_var,
                        "sink_attn": sink_mu,
                        "sink_attn_var": sink_var,
                        "out_norm": out_mu,
                        "out_norm_var": out_var,
                        "layer": L, 
                        "head": args.head,
                        "target_q": target_q,
                        "k_spread": k_spread_mu,
                        "k_spread_var": k_spread_var,
                        "v_spread": v_spread_mu,
                        "v_spread_var": v_spread_var,
                        "out_spread": o_spread,
                    })
            return stats_by_idx, scan_maps, scan_titles

        else:
            for L in sorted(per_layer_maps.keys()):
                if per_layer_maps[L]:
                    head_attn = np.mean(np.stack(per_layer_maps[L], axis=0), axis=0)
                    scan_maps.append(head_attn)
                    # DEBUG
                    src = "CACHE" if (hasattr(model, "_lower_attn_cache") and L in model._lower_attn_cache) else "ATTNS"
                    scan_titles.append(f"Layer {L} Head {args.head} [{src}]")
            return [], scan_maps, scan_titles

    lower_attn_handles = []
    if args.lower_attn:
        if args.only_stop_layer is not None:
            target_layers = [int(args.only_stop_layer)]
        else:
            target_layers = list(range(num_layers))
        if target_layers:
            print(f"[lower_attn] target_layers: {target_layers}")
            lower_attn_handles = _install_lower_attn_hooks(
                model, layers=target_layers, factor=args.lower_factor, sink_k_idx=int(args.sink_idx[0]),
                only_return_sink_value=args.only_return_sink_value
            )

    scan_maps, scan_titles = [], []

    if args.decompose_mlp_in:
        L = int(args.layer)
        tok_ablate = 0
        tok_ref = 8
        scan_layers = list(range(L + 1, model.config.num_hidden_layers))

        def _orth_unit(u):
            u = u.to(dtype=torch.float32)
            return F.normalize(u, dim=0)

        def _orth_basis(Q):
            Q = Q.to(dtype=torch.float32)
            Qb, _R = torch.linalg.qr(Q, mode="reduced")
            col_norm = Qb.norm(dim=0)
            keep = (col_norm > 1e-6).nonzero(as_tuple=False).view(-1)
            if int(keep.numel()) == 0:
                return None
            return Qb[:, keep].contiguous()

        def _make_tok0_subspace_ablation_mlp_prehook(Q_basis, tok_ablate=0):
            assert Q_basis is not None
            Qb = Q_basis.to(dtype=torch.float32).contiguous()

            def _prehook(_module, args):
                x = args[0]
                x2 = x.clone()
                v = x2[:, tok_ablate, :].to(dtype=torch.float32) # [B, D]
                Qd = Qb.to(device=v.device, dtype=torch.float32) # [D, r]
                coeff = v @ Qd # [B, r]
                proj = coeff @ Qd.T # [B, D]
                v_new = v - proj
                x2[:, tok_ablate, :] = v_new.to(dtype=x2.dtype)
                return (x2,)

            return _prehook

        def _make_tok0_Uk_gproj_ablation_preln_prehook(
            Uk_basis,
            gamma_vec,
            tok_ablate=0,
            cache=None
        ):
            assert Uk_basis is not None and Uk_basis.ndim == 2
            B0 = Uk_basis.to(dtype=torch.float32).contiguous()
            g = gamma_vec.to(dtype=torch.float32).contiguous()
            if cache is None:
                cache = {}
            cache.setdefault("proj_s_list", [])
            cache.setdefault("coeff_list", [])

            def _prehook(_module, args):
                s = args[0]
                s2 = s.clone()
                v = s2[:, tok_ablate, :].to(dtype=torch.float32)

                Bd = B0.to(device=v.device, dtype=torch.float32)
                gd = g.to(device=v.device, dtype=torch.float32).clamp_min(1e-6)
                g2d = gd * gd

                vG = v * g2d.view(1, -1)
                coeff = vG @ Bd
                proj_s = coeff @ Bd.T

                cache["proj_s_list"].append(proj_s.detach().cpu())
                cache["coeff_list"].append(coeff.detach().cpu())

                v_new = v - proj_s
                s2[:, tok_ablate, :] = v_new.to(dtype=s2.dtype)
                return (s2,)

            return _prehook, cache

        def _sample_random_subspace_orth_to(Q_ref, D, r):
            assert Q_ref is not None and Q_ref.ndim == 2
            Q_ref = Q_ref.to(dtype=torch.float32).cpu()
            for t in range(10):
                g = torch.Generator(device="cpu")
                g.manual_seed(19 + 997 * t)
                A = torch.randn((D, r), generator=g, dtype=torch.float32, device="cpu")
                A = A - Q_ref @ (Q_ref.T @ A)
                Qr = _orth_basis(A)
                if Qr is not None and int(Qr.size(1)) == int(r):
                    return Qr
            return None

        def _run_tok0_subspace_ablation(Q_basis=None):
            h = None
            mlp = model.model.layers[L].mlp
            try:
                if Q_basis is not None:
                    h = mlp.register_forward_pre_hook(
                        _make_tok0_subspace_ablation_mlp_prehook(
                            Q_basis,
                            tok_ablate=0
                        )
                    )
                data = _collect_mlp_internals_for_layer(model, L, [tok_ablate, tok_ref], tokenized)
                Z0 = data["Z"][tok_ablate]
                Z8 = data["Z"][tok_ref]

                attn_stats = _scan_sink_attention_only(model, tokenized, scan_layers)

                return {
                    "Z0": Z0,
                    "Z8": Z8,
                    "attn_stats": attn_stats,
                }
            
            finally:
                if h is not None:
                    h.remove()

        mlp_data = _collect_mlp_internals_for_layer(model, L, [tok_ablate, tok_ref], tokenized)
        Z0 = mlp_data["Z"][tok_ablate]

        z_idx, z_stats = _topk_energy_dims(Z0, k=3, min_median_mass=0.55)
        if not z_idx:
            raise ValueError(f"No consistent top-k energy dimensions found.")

        X0 = mlp_data["X"][tok_ablate]
        Xb = mlp_data["X"][tok_ref]
        mu_b = Xb.mean(dim=0)

        lambdas_list = []
        proj_p0_list = []
        proj_pb_list = []
        labels = []
        v1_list = []
        v1_dim_list = []

        def _eff_rank_from_eigs(eigs):
            x = torch.as_tensor(eigs, dtype=torch.float32)
            x = torch.clamp(x, min=0.0)
            s = torch.sum(x).clamp_min(EPS)
            p = x / s
            H = -(p * torch.log(p.clamp_min(EPS))).sum()
            return float(torch.exp(H).item())

        def _gram_diagnostics(vs, names):
            V = torch.stack([v.detach().to(dtype=torch.float32) for v in vs], dim=1)
            V = F.normalize(V, dim=0)
            Gm = (V.T @ V).cpu()

            C = Gm.numpy()
            print(f"[Gram] pairwise cosine matrix (k={C.shape[0]})")
            hdr = " " * 14 + " ".join([f"{n:>10s}" for n in names])
            print(hdr)
            for i, n in enumerate(names):
                row = " ".join([f"{C[i, j]:>10.3f}" for j in range(C.shape[1])])
                print(f"[Gram] {n:>10s} {row}")

            eigs = torch.linalg.eigvalsh(Gm).flip(0)
            s = eigs.sum().clamp_min(EPS)
            frac = (eigs / s)
            cum = torch.cumsum(frac, dim=0)

            eff_rank = _eff_rank_from_eigs(eigs)
            pr = float((s * s / torch.sum(eigs * eigs).clamp_min(EPS)).item())

            eigs_list = [float(x) for x in eigs.tolist()]
            frac_list = [float(x) for x in frac.tolist()]
            cum_list = [float(x) for x in cum.tolist()]
            print(f"[Gram] eigvals: {eigs_list}")
            print(f"[Gram] energy frac: {[round(x, 4) for x in frac_list]}")
            print(f"[Gram] cum frac: {[round(x, 4) for x in cum_list]}")
            print(f"[Gram] eff_rank(entropy): {eff_rank:.3f} | eff_rank(participation): {pr:.3f}")

        for j in z_idx:
            G = _collect_mlp_in_gradients_for_single_zdim(
                model,
                L,
                tok_ablate,
                tokenized,
                j,
            )

            Gm = G.to(dtype=torch.float32)
            _, S, Vh = torch.linalg.svd(Gm, full_matrices=False)
            lam = (S.pow(2) / float(Gm.size(0))).detach().cpu().numpy()
            lam = np.sort(lam)[::-1]

            v1 = Vh[0].detach().cpu()
            v1_list.append(v1)
            v1_dim_list.append(int(j))

            p0 = ((X0 - mu_b) @ v1).numpy()
            pb = ((Xb - mu_b) @ v1).numpy()


            lambdas_list.append(lam)
            proj_p0_list.append(p0)
            proj_pb_list.append(pb)
            labels.append(f"zdim={int(j)} | f=z[{int(j)}]^2")

        out_path = os.path.join(args.outdir, f"mlp_in_eig_L{L}_tok{tok_ablate}.png")
        title = f"Active subspace spectrum | L={L} tok={tok_ablate} | f=z[idx]^2 | z_idx={z_idx}"
        _plot_grad_second_moment_spectrum_stacked(lambdas_list, labels, out_path, title)

        out_path_proj = os.path.join(args.outdir, f"mlp_in_eig_proj_L{L}_tok{tok_ablate}.png")
        title_proj = (
            f"Projection onto top v | L={L} tok={tok_ablate} \n"
            f"p = v^T (x - mu_b)"
        )
        _plot_projection_histogram_stacked(proj_p0_list, proj_pb_list, labels, out_path_proj, title_proj, bins=60)

        v_names = [f"z{int(j)}" for j in z_idx]
        print()
        print(f"[decompose_mlp_in] L={L} tok={tok_ablate} z_idx={z_idx}")
        _gram_diagnostics(v1_list, v_names)
        print()

        torch.manual_seed(19)
        np.random.seed(19)
        random.seed(19)

        Dm = int(X0.size(1))
        Vmat = torch.stack([_orth_unit(v) for v in v1_list], dim=1)
        Q_joint = _orth_basis(Vmat.cpu())
        r = int(Q_joint.size(1))
        Q_rand_orth = _sample_random_subspace_orth_to(Q_joint, Dm, r)

        ln = model.model.layers[L].post_attention_layernorm
        gamma = ln.weight.detach().float().cpu()
        gamma_safe = gamma.sign() * gamma.abs().clamp_min(1e-6)
        
        Uk_basis = (Q_joint / gamma_safe[:, None]).contiguous()

        def _compute_g_energy(S, Qb, g, name):
            S = S.to(dtype=torch.float32)
            Qd = Qb.to(dtype=torch.float32, device=S.device)
            gd = g.to(dtype=torch.float32, device=S.device).clamp_min(1e-6)

            xlin = S * gd.view(1, -1)
            C = xlin @ Qd
            e = (C ** 2).sum(dim=1)
            tot = (xlin ** 2).sum(dim=1).clamp_min(EPS)
            frac = e / tot

            mu_e = float(e.mean().item()); sd_e = float(e.std(unbiased=False).item())
            mu_f = float(frac.mean().item()); sd_f = float(frac.std(unbiased=False).item())
            print(f"[Uk energy @S] {name}: E={mu_e:.3e}±{sd_e:.3e} | F={mu_f:.3f}±{sd_f:.3f}")
        
        run_base = _run_tok0_subspace_ablation(Q_basis=None)
        run_joint = _run_tok0_subspace_ablation(Q_basis=Q_joint)
        run_rand_orth = _run_tok0_subspace_ablation(Q_basis=Q_rand_orth)

        def _run_tok0_preln_gorth_ablation(do_ablate):
            h = None
            ln2 = model.model.layers[L].post_attention_layernorm
            hook_cache = {}
            try:
                if do_ablate:
                    prehook, hook_cache = _make_tok0_Uk_gproj_ablation_preln_prehook(
                        Uk_basis,
                        gamma_safe,
                        tok_ablate,
                        hook_cache,
                    )
                    h = ln2.register_forward_pre_hook(prehook)

                data = _collect_mlp_internals_for_layer(model, L, [tok_ablate, tok_ref], tokenized)
                X0 = data["X"][tok_ablate]
                X8 = data["X"][tok_ref]
                S0 = data["LN_in"][tok_ablate]
                S8 = data["LN_in"][tok_ref]
                attn_stats = _scan_sink_attention_only(model, tokenized, scan_layers)
                return {
                    "X0": X0,
                    "X8": X8,
                    "S0": S0,
                    "S8": S8,
                    "attn_stats": attn_stats,
                    "hook_cache": hook_cache,
                }
            finally:
                if h is not None:
                    h.remove()

        run_pre_base = _run_tok0_preln_gorth_ablation(do_ablate=False)
        run_pre_joint = _run_tok0_preln_gorth_ablation(do_ablate=True)

        def _uk_gcoords(S, Uk_basis, gamma_safe):
            S = S.to(dtype=torch.float32)
            B = Uk_basis.to(device=S.device, dtype=torch.float32)
            g2 = (gamma_safe.to(device=S.device, dtype=torch.float32) ** 2).clamp_min(EPS)
            return (S * g2.view(1, -1)) @ B

        S0_base = run_pre_base["S0"]
        S8_base = run_pre_base["S8"]

        C0 = _uk_gcoords(S0_base, Uk_basis, gamma_safe).detach().cpu().numpy()
        C8 = _uk_gcoords(S8_base, Uk_basis, gamma_safe).detach().cpu().numpy()

        r_eff = int(C0.shape[1])
        p0_list = [C0[:, i] for i in range(r_eff)]
        p8_list = [C8[:, i] for i in range(r_eff)]
        labels = [f"gcoord_{i} = (Uk^T G S)_{i}" for i in range(r_eff)]

        out_path_proj_pre = os.path.join(args.outdir, f"preln_Uk_gcoords_L{L}.png")
        title = f"G-orth coordinates onto Uk in S-space | L={L}\n"
        _plot_projection_histogram_stacked(
            p0_list=p0_list,
            pb_list=p8_list,
            labels=labels,
            out_path=out_path_proj_pre,
            title=title,
            bins=60,
        )

        Pall = torch.cat(run_pre_joint["hook_cache"]["proj_s_list"], dim=0)
        g2 = (gamma_safe ** 2).to(dtype=torch.float32)
        Pg = (Pall.to(dtype=torch.float32) ** 2) * g2.view(1, -1)
        e = Pg.sum(dim=1)
        mu_e = float(e.mean().item())
        sd_e = float(e.std(unbiased=False).item())
        print(f"[Uk proj] dataset mean ||Proj_Uk^G(S0)|| = {mu_e:.3e}±{sd_e:.3e}")

        Z0_base, Z8_base = run_base["Z0"], run_base["Z8"]
        Z0_joint = run_joint["Z0"]
        Z0_rand_orth = run_rand_orth["Z0"]

        out_path_zstack = os.path.join(args.outdir, f"mlp_in_ablate_Qproj_zstack_L{L}_tok{tok_ablate}.png")
        _plot_z_topk_energy_stacks(
            Z_list=[Z0_base, Z0_joint, Z0_rand_orth, Z8_base],
            tok_labels=[
                "tok0 (base)", "tok0 (-span(Q))", "tok0 (-span(rand⊥))", "tok8 (base)",
            ],
            topk=24,
            out_path=out_path_zstack,
            title=(
                f"Z: norm + top-k energy segments | L={L} z_idx={z_idx}"
            ),
        )

        X0_pre_base = run_pre_base["X0"]
        X0_pre_joint = run_pre_joint["X0"]

        def _coords_uncentered(X):
            Xf = X.to(dtype=torch.float32)
            Qd = Q_joint.to(dtype=torch.float32, device=X.device)
            C = Xf @ Qd # [N, r]
            return C.detach().cpu().numpy()

        C_before = _coords_uncentered(X0_pre_base)
        C_after_joint = _coords_uncentered(X0_pre_joint)

        r_eff = int(C_before.shape[1])
        proj_before_list = [C_before[:, i] for i in range(r_eff)]
        proj_after_joint_list = [C_after_joint[:, i] for i in range(r_eff)]
        proj_labels = [f"coord_{i} = X0·Q[:,{i}]" for i in range(r_eff)]

        out_path_proj_pre = os.path.join(args.outdir, f"preln_ablate_S0_target_projQ_L{L}.png")
        _plot_projection_histogram_before_after_stacked(
            p_before_list=proj_before_list,
            p_after_list=proj_after_joint_list,
            labels=proj_labels,
            out_path=out_path_proj_pre,
            title=f"Projection onto sink subspace in MLP space | S0 ablation @ L={L} tok={tok_ablate}",
            bins=60,
        )

        stats_base = run_base["attn_stats"]
        stats_joint = run_joint["attn_stats"]
        stats_rand_orth = run_rand_orth["attn_stats"]

        stats_pre_joint = run_pre_joint["attn_stats"]

        series = [
            ("baseline", sorted(stats_base, key=lambda x: x["layer"])),
            ("X0 (-span(X subspace))", sorted(stats_joint, key=lambda x: x["layer"])),
            ("X0 (-span(X rand⊥))", sorted(stats_rand_orth, key=lambda x: x["layer"])),
            ("S0 (-span(S subspace))", sorted(stats_pre_joint, key=lambda x: x["layer"])),
        ]
        
        out_path_sink = f"mlp_in_ablate_Qproj_attnL{L}_tok{tok_ablate}.png"
        _plot_progression_multi(
            series,
            args.outdir,
            key="sink_attn",
            ylabel=f"Attn(Q -> K[0]) (var across heads)",
            title=f"Sink attention after ablating direction @ L={L}",
            fname=out_path_sink,
            mode="prob",
        )

        return

    if args.ablate_mlp_out:
        L = int(args.layer)
        scan_layers = list(range(L + 1, num_layers))
        
        vec_idx = None
        if args.ablate_mlp_out == "direction":
            mlp_data = _collect_mlp_internals_for_layer(model, L, [0], tokenized)
            Z0 = mlp_data["Z"][0]
            vec_idx, _zst = _topk_energy_dims(
                Z0, k=3, min_median_mass=0.55
            )

        Q = None
        if args.ablate_mlp_out == "direction":
            Q = _compute_Wd_subspace(model, L, vec_idx)
        
        stats_before = _scan_sink_attention_only(model, tokenized, scan_layers)
        ablate_handles = _install_mlp_out_ablation_hook(
            model, 
            L, 
            args.ablate_mlp_out, 
            vec_idx, 
            tok_ablate=0, 
            tok_ref=8
        )
        try:
            stats_after = _scan_sink_attention_only(model, tokenized, scan_layers)
        finally:
            for h in ablate_handles:
                h.remove()

        stats_local = []
        
        if Q is not None:
            for L2 in scan_layers:
                local_handles = _install_local_input_subspace_ablation_hook(
                    model,
                    layer_idx=L2,
                    Q=Q,
                    tok_ablate=0,
                )
                try:
                    stats = _scan_sink_attention_only(model, tokenized, [L2])
                    stats_local.extend(stats)
                finally:
                    for h in local_handles:
                        h.remove()

        before = sorted(stats_before, key=lambda x: x["layer"])
        after = sorted(stats_after, key=lambda x: x["layer"])
        local = sorted(stats_local, key=lambda x: x["layer"])
        series = [("baseline", before), ("mlp_out_ablated", after), ("layer_input_ablated", local)]

        tag = f"ablate_{args.ablate_mlp_out}_L{L}_token_0"
        if args.ablate_mlp_out == "direction" and args.vec_idx:
            tag += "_vec" + "_".join(map(str, args.vec_idx))

        _plot_progression_multi(
            series, args.outdir, key="sink_attn", ylabel=f"Attn(Q[q>=4] -> K[0]) (var across heads)",
            title=f"Attention score before/after ablation={args.ablate_mlp_out} | L={L} | token=0",
            fname=f"ablate_mlp_out_{tag}_L{L}.png", 
            mode="prob",
        )
        return

    if args.scan:
        # perturbed pass
        scan_layers = list(range(0, num_layers, args.scan_interval))
        scan_stats_by_sink, scan_maps, scan_titles = _run_scan_pass(scan_layers, rope_overrides, sink_idxs)

        if args.print_mode == "qkv":
            for s in scan_stats_by_sink:
                scan_stats_by_sink[s] = sorted(scan_stats_by_sink[s], key=lambda x: x["layer"])
            q_label = f"Q[{args.target_idx}]" if args.target_idx is not None else "Q[last]"

            series = [(f"sink={int(s)}", scan_stats_by_sink[int(s)]) for s in sorted(scan_stats_by_sink.keys())]

            _plot_progression_multi(
                series, args.outdir, key="k_norm", ylabel=f"||K|| (var across heads)",
                title=f"K-norm progression", fname="scan_knorm.png",
            )
            _plot_progression_multi(
                series, args.outdir, key="v_norm", ylabel=f"||V|| (var across heads)",
                title=f"V-norm progression", fname="scan_vnorm.png",
            )
            _plot_progression_multi(
                series, args.outdir, key="cos", ylabel=f"cos(Q[q>=4], K) (var across heads)",
                title=f"Cosine to K across layers", fname="scan_cosine.png",
                mode="cos"
            )
            _plot_progression_multi(
                series, args.outdir, key="sink_attn", ylabel=f"Attn(Q[q>=4] -> K) (var across heads)",
                title=f"Mean attention score across layers", fname="scan_sink_attn_mean.png",
                mode="prob",
            )
            _plot_progression_multi(
                series, args.outdir, key="out_norm", ylabel=f"log ||MLP_out|| (var across samples)",
                title=f"MLP output activation norm progression", fname="scan_mlp_out_norm.png",
                mode="res",
            )
            _plot_progression_multi(
                series, args.outdir, key="k_spread", ylabel=f"spread(K) (var across heads)",
                title=f"K spread progression", fname="scan_k_spread.png",
                mode="spread",
            )
            _plot_progression_multi(
                series, args.outdir, key="v_spread", ylabel=f"spread(V) (var across heads)",
                title=f"V spread progression", fname="scan_v_spread.png",
                mode="spread",
            )
            _plot_progression_multi(
                series, args.outdir, key="out_spread", ylabel=f"spread(MLP_out) (var across heads)",
                title=f"MLP_out spread progression", fname="scan_mlp_out_spread.png",
                mode="spread",
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
            if args.lower_attn:
                segment = "lower_attn"
                if args.only_stop_layer is not None:
                    segment += f" only_stop[{args.only_stop_layer}]"
                grid_title += f" - {segment} (x{args.lower_factor})"

            _plot_heatmap_grid(
                scan_maps, scan_titles, os.path.join(args.outdir, "scan_heatmaps.png"), rows=4,
                suffix="", suptitle=grid_title,
            )
            
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
            mlp_sections = set(args.mlp) if args.mlp is not None else {"z", "g", "g-row", "g-sign", "u", "residual"}

            mlp_data = _collect_mlp_internals_for_layer(model, L, tok_idxs, tokenized)
            assert 0 in mlp_data["Y"]

            # fetch down projection column vectors
            mlp = model.model.layers[L].mlp
            Wd = mlp.down_proj.weight.detach().to(dtype=torch.float32).cpu()
            Wg = mlp.gate_proj.weight.detach().to(dtype=torch.float32).cpu()
            Wu = mlp.up_proj.weight.detach().to(dtype=torch.float32).cpu()

            gamma = model.model.layers[L].post_attention_layernorm.weight.detach().to(dtype=torch.float32).cpu()
            gamma_neg_count = int((gamma < 0).sum().item())
            gamma_neg_top_idx, gamma_neg_top_vals = _vec_neg_topk(gamma, k=gamma_neg_count)
            
            _g_abs = gamma.abs()
            gamma_topk_idx = torch.topk(_g_abs, k=2).indices.detach().cpu().tolist()
            gamma_top_idx = gamma_topk_idx[0]
            gamma_top_idx_2 = gamma_topk_idx[1]

            k = 3
            k_fracs = (0.25, 0.75, 0.99)

            col_names = ["tok0", "tok1", "tok8"]

            Z0 = mlp_data["Z"][0]; Z1 = mlp_data["Z"][1]; Z8 = mlp_data["Z"][8] 
            G0 = mlp_data["G"][0]; G1 = mlp_data["G"][1]; G8 = mlp_data["G"][8]
            U0 = mlp_data["U"][0]; U1 = mlp_data["U"][1]; U8 = mlp_data["U"][8]
            A0 = mlp_data["A"][0]; A1 = mlp_data["A"][1]; A8 = mlp_data["A"][8]
            X0 = mlp_data["X"][0]; X1 = mlp_data["X"][1]; X8 = mlp_data["X"][8]

            S0 = mlp_data["LN_in"][0].float()
            S1 = mlp_data["LN_in"][1].float()
            S8 = mlp_data["LN_in"][8].float()

            S0_rms = S0 / _rms(S0, dim=1, keepdim=True)
            S1_rms = S1 / _rms(S1, dim=1, keepdim=True)
            S8_rms = S8 / _rms(S8, dim=1, keepdim=True)

            # sanity checks
            diff0 = Z0 - A0 * U0
            z0_rel_l2 = float((
                diff0.norm(dim=1) / Z0.norm(dim=1).clamp_min(EPS)
            ).mean().item())
            print(f"[Sanity check] reconstruction mean_rel_l2={z0_rel_l2:.4e}")

            def _print_mlp_z_section():
                S_act0 = _topk_activated_set(Z0, k) # should be the same as S_proj
                S_act1 = _topk_activated_set(Z1, k)
                S_act8 = _topk_activated_set(Z8, k)

                z_norm_vals = [
                    _vec_norm_mu_sd_str(Z0),
                    _vec_norm_mu_sd_str(Z1),
                    _vec_norm_mu_sd_str(Z8),
                ]
                z_kfrac_vals = [
                    _k_for_energy_frac_str(Z0, k_fracs),
                    _k_for_energy_frac_str(Z1, k_fracs),
                    _k_for_energy_frac_str(Z8, k_fracs),
                ]
                z_topk_idx_vals = [
                    _print_list(map(str, _topk_energy_dims(Z0, k=k)[0])),
                    _print_list(map(str, _topk_energy_dims(Z1, k=k)[0])),
                    _print_list(map(str, _topk_energy_dims(Z8, k=k)[0])),
                ]

                z0_topk_act_idx = _topk_energy_dims(Z0, k=k)[0]

                w_ref = F.normalize(Wd[:, z0_topk_act_idx[0]], dim=0)
                wdown_cos_to_top1 = []
                for i in z0_topk_act_idx:
                    wi = F.normalize(Wd[:, i], dim=0)
                    wdown_cos_to_top1.append(float(torch.dot(wi, w_ref).item()))
                wdown_cos_to_top1_vals = [
                    _print_list(f"{v:.2f}" for v in wdown_cos_to_top1),
                    None,
                    None,
                ]

                colnorm_vals = [
                    _mean_topk_column_norm(Wd, S_act0),
                    _mean_topk_column_norm(Wd, S_act1),
                    _mean_topk_column_norm(Wd, S_act8),
                ]

                metrics_z = [
                    ("z_norm", z_norm_vals, "sci"),
                    ("z_frac_energy [.25/.75/.99]", z_kfrac_vals, ""),
                    (f"z_topk_act_idx k={k}", z_topk_idx_vals, ""),
                    (f"W_down_cos_to_top1_aligned_vec k={k}", wdown_cos_to_top1_vals, ""),
                    (f"W_down_topk_act_norm k={k}", colnorm_vals, "float"),
                ]
                print()
                _print_metrics_table(metrics_z, col_names, title=f"[MLP subspace: Z/W_d] layer={L}")

                out_path = os.path.join(args.outdir, f"MLP_Z_topk_energy_stacks_L{L}.png")
                _plot_z_topk_energy_stacks(
                    Z_list=[Z0, Z1, Z8],
                    tok_labels=["tok0", "tok1", "tok8"],
                    topk=24,
                    out_path=out_path,
                    title=f"Z: norm + top-k energy segments per token | L={L}"
                )
            
            def _print_mlp_g_section():
                g_entropy_vals = [
                    _mean_activation_entropy(G0),
                    _mean_activation_entropy(G1),
                    _mean_activation_entropy(G8),
                ]
                a_entropy_vals = [
                    _mean_activation_entropy(A0),
                    _mean_activation_entropy(A1),
                    _mean_activation_entropy(A8),
                ]
                g_pos_ratio_vals = [
                    _pos_mu_sd_str(G0),
                    _pos_mu_sd_str(G1),
                    _pos_mu_sd_str(G8),
                ]
                a_norm_vals = [
                    _vec_norm_mu_sd_str(A0),
                    _vec_norm_mu_sd_str(A1),
                    _vec_norm_mu_sd_str(A8)
                ]
                a_kfrac_vals = [
                    _k_for_energy_frac_str(A0, (0.25, 0.50, 0.7)),
                    _k_for_energy_frac_str(A1, (0.25, 0.50, 0.7)),
                    _k_for_energy_frac_str(A8, (0.25, 0.50, 0.7)),
                ]
                g_top_abs_vals = [
                    _topk_vals_mu_sd_str(G0, k=3, mode="abs"),
                    _topk_vals_mu_sd_str(G1, k=3, mode="abs"),
                    _topk_vals_mu_sd_str(G8, k=3, mode="abs"),
                ]
                k_a = 5
                a_topk_idx0 = _topk_energy_dims(A0, k=k_a)[0]
                a_topk_idx1 = _topk_energy_dims(A1, k=k_a)[0]
                a_topk_idx8 = _topk_energy_dims(A8, k=k_a)[0]

                g_top_pos_vals = [
                    _topk_vals_mu_sd_str(G0, k=3, mode="pos"),
                    _topk_vals_mu_sd_str(G1, k=3, mode="pos"),
                    _topk_vals_mu_sd_str(G8, k=3, mode="pos"),
                ]
                a_top_pos_vals = [
                    _topk_vals_mu_sd_str(A0, k=3, mode="pos"),
                    _topk_vals_mu_sd_str(A1, k=3, mode="pos"),
                    _topk_vals_mu_sd_str(A8, k=3, mode="pos"),
                ]

                metrics_ga = [
                    ("g_entropy", g_entropy_vals, "float"),
                    ("g_pos/all", g_pos_ratio_vals, ""),
                    ("g_max_vals (abs)", g_top_abs_vals, ""),
                    ("g_max_vals (pos)", g_top_pos_vals, ""),
                    ("a_max_vals (pos)", a_top_pos_vals, ""),
                    ("a_entropy", a_entropy_vals, "float"),
                    ("a_norm", a_norm_vals, "sci"),
                    (f"a_topk_act_idx k={k_a}", [
                        _print_list(map(str, a_topk_idx0)),
                        _print_list(map(str, a_topk_idx1)),
                        _print_list(map(str, a_topk_idx8)),
                    ], ""),
                    # (f"a_frac_energy [.25/.5/.99]", a_kfrac_vals, ""),
                    (f"a_explained_energy@top{k}_idx", [
                        _fracmass_on_set(A0, set(a_topk_idx0)),
                        _fracmass_on_set(A1, set(a_topk_idx1)),
                        _fracmass_on_set(A8, set(a_topk_idx8)),
                    ], "float")
                ]

                print()
                _print_metrics_table(metrics_ga, col_names, title=f"[MLP subspace: G/A] layer={L}")

            def _print_mlp_u_section():
                u_norm_vals = [
                    float(U0.norm(dim=1).mean().item()),
                    float(U1.norm(dim=1).mean().item()),
                    float(U8.norm(dim=1).mean().item()),
                ]
                u_topk_idx0 = _topk_energy_dims(U0, k=k)[0]
                u_topk_idx1 = _topk_energy_dims(U1, k=k)[0]
                u_topk_idx8 = _topk_energy_dims(U8, k=k)[0]

                u_topk_idx_vals = [
                    _print_list(map(str, u_topk_idx0)),
                    _print_list(map(str, u_topk_idx1)),
                    _print_list(map(str, u_topk_idx8)),
                ]

                u_topk_explained_energy_vals = [
                    _fracmass_on_set(U0, set(u_topk_idx0)),
                    _fracmass_on_set(U1, set(u_topk_idx1)),
                    _fracmass_on_set(U8, set(u_topk_idx8)),
                ]

                metrics_u = [
                    ("up_u_norm", u_norm_vals, "sci"),
                    (f"up_u_tok_act_idx k={k}", u_topk_idx_vals, ""),
                    (f"u_explained_energy@top{k}_idx", u_topk_explained_energy_vals, "float"),
                ]

                print()
                _print_metrics_table(metrics_u, col_names, title=f"[MLP subspace: U] layer={L}")  

                probe_idx = [8518, 422, 5723]
                _plot_wg_row_energy_stack(
                    S0_rms,
                    Wu,
                    row_idxs=probe_idx,
                    out_path=os.path.join(args.outdir, f"MLP_Wu_row_contrib_to_dot_tok0_L{L}.png"),
                    title=f"Per-dim contrib to dot product (Xpre_rms_tok0 @ Wu_row) | L={L}",
                    squash_boundary=24,
                    topk=3,
                )

            def _print_mlp_residual_section():
                probe_idx = [8518, 422, 5723]

                top2_dim, _cnts, _esum = _select_topk_dims_from_rows(
                    S0_rms,
                    [Wg, Wu],
                    probe_idx=probe_idx,
                    topk=2,
                    topk_per_row=3,
                )

                decomp = _collect_residual_decomp_to_layer(model, L, tok_idxs=[0, 8], tokenized=tokenized)
                labels = decomp["labels"]
                data = decomp["data"]

                comp_keys = []
                comp_names = []
                comp_keys.append("embed"); comp_names.append(labels["embed"])
                for i in range(0, L):
                    comp_keys.append(f"attn_{i}"); comp_names.append(labels[f"attn_{i}"])
                    comp_keys.append(f"mlp_{i}"); comp_names.append(labels[f"mlp_{i}"])
                comp_keys.append(f"attn_{L}"); comp_names.append(labels[f"attn_{L}"])

                dims = [int(d) for d in top2_dim]
                C = int(len(comp_keys))
                tok0_mat = np.zeros((len(dims), C), dtype=np.float32)
                diff_mat = np.zeros((len(dims), C), dtype=np.float32)

                for ci, key in enumerate(comp_keys):
                    X0 = data[key].get(0, None)
                    X8 = data[key].get(8, None)
                    assert X0 is not None and X8 is not None

                    for di, d in enumerate(dims):
                        v0 = X0[:, d].to(dtype=torch.float32).mean().item()
                        vd = (X0[:, d].to(dtype=torch.float32) - X8[:, d].to(dtype=torch.float32)).mean().item()
                        tok0_mat[di, ci] = float(v0)
                        diff_mat[di, ci] = float(vd)

                out_path = os.path.join(args.outdir, f"MLP_residual_decomp_dims_tok0_vs_tok8_L{L}.png")
                title = f"[Residual decomposition to MLP input] layer={L} | dims={dims} | row_vecs={probe_idx}"
                _plot_residual_dim_heatmaps(
                    dim_list=dims,
                    comp_names=comp_names,
                    tok0_mat=tok0_mat,
                    diff_mat=diff_mat,
                    out_path=out_path,
                    title=title,
                )
                print()
                print(f"[MLP residual decomp] auto-picked top dims: {dims} | saved: {out_path}")

            if "z" in mlp_sections:
                _print_mlp_z_section()
            if "g" in mlp_sections:
                _print_mlp_g_section()
            if "u" in mlp_sections:
                _print_mlp_u_section()
            if "residual" in mlp_sections:
                _print_mlp_residual_section()

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

if __name__ == "__main__":
    main()