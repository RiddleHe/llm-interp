import argparse, os, json, numpy as np, torch, matplotlib.pyplot as plt, math, copy
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

def tokenize(tok, text, add_special_tokens=True):
    enc = tok(text, return_tensors="pt", add_special_tokens=add_special_tokens)
    return enc["input_ids"]

def _apply_input_id_edits(input_ids, set_pairs=None, swap_pair=None):
    """
    Apply simple, position-indexed edits to a 1D sequence (batched as [1, S]).

    - set_pairs: list[(pos, token_id)] to overwrite
    - swap_pair: tuple(pos_a, pos_b) to swap
    """
    if (set_pairs is None or len(set_pairs) == 0) and (swap_pair is None):
        return input_ids
    ids = input_ids.clone()
    S = int(ids.shape[1])

    if set_pairs:
        for pos, tid in set_pairs:
            if 0 <= pos < S:
                ids[0, pos] = int(tid)
            else:
                raise ValueError(f"[set-token-ids] position {pos} out of range (seq_len={S})")

    if swap_pair is not None:
        a, b = int(swap_pair[0]), int(swap_pair[1])
        if not (0 <= a < S and 0 <= b < S):
            raise ValueError(f"[swap-token-positions] positions {a}, {b} out of range (seq_len={S})")
        tmp = ids[0, a].clone()
        ids[0, a] = ids[0, b]
        ids[0, b] = tmp
    return ids

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

def _install_sink_key_ablation_hooks(
    model,
    layers,
    head_idxs,
    sink_k_idx=0,
    factor=0.0,
    renorm=True,
    q_start=None,
    q_end=None,
):
    """
    Selectively scale attention probability to a specific key index (sink_k_idx)
    for a subset of heads at specific layers, optionally re-normalizing rows.

    This is intended for causal tests: does removing attn-to-key0 in a small set
    of "registration heads" eliminate the sink-attention jump?
    """
    head_idxs = [int(h) for h in head_idxs]
    layers = [int(L) for L in layers]
    factor = float(factor)
    renorm = bool(renorm)

    handles = []
    num_heads = model.config.num_attention_heads
    num_kv = model.config.num_key_value_heads
    num_groups = max(1, num_heads // max(1, num_kv))

    model._lower_attn_cache = {}  # reuse existing cache reader paths

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
            attn_output, attn_probs = output[0], output[1]  # [B,H,Q,K]
            probs = attn_probs.clone()

            B, H, Q, K = probs.shape
            hs = [h for h in head_idxs if 0 <= h < H]
            if not hs:
                return output
            qs = 0 if q_start is None else max(0, min(int(q_start), Q))
            qe = Q if q_end is None else max(0, min(int(q_end), Q))
            if qe <= qs:
                qs, qe = 0, Q

            probs[:, hs, qs:qe, sink_k_idx] *= factor
            if renorm:
                sub = probs[:, hs, qs:qe, :]
                denom = sub.sum(dim=-1, keepdim=True).clamp_min(EPS)
                probs[:, hs, qs:qe, :] = sub / denom

            v = _cache["v"]
            ctx = torch.matmul(probs, v)  # [B,H,Q,D]
            ctx = ctx.transpose(1, 2).contiguous().view(B, Q, -1)
            new_out = _module.o_proj(ctx)

            model._lower_attn_cache[_L] = probs.detach().float().cpu()

            out_list = list(output)
            out_list[0] = new_out
            out_list[1] = probs
            return tuple(out_list)

        h1 = attn.register_forward_pre_hook(_prehook, with_kwargs=True)
        h2 = attn.register_forward_hook(_fwdhook)
        handles.append([h1, h2])

    return handles

def _install_mlp_out_ablation_hooks(model, layers, tok_idxs, factor=0.0):
    """
    Selectively scale MLP output for certain token positions at certain layers.
    This ablates the *write* step (your L6 MLP finding).
    """
    layers = [int(L) for L in layers]
    tok_idxs = [int(t) for t in tok_idxs]
    factor = float(factor)
    handles = []
    for L in layers:
        mlp = model.model.layers[L].mlp

        def _mk_hook(_L):
            def fwd_hook(_module, args, output):
                # output: [B, S, D]
                out = output
                try:
                    B, S, D = out.shape
                except Exception:
                    return output
                out2 = out.clone()
                for t in tok_idxs:
                    if 0 <= t < S:
                        out2[:, t, :] *= factor
                return out2
            return fwd_hook

        h = mlp.register_forward_hook(_mk_hook(L))
        handles.append(h)
    return handles

def _install_attn_out_ablation_hooks(model, layers, tok_idxs, factor=0.0):
    """
    Selectively scale attention output (the tensor returned by self_attn, before residual add)
    for certain token positions at certain layers.

    This lets us do Experiment-2 style "output point = 0" interventions for the Attention branch.
    """
    layers = [int(L) for L in layers]
    tok_idxs = [int(t) for t in tok_idxs]
    factor = float(factor)
    handles = []
    for L in layers:
        attn = model.model.layers[L].self_attn

        def _mk_hook(_L):
            def fwd_hook(_module, args, output):
                # output: typically tuple(attn_out, attn_probs, ...)
                try:
                    attn_out = output[0]  # [B, S, D]
                except Exception:
                    return output
                try:
                    B, S, D = attn_out.shape
                except Exception:
                    return output
                out2 = attn_out.clone()
                for t in tok_idxs:
                    if 0 <= t < S:
                        out2[:, t, :] *= factor
                out_list = list(output) if isinstance(output, (tuple, list)) else [output]
                if out_list:
                    out_list[0] = out2
                    return tuple(out_list) if isinstance(output, tuple) else out_list
                return output
            return fwd_hook

        h = attn.register_forward_hook(_mk_hook(L))
        handles.append(h)
    return handles

def _install_mask_edit_hooks(
    model,
    layers,
    split_boundary=None,
    split_renorm=True,
    q0_allow_k=None,
):
    """
    Mask-edit interventions implemented as post-softmax probability edits + recompute attention output.

    - split_boundary=m: for queries q>=m, set probs[..., :m]=0 then renorm (creates a second causal boundary).
    - q0_allow_k=k: override q=0 probs to be uniform over keys [0..k-1] (breaks the unique q=0 boundary signature).
    """
    layers = [int(L) for L in layers]
    split_boundary = None if split_boundary is None else int(split_boundary)
    q0_allow_k = None if q0_allow_k is None else int(q0_allow_k)
    split_renorm = bool(split_renorm)

    handles = []
    num_heads = model.config.num_attention_heads
    num_kv = model.config.num_key_value_heads
    num_groups = max(1, num_heads // max(1, num_kv))

    # Reuse cache readers for heatmaps / sink-attn computations
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
            attn_output, attn_probs = output[0], output[1]  # [B,H,Q,K]
            probs = attn_probs.clone()
            B, H, Q, K = probs.shape

            # 1) split boundary: second causal boundary at m
            if split_boundary is not None:
                m = max(0, min(split_boundary, Q))
                if m < Q:
                    probs[:, :, m:, :m] = 0.0
                    if split_renorm:
                        sub = probs[:, :, m:, :]
                        denom = sub.sum(dim=-1, keepdim=True).clamp_min(EPS)
                        probs[:, :, m:, :] = sub / denom

            # 2) break q=0 boundary privilege: overwrite probs[q=0] to uniform over first k keys
            if q0_allow_k is not None and Q > 0:
                k = max(1, min(q0_allow_k, K))
                probs[:, :, 0, :] = 0.0
                probs[:, :, 0, :k] = 1.0 / float(k)

            v = _cache["v"]
            ctx = torch.matmul(probs, v)  # [B,H,Q,D]
            ctx = ctx.transpose(1, 2).contiguous().view(B, Q, -1)
            new_out = _module.o_proj(ctx)

            model._lower_attn_cache[_L] = probs.detach().float().cpu()

            out_list = list(output)
            out_list[0] = new_out
            out_list[1] = probs
            return tuple(out_list)

        h1 = attn.register_forward_pre_hook(_prehook, with_kwargs=True)
        h2 = attn.register_forward_hook(_fwdhook)
        handles.append([h1, h2])

    return handles

@torch.no_grad()
def eval_loss(model, input_ids, position_ids):
    device = next(model.parameters()).device
    ids = input_ids.to(device)
    pos = position_ids.to(device)
    out = model(
        input_ids=ids,
        position_ids=pos,
        labels=ids,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
    )
    return float(out.loss.detach().float().item())

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

def _row_norm_rank_desc(W):
    W = W.to(dtype=torch.float32)
    norms = W.norm(dim=1)
    order = torch.argsort(norms, descending=True)
    rank = torch.empty_like(order)
    rank[order] = torch.arange(1, order.numel() + 1, device=order.device)
    return norms, rank

def _mean_cos_row_vs_X(w_row, X):
    w = F.normalize(w_row.to(dtype=torch.float32), dim=0)
    Xu = F.normalize(X.to(dtype=torch.float32), dim=1)
    return float((Xu @ w).mean().item())

def _mean_frac_energy_on_dir(X, w_dir):
    X = X.to(dtype=torch.float32)
    u = F.normalize(w_dir.to(dtype=torch.float32), dim=0).to(device=X.device)
    denom = (X ** 2).sum(dim=1).clamp_min(EPS)
    num = (X @ u).pow(2)
    return float((num / denom).mean().item())

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
    attn = model.model.layers[layer_idx].self_attn
    cache = {}

    h_blk = model.model.layers[layer_idx].register_forward_pre_hook(
        lambda m, a, k, _c=cache: _c.__setitem__("attn_in", _get_hidden_from_hook_args(a, k).detach().float().cpu()),
        with_kwargs=True,
    ) # capture attn input
    def _attn_hook(_module, args, output, _cache=cache):
        attn_out = output[0]
        _cache["attn_out"] = attn_out.detach().float().cpu()
    h_attn = attn.register_forward_hook(_attn_hook) # capture attn output
    # attn output + attn input -> MLP input

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
    Attn_in_tok = copy.deepcopy(X_tok)
    Attn_out_tok = copy.deepcopy(X_tok)

    try:
        for input_ids, base_pos in tqdm(
            tokenized,
            desc=f"Collecting MLP internals at L={layer_idx}",
            leave=False,
        ):
            cache.clear()
            _ = prefill(model, input_ids, base_pos)
            assert all(k in cache for k in ["x", "g", "u", "a", "z", "y", "attn_in", "attn_out"])

            x = cache["x"][0]
            g = cache["g"][0]
            u = cache["u"][0]
            a = cache["a"][0]
            z = cache["z"][0]
            y = cache["y"][0]
            attn_in = cache["attn_in"][0]
            attn_out = cache["attn_out"][0]
            S = int(x.shape[0])
            for t in tok_idxs:
                assert t < S
                X_tok[t].append(x[t].clone())
                G_tok[t].append(g[t].clone())
                U_tok[t].append(u[t].clone())
                A_tok[t].append(a[t].clone())
                Z_tok[t].append(z[t].clone())
                Y_tok[t].append(y[t].clone())
                Attn_in_tok[t].append(attn_in[t].clone())
                Attn_out_tok[t].append(attn_out[t].clone())

    finally:
        for _h in [h, h_attn, h_blk]:
            try:
                _h.remove()
            except Exception:
                pass

    out = {"X": {}, "G": {}, "U": {}, "A": {}, "Z": {}, "Y": {}, "Attn_in": {}, "Attn_out": {}}
    for name, bucket in [
        ("X", X_tok),
        ("G", G_tok),
        ("U", U_tok),
        ("A", A_tok),
        ("Z", Z_tok),
        ("Y", Y_tok),
        ("Attn_in", Attn_in_tok),
        ("Attn_out", Attn_out_tok),
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

def parse_int_list(s):
    """'6,23,5' -> [6, 23, 5]"""
    if not s:
        return []
    out = []
    for part in str(s).split(","):
        part = part.strip()
        if part:
            out.append(int(part))
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

def pick_head_or_mean(attentions, layer_idx, head_idx):
    """
    head_idx:
      - int >= 0: pick that head
      - -1: mean over heads
    """
    attn_l = attentions[layer_idx]
    if int(head_idx) == -1:
        return attn_l[0].mean(dim=0).detach().float().cpu().numpy()
    return attn_l[0, int(head_idx)].detach().float().cpu().numpy()

def _pick_head_with_caches(model, attns, layer_idx, head_idx):
    if hasattr(model, "_lower_attn_cache") and (layer_idx in model._lower_attn_cache):
        return pick_head(model._lower_attn_cache, layer_idx, head_idx)
    return pick_head(attns, layer_idx, head_idx)

def _pick_head_or_mean_with_caches(model, attns, layer_idx, head_idx):
    if hasattr(model, "_lower_attn_cache") and (layer_idx in model._lower_attn_cache):
        return pick_head_or_mean(model._lower_attn_cache, layer_idx, head_idx)
    return pick_head_or_mean(attns, layer_idx, head_idx)

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

def _plot_step_cos(step_names, mean_cos_by_tok, outdir, fname, title):
    """
    step_names: list[str] for steps *excluding* the initial "embed" step, i.e. these are transitions
               where y[i] = cos(step_i, step_{i-1}).
    mean_cos_by_tok: dict[int -> list[float]] each list aligned with step_names.
    """
    if not step_names or not mean_cos_by_tok:
        return
    xs = np.arange(len(step_names))
    plt.figure(figsize=(max(9, 0.55 * len(step_names)), 3.6))
    ax = plt.gca()
    for t, ys in sorted(mean_cos_by_tok.items(), key=lambda kv: kv[0]):
        if ys is None or len(ys) != len(step_names):
            continue
        ax.plot(xs, ys, marker="o", linewidth=1.4, markersize=3, label=f"tok[{t}]")
    ax.set_xticks(xs)
    ax.set_xticklabels(step_names, rotation=55, ha="right")
    ax.set_ylabel("cos(step_t, step_{t-1})")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.25)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8, ncols=2)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, fname), dpi=300)
    plt.close()

def _plot_zero_point_sweep(results, outdir, fname, title, sort=True, yerr=None, yerr_label=None):
    """
    results: list[dict] with keys:
      - point (str), layer (int), kind (str)
      - baseline_jump, ablated_jump, delta_jump
      - baseline_L, baseline_Lm1, ablated_L, ablated_Lm1
    """
    if not results:
        return
    rows = results[:]
    if sort:
        rows = sorted(rows, key=lambda r: float(r.get("delta_jump", 0.0)), reverse=True)
    labels = [r["point"] for r in rows]
    ys = [float(r.get("delta_jump", 0.0)) for r in rows]
    xs = np.arange(len(labels))
    plt.figure(figsize=(max(10, 0.55 * len(labels)), 3.9))
    ax = plt.gca()
    if yerr is not None:
        ax.bar(xs, ys, yerr=yerr, capsize=2.5, color="tab:blue", alpha=0.85, ecolor="black", linewidth=0.4)
        if yerr_label:
            ax.text(
                0.99, 0.98, f"error bar: {yerr_label}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8, alpha=0.8
            )
    else:
        ax.bar(xs, ys, color="tab:blue", alpha=0.85)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=65, ha="right", fontsize=8)
    ax.set_ylabel("delta jump = baseline_jump - ablated_jump")
    ax.grid(axis="y", alpha=0.25)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, fname), dpi=300)
    plt.close()

def _find_layer_attn_input_norm(layer):
    """
    Best-effort find the *pre-attention* norm module inside a decoder layer.
    We hook this module to edit the tensor fed into attention (while keeping the residual path unchanged).
    """
    cand = [
        "input_layernorm",           # LLaMA-style
        "input_layer_norm",
        "ln_1",                      # GPT-style
        "norm1",
        "self_attn_layer_norm",
        "attention_norm",
        "attn_norm",
    ]
    for name in cand:
        if hasattr(layer, name):
            return getattr(layer, name), name
    # Fall back to a heuristic search by name/type.
    for name, mod in layer.named_modules():
        n = name.lower()
        if any(k in n for k in ["input_layernorm", "input_layer_norm", "ln_1", "norm1", "attn_norm", "attention_norm", "self_attn_layer_norm"]):
            return mod, name
    return None, None

def _install_l0mlp_path_split_hooks(model, tok_idx=0, mode="no_into_l1_attn"):
    """
    Split the influence of L0 MLP(t0) into two routes at L1:
      - mode='no_into_l1_attn': remove L0_mlp[t0] from the tensor that is *normalized and fed into L1 attention*.
                               Residual add around L1 attention still carries L0_mlp[t0] forward.
      - mode='no_bypass_l1_attn': keep L1 attention computation unchanged, but subtract L0_mlp[t0] from L1 attn_out[t0]
                                  so that after residual add, the direct bypass contribution is cancelled.
    """
    t = int(tok_idx)
    handles = []
    cache = {}

    # 1) Capture L0 MLP output vector for tok t (per prompt).
    mlp0 = model.model.layers[0].mlp

    def _cap_mlp0(_module, _args, output, _cache=cache, _t=t):
        out = output
        try:
            B, S, D = out.shape
        except Exception:
            return output
        if 0 <= _t < S:
            _cache["mlp0_tok"] = out[:, _t, :].detach()
        else:
            _cache["mlp0_tok"] = None
        return output

    handles.append(mlp0.register_forward_hook(_cap_mlp0))

    if mode == "no_into_l1_attn":
        # 2A) Edit the input to L1 pre-attn norm (runs after residual snapshot, before attention compute).
        layer1 = model.model.layers[1]
        norm_mod, norm_name = _find_layer_attn_input_norm(layer1)
        if norm_mod is None:
            raise RuntimeError("[l0mlp-path] Could not find L1 pre-attention norm module to hook (input_layernorm/ln_1/etc).")

        def _pre_norm_hook(_module, args, _cache=cache, _t=t):
            if not args:
                return None
            x = args[0]
            vec = _cache.get("mlp0_tok", None)
            if vec is None:
                return None
            try:
                B, S, D = x.shape
            except Exception:
                return None
            if not (0 <= _t < S):
                return None
            x2 = x.clone()
            x2[:, _t, :] = x2[:, _t, :] - vec
            return (x2,)

        handles.append(norm_mod.register_forward_pre_hook(_pre_norm_hook))
        return handles

    if mode == "no_bypass_l1_attn":
        # 2B) Cancel the bypass contribution by subtracting mlp0_tok from L1 attn_out[t].
        attn1 = model.model.layers[1].self_attn

        def _attn1_hook(_module, args, output, _cache=cache, _t=t):
            vec = _cache.get("mlp0_tok", None)
            if vec is None:
                return output
            try:
                attn_out = output[0]
            except Exception:
                return output
            try:
                B, S, D = attn_out.shape
            except Exception:
                return output
            if not (0 <= _t < S):
                return output
            out2 = attn_out.clone()
            out2[:, _t, :] = out2[:, _t, :] - vec
            out_list = list(output) if isinstance(output, (tuple, list)) else [output]
            if out_list:
                out_list[0] = out2
                return tuple(out_list) if isinstance(output, tuple) else out_list
            return output

        handles.append(attn1.register_forward_hook(_attn1_hook))
        return handles

    raise ValueError(f"[l0mlp-path] Unknown mode: {mode}")

@torch.no_grad()
def run_step_cos_experiment(model, tokenized, rope_overrides, max_layer, tok_idxs, outdir):
    """
    Experiment 1 (user request):
      token -> embedding -> L0_attn -> L0_mlp -> L1_attn -> ...
    For each transition, compute cos(step_t, step_{t-1}) for selected token indices.

    We define "step vectors" as RESIDUAL STREAM states:
      embed = residual entering layer0 attention
      L_attn = residual_pre + attn_out
      L_mlp  = L_attn + mlp_out
    """
    os.makedirs(outdir, exist_ok=True)
    device = next(model.parameters()).device
    num_layers = int(model.config.num_hidden_layers)
    max_L = max(0, min(int(max_layer), num_layers - 1))
    layers = list(range(0, max_L + 1))

    # Ensure token idxs are valid for all prompts.
    min_len = min(ids.shape[1] for (ids, _bp) in tokenized)
    tok_idxs = [int(t) for t in tok_idxs if 0 <= int(t) < int(min_len)]
    if not tok_idxs:
        raise ValueError(f"[step-cos] No valid --step-cos-tok-idxs remain after min_len={int(min_len)} filter.")

    # Capture residual + (attn_out, mlp_out) for each scanned layer.
    store = {}
    handles = _capture_qkv_multi(
        model,
        layers,
        store,
        capture_attn_out=True,
        capture_mlp_out=True,
    )

    # Step names (transitions exclude the initial embed state).
    step_names = []
    for L in layers:
        step_names.append(f"L{L}_attn")
        step_names.append(f"L{L}_mlp")

    # Accumulate per-prompt cos for each transition step.
    # acc[tok_idx][step_i] where step_i aligns with step_names (i.e., current step index).
    acc = {int(t): [[] for _ in step_names] for t in tok_idxs}
    try:
        for input_ids, base_pos in tqdm(tokenized, desc="[step-cos] prompts", position=0, leave=False):
            for L in list(store.keys()):
                store[L].clear()
            pos_ids, _ = apply_perturbations(base_pos, rope_overrides=rope_overrides)
            _ = model(
                input_ids=input_ids.to(device),
                position_ids=pos_ids.to(device),
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
            )

            # Build residual-step tensors for this prompt.
            # embed state: residual entering layer0 attn
            R_prev = store[0]["residual"][0]  # [S, D]
            step_vecs = []
            for L in layers:
                entry = store[L]
                attn_out = entry.get("attn_out", None)
                mlp_out = entry.get("mlp_out", None)
                if attn_out is None or mlp_out is None:
                    raise RuntimeError(f"[step-cos] Missing attn_out/mlp_out in store for layer {L}.")
                attn_out = attn_out[0]
                mlp_out = mlp_out[0]
                R_attn = entry["residual"][0] + attn_out
                R_mlp = R_attn + mlp_out
                step_vecs.append(R_attn)
                step_vecs.append(R_mlp)

            # Compute transition cosines for selected tokens.
            for si, cur in enumerate(step_vecs):
                for t in tok_idxs:
                    prev_v = R_prev[t]
                    cur_v = cur[t]
                    cos = float(F.cosine_similarity(cur_v, prev_v, dim=0).item())
                    acc[int(t)][si].append(cos)
                R_prev = cur
    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass

    # Summarize mean/std and plot.
    report = {
        "experiment": "step_cos",
        "max_layer": int(max_L),
        "tok_idxs": [int(t) for t in tok_idxs],
        "step_names": step_names,
        "n_prompts": int(len(tokenized)),
        "min_len": int(min_len),
        "series": {},  # tok -> list[{step, mean, std}]
    }
    mean_cos_by_tok = {}
    for t in tok_idxs:
        series = []
        means = []
        for si, name in enumerate(step_names):
            vals = acc[int(t)][si]
            mu = float(np.mean(vals)) if vals else float("nan")
            sd = float(np.std(vals)) if vals else float("nan")
            series.append({"step": name, "mean": mu, "std": sd})
            means.append(mu)
        report["series"][str(int(t))] = series
        mean_cos_by_tok[int(t)] = means

    out_json = os.path.join(outdir, "step_cos_report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    _plot_step_cos(
        step_names=step_names,
        mean_cos_by_tok=mean_cos_by_tok,
        outdir=outdir,
        fname="step_cos.png",
        title=f"Stepwise cosine drift (max_layer={int(max_L)}, prompts={int(len(tokenized))})",
    )
    print(f"[step-cos] wrote {out_json}")
    print(f"[step-cos] wrote {os.path.join(outdir, 'step_cos.png')}")

def _plot_sink_attn_heads(head_means_by_layer, outdir, fname, top_heads, title, suffix=""):
    """
    Plot sink-attention (mean prob to a fixed key) across layers for multiple heads.

    head_means_by_layer: dict[layer -> np.ndarray shape [H]]
    top_heads: list[int]
    """
    if not head_means_by_layer or not top_heads:
        return
    layers = sorted(head_means_by_layer.keys())
    plt.figure(figsize=(8.5, 3.8))
    ax = plt.gca()
    for h in top_heads:
        y = [float(head_means_by_layer[L][h]) for L in layers]
        ax.plot(layers, y, marker="o", linewidth=1.4, label=f"H{h}")
    ax.set_xlabel("Layer")
    ax.set_ylabel("mean attn prob to sink key")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(min(layers), max(layers))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.set_title(title)
    ax.legend(loc="best", ncols=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(_append_suffix(os.path.join(outdir, fname), suffix), dpi=300)
    plt.close()

def _plot_sink_attn_jump_bars(jumps, outdir, fname, title, suffix="", sort=False):
    """
    Plot a bar chart of per-head jump in sink-attention between two layers, i.e.,
    jump_h = sink_attn(L) - sink_attn(L_prev) for each head h.

    jumps: np.ndarray shape [H]
    """
    if jumps is None:
        return
    jumps = np.asarray(jumps).astype(float)
    if jumps.ndim != 1 or jumps.size == 0:
        return
    H = int(jumps.size)
    idx = np.arange(H, dtype=int)
    if sort:
        order = np.argsort(jumps)[::-1]
        idx = idx[order]
        jumps = jumps[order]

    plt.figure(figsize=(10.5, 3.8))
    ax = plt.gca()
    ax.bar(np.arange(H), jumps, color="tab:blue")
    ax.axhline(0.0, color="black", linewidth=0.7)
    ax.set_xlabel("Head")
    ax.set_ylabel("jump in mean attn prob to sink key")
    ax.set_title(title)
    ax.set_xlim(-0.5, H - 0.5)
    # Show head indices as labels (rotate for readability)
    ax.set_xticks(np.arange(H))
    ax.set_xticklabels([str(int(i)) for i in idx], rotation=90, fontsize=7)
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
    p.add_argument("--max-prompts", type=int, default=None, help="Only use the first N prompts (useful for quick runs)")
    # index-shift controls (registration vs causal boundary)
    p.add_argument("--prepend-text", default=None, help="Prepend this text (tokenized without special tokens) to every prompt before analysis")
    # input controls (position/content tests)
    p.add_argument("--no-special-tokens", action="store_true", help="Tokenize without adding special tokens (BOS/EOS)")
    p.add_argument("--set-token-ids", default=None, help="Comma-separated list of pos=token_id to overwrite input_ids, eg. '0=151643,1=42'")
    p.add_argument("--swap-token-positions", type=int, nargs=2, default=None, help="Swap token ids at two positions, eg. '--swap-token-positions 0 1'")
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--head", type=int, default=0)
    # scanning mode
    p.add_argument("--scan", action="store_true")
    p.add_argument("--scan-interval", type=int, default=2, help="Interval to scan layers")
    p.add_argument("--dump-scan-stats", action="store_true", help="Dump scan_stats as JSON to outdir for programmatic comparison")
    # registration metrics (attention sink onset)
    p.add_argument("--plot-sink-attn", action="store_true", help="In --scan qkv mode, plot mean attention mass to --sink-idx across layers")
    p.add_argument("--sink-attn-head", type=int, default=None, help="Which head to use for sink-attn. Use -1 for mean over heads. Default: --head")
    p.add_argument("--sink-attn-q-start", type=int, default=4, help="Start query index (inclusive) when averaging sink attention")
    p.add_argument("--sink-attn-q-end", type=int, default=None, help="End query index (exclusive) when averaging sink attention (default: end)")
    p.add_argument("--plot-sink-attn-heads", action="store_true", help="In --scan qkv mode, plot sink-attn progression for top-k heads (by rank metric)")
    p.add_argument("--sink-attn-head-topk", type=int, default=8, help="Top-k heads to display when using --plot-sink-attn-heads")
    p.add_argument("--sink-attn-head-rank-layer", type=int, default=7, help="Layer to rank heads by (default: 7)")
    p.add_argument("--sink-attn-head-rank-metric", choices=["jump", "value"], default="jump", help="How to rank heads: jump=L-Lprev, value=at rank layer")
    p.add_argument("--plot-sink-attn-jump-bars", action="store_true", help="Plot a bar chart of per-head sink-attn jump at rank layer (default: L7-L6)")
    p.add_argument("--sink-attn-jump-sort", action="store_true", help="Sort heads by jump descending in the bar chart")
    # Experiment 2: Q/K metrics for multiple heads & layers (multi-query)
    p.add_argument("--k-metrics-headscan", action="store_true", help="Print K-metrics tables for multiple heads & layers using many query positions (q in [sink-attn-q-start, sink-attn-q-end))")
    p.add_argument("--k-metrics-heads", default=None, help="Comma-separated head indices, e.g. '6,23,5,9,22,4,20,7' (default: use --head)")
    p.add_argument("--k-metrics-layers", default="6,7", help="Comma-separated layer indices, default '6,7'")
    p.add_argument("--k-metrics-tok-idxs", default="0,1,8", help="Comma-separated token indices to compare, default '0,1,8'")
    # Experiment 3: causal ablation of sink attention in selected heads at selected layers
    p.add_argument("--ablate-sink-key", action="store_true", help="Ablate attn prob to --sink-idx for selected heads/layers (causal test)")
    p.add_argument("--ablate-layers", default="7", help="Comma-separated layer indices to ablate (default: '7')")
    p.add_argument("--ablate-heads", default="6,23,5,9,22,4,20,7", help="Comma-separated head indices to ablate (default: registration heads)")
    p.add_argument("--ablate-factor", type=float, default=0.0, help="Multiply attn prob to sink key by this factor (default: 0.0)")
    p.add_argument("--ablate-renorm", action="store_true", help="Re-normalize attention rows after ablation (recommended)")
    p.add_argument("--ablate-q-start", type=int, default=None, help="Only ablate queries q in [start,end); default: all")
    p.add_argument("--ablate-q-end", type=int, default=None, help="Only ablate queries q in [start,end); default: all")
    # Priority-1: ablate MLP_out (write step)
    p.add_argument("--ablate-mlp-out", action="store_true", help="Ablate MLP output at specified layers/token indices (tests if L6 write causes L7 registration)")
    p.add_argument("--ablate-mlp-layers", default="6", help="Comma-separated layer indices for MLP_out ablation (default: '6')")
    p.add_argument("--ablate-mlp-tok-idxs", default="0", help="Comma-separated token indices to ablate in MLP_out (default: '0')")
    p.add_argument("--ablate-mlp-factor", type=float, default=0.0, help="Multiply MLP_out by this factor at selected positions (default: 0.0)")
    # Priority-2: eval loss / perplexity under interventions
    p.add_argument("--eval-loss", action="store_true", help="Compute mean LM loss (and ppl) over prompts (supports ablations)")
    # User Exp-1: stepwise cosine drift (token representation angle changes across sublayer steps)
    p.add_argument("--step-cos", action="store_true", help="(User Exp1) Compute cos(step_t, step_{t-1}) over residual steps: embed -> L0_attn -> L0_mlp -> ...")
    p.add_argument("--step-cos-max-layer", type=int, default=6, help="Max layer L to include in step sequence (default: 6)")
    p.add_argument("--step-cos-tok-idxs", default="0,1,8", help="Comma-separated token indices to track (default: '0,1,8')")
    # User Exp-2: sweep zeroing output points (attn_out/mlp_out) before a rank layer
    p.add_argument("--zero-point-sweep", action="store_true", help="(User Exp2) Sweep zeroing one output point at a time (attn_out/mlp_out) and measure sink weakening at a rank layer")
    p.add_argument("--zero-point-max-layer", type=int, default=6, help="Only sweep layers <= this (default: 6)")
    p.add_argument("--zero-point-tok-idxs", default=None, help="Comma-separated token indices to zero at the output point (default: use --sink-idx)")
    p.add_argument("--zero-point-rank-layer", type=int, default=7, help="Read sink metric at this layer (default: 7)")
    p.add_argument("--zero-point-sort", action="store_true", help="Sort sweep bars by effect size")
    p.add_argument("--zero-point-errorbar", choices=["none", "std", "sem"], default="none", help="Add error bars to sweep bars using per-prompt delta_jump dispersion (std or sem)")
    # Path-split follow-up: isolate how L0_mlp[tok0] influences L1 (into attention vs residual bypass)
    p.add_argument("--l0mlp-path-split", action="store_true", help="Split L0 MLP(tok) influence into 'into L1 attn' vs 'bypass around L1 attn' and measure effect on jump at rank layer")
    p.add_argument("--l0mlp-path-tok-idx", type=int, default=0, help="Token index to use as 'tok0' for l0mlp path split (default: 0)")
    p.add_argument("--l0mlp-path-rank-layer", type=int, default=7, help="Rank layer L to read sink jump (L-L-1) for l0mlp path split (default: 7)")
    # Step1 mask experiments
    p.add_argument("--mask-split", type=int, default=None, help="Create a second causal boundary at m: queries q>=m cannot attend to keys < m.")
    p.add_argument("--mask-split-renorm", action="store_true", help="Renormalize probs after applying --mask-split (recommended)")
    p.add_argument("--mask-q0-allow-k", type=int, default=None, help="Override q=0 probs to uniform over keys [0..k-1] (break q=0 boundary signature).")
    p.add_argument("--mask-edit-layers", default=None, help="Comma-separated layer indices to apply mask edits (default: all layers)")
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
        if args.max_prompts is not None:
            prompts = prompts[:max(0, int(args.max_prompts))]
    else:
        if args.prompt:
            prompts = [args.prompt]
        else:
            prompts = ["To understand the failure of window attention, we find an interesting phenomenon of autoregressive LLMs: a surprisingly large amount of attention score is allocated to the initial tokens, irrespective of their relevance to the language modeling task"]

    set_token_pairs = parse_overrides(args.set_token_ids) if args.set_token_ids else None
    add_special = (not args.no_special_tokens)
    prepend_ids = None
    if args.prepend_text:
        enc = tok(args.prepend_text, return_tensors="pt", add_special_tokens=False)
        prepend_ids = enc["input_ids"]
        if prepend_ids is not None and int(prepend_ids.shape[1]) == 0:
            prepend_ids = None

    tokenized = []
    for text in prompts:
        input_ids = tokenize(tok, text, add_special_tokens=add_special)
        if prepend_ids is not None:
            input_ids = torch.cat([prepend_ids, input_ids], dim=1)
        input_ids = _apply_input_id_edits(
            input_ids,
            set_pairs=set_token_pairs,
            swap_pair=tuple(args.swap_token_positions) if args.swap_token_positions is not None else None,
        )
        base_pos = make_position_ids(input_ids.shape[1])
        tokenized.append((input_ids, base_pos))
    print(f"[Tokenized] Tokens[0, 0] is {tokenized[0][0][0, 0]}")
    print(f"[Tokenized] Tokens[0, 0] text is {tok.decode(tokenized[0][0][0, 0])}")
    if args.swap_token_positions is not None:
        a, b = int(args.swap_token_positions[0]), int(args.swap_token_positions[1])
        if 0 <= a < tokenized[0][0].shape[1] and 0 <= b < tokenized[0][0].shape[1]:
            print(f"[Tokenized] swap_token_positions={a}<->{b} now tok[{a}]='{tok.decode(tokenized[0][0][0, a])}', tok[{b}]='{tok.decode(tokenized[0][0][0, b])}'")
    if set_token_pairs:
        spec = ",".join(f"{pos}={tid}" for pos, tid in set_token_pairs)
        print(f"[Tokenized] set_token_ids={spec}")

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

    # Install mask-edit hooks (Step1 experiments)
    mask_handles = []
    if (args.mask_split is not None) or (args.mask_q0_allow_k is not None):
        if args.lower_attn or args.ablate_sink_key:
            raise ValueError("Mask-edit experiments cannot be combined with --lower-attn/--ablate-sink-key in the same run.")
        layers_edit = parse_int_list(args.mask_edit_layers) if args.mask_edit_layers else list(range(num_layers))
        print(f"[mask_edit] layers={layers_edit} split={args.mask_split} renorm={args.mask_split_renorm} q0_allow_k={args.mask_q0_allow_k}")
        mask_handles = _install_mask_edit_hooks(
            model,
            layers=layers_edit,
            split_boundary=args.mask_split,
            split_renorm=bool(args.mask_split_renorm),
            q0_allow_k=args.mask_q0_allow_k,
        )

    # If user only wants loss, run a fast evaluation and exit.
    if args.eval_loss:
        mlp_handles = []
        attn_handles = []
        if args.ablate_mlp_out:
            mlp_layers = parse_int_list(args.ablate_mlp_layers)
            mlp_tok = parse_int_list(args.ablate_mlp_tok_idxs)
            print(f"[ablate_mlp_out] layers={mlp_layers} tok_idxs={mlp_tok} factor={args.ablate_mlp_factor}")
            mlp_handles = _install_mlp_out_ablation_hooks(model, mlp_layers, mlp_tok, factor=args.ablate_mlp_factor)
        if args.ablate_sink_key:
            ablate_layers = parse_int_list(args.ablate_layers)
            ablate_heads = parse_int_list(args.ablate_heads)
            print(f"[ablate_sink_key] layers={ablate_layers} heads={ablate_heads} sink_idx={args.sink_idx} factor={args.ablate_factor} renorm={args.ablate_renorm}")
            attn_handles = _install_sink_key_ablation_hooks(
                model,
                layers=ablate_layers,
                head_idxs=ablate_heads,
                sink_k_idx=int(args.sink_idx),
                factor=float(args.ablate_factor),
                renorm=bool(args.ablate_renorm),
                q_start=args.ablate_q_start,
                q_end=args.ablate_q_end,
            )

        losses = []
        for input_ids, base_pos in tqdm(tokenized, desc="[eval-loss] prompts", position=0, leave=False):
            pos_ids, _ = apply_perturbations(base_pos, rope_overrides=rope_overrides)
            losses.append(eval_loss(model, input_ids, pos_ids))
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        ppl = float(np.exp(mean_loss)) if np.isfinite(mean_loss) else float("nan")
        report = {"mean_loss": mean_loss, "ppl": ppl, "n_prompts": len(losses)}
        out_path = os.path.join(args.outdir, "loss_report.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"[eval-loss] mean_loss={mean_loss:.6f}  ppl={ppl:.3f}  n={len(losses)}")
        print(f"[eval-loss] wrote {out_path}")

        for h in mlp_handles:
            try: h.remove()
            except Exception: pass
        for pair in attn_handles:
            for h in pair:
                try: h.remove()
                except Exception: pass
        for pair in mask_handles:
            for h in pair:
                try: h.remove()
                except Exception: pass
        return

    # -------------------------
    # User Experiment 1: stepwise cosine drift
    # -------------------------
    if args.step_cos:
        tok_idxs = parse_int_list(args.step_cos_tok_idxs)
        if not tok_idxs:
            tok_idxs = [int(args.sink_idx)]
        run_step_cos_experiment(
            model=model,
            tokenized=tokenized,
            rope_overrides=rope_overrides,
            max_layer=int(args.step_cos_max_layer),
            tok_idxs=tok_idxs,
            outdir=args.outdir,
        )
        for pair in mask_handles:
            for h in pair:
                try: h.remove()
                except Exception: pass
        return

    # -------------------------
    # Experiment 2: head-level Q/K metrics across many queries
    # -------------------------
    if args.k_metrics_headscan:
        heads = parse_int_list(args.k_metrics_heads) if args.k_metrics_heads else [int(args.head)]
        layers = parse_int_list(args.k_metrics_layers)
        tok_idxs = parse_int_list(args.k_metrics_tok_idxs)

        if not layers:
            raise ValueError("--k-metrics-layers must be non-empty")
        if not heads:
            raise ValueError("--k-metrics-heads must be non-empty (or provide --head)")
        if not tok_idxs:
            raise ValueError("--k-metrics-tok-idxs must be non-empty")

        min_len = min(ids.shape[1] for (ids, _bp) in tokenized)
        q0 = max(0, min(int(args.sink_attn_q_start), min_len))
        q1 = min_len if args.sink_attn_q_end is None else max(0, min(int(args.sink_attn_q_end), min_len))
        if q1 <= q0:
            q0, q1 = 0, min_len
        q_count = int(q1 - q0)

        for t in tok_idxs:
            if not (0 <= int(t) < min_len):
                raise ValueError(f"[k-metrics-headscan] token idx {t} out of range for min_len={min_len}")
        for L in layers:
            if not (0 <= int(L) < num_layers):
                raise ValueError(f"[k-metrics-headscan] layer {L} out of range (num_layers={num_layers})")
        for h in heads:
            if not (-1 <= int(h) < num_heads):
                raise ValueError(f"[k-metrics-headscan] head {h} out of range (num_heads={num_heads})")
        if any(int(h) == -1 for h in heads):
            raise ValueError("[k-metrics-headscan] head=-1 (mean over heads) is not supported here; provide explicit head indices")

        # accumulators: per layer -> per head -> lists
        K_vecs = {L: {h: {t: [] for t in tok_idxs} for h in heads} for L in layers}
        Q_vecs = {L: {h: [] for h in heads} for L in layers}
        COS = {L: {h: {t: [] for t in tok_idxs} for h in heads} for L in layers}

        store = {}
        handles = _capture_qkv_multi(model, layers, store)
        try:
            for input_ids, base_pos in tqdm(tokenized, desc=f"[k-metrics] Collecting Q/K (layers={layers})", position=0, leave=False):
                for L in list(store.keys()):
                    store[L].clear()
                _ = prefill(model, input_ids, base_pos)

                for L in layers:
                    qkv = store.get(L, None)
                    if not qkv:
                        continue
                    Q = qkv["q"]  # [B,H,S,D] cpu float
                    K = qkv["k"]
                    for h in heads:
                        q_mat = Q[0, int(h), q0:q1].to(dtype=torch.float32)  # [Q,D]
                        Q_vecs[L][h].append(q_mat)
                        for t in tok_idxs:
                            k_vec = K[0, int(h), int(t)].to(dtype=torch.float32)  # [D]
                            K_vecs[L][h][t].append(k_vec)
                            cos = F.cosine_similarity(q_mat, k_vec.unsqueeze(0).expand_as(q_mat), dim=1)
                            COS[L][h][t].append(cos.detach().cpu())
        finally:
            for h in handles:
                try:
                    h.remove()
                except Exception:
                    pass

        col_names = [f"tok{t}" for t in tok_idxs]
        print()
        print(f"[K metrics headscan] q_range=[{q0},{q1}) q_count={q_count} prompts={len(tokenized)} layers={layers} heads={heads}")

        for L in layers:
            for h in heads:
                XQ = torch.cat(Q_vecs[L][h], dim=0) if Q_vecs[L][h] else None  # [P*Q,D]
                if XQ is None or XQ.numel() == 0:
                    continue
                XQ_c = XQ - XQ.mean(dim=0, keepdim=True)

                spread_vals = []
                norm_vals = []
                cos_vals = []
                cos_c_vals = []
                frac_vals = []

                for t in tok_idxs:
                    Xk = torch.stack(K_vecs[L][h][t], dim=0)  # [P,D]
                    spread_vals.append(_cloud_radius(Xk))
                    norm_vals.append(_mean_vec_norm(Xk))

                    # cos(Q, K): average over prompts and query positions (paired by prompt via concatenation order)
                    cs = torch.cat(COS[L][h][t], dim=0) if COS[L][h][t] else torch.tensor([])
                    cos_vals.append(float(cs.mean().item()) if cs.numel() > 0 else 0.0)

                    # centered cosine: pair each query with its prompt's K (repeat_interleave by q_count)
                    Xk_c = Xk - Xk.mean(dim=0, keepdim=True)
                    Xk_rep_c = Xk_c.repeat_interleave(q_count, dim=0)
                    cc = F.cosine_similarity(XQ_c, Xk_rep_c, dim=1)
                    cos_c_vals.append(float(cc.mean().item()))

                    # frac of Q energy along mean(K)
                    mu = Xk.mean(dim=0)
                    frac_vals.append(_frac_energy_on_direction(XQ, mu))

                metrics = [
                    ("spread_radius", spread_vals, "sci"),
                    ("mean_norm", norm_vals, "sci"),
                    ("cos(Q, K)", cos_vals, "float"),
                    ("cos_centered(Q, K)", cos_c_vals, "float"),
                    ("frac_Q_along_mean_K", frac_vals, "float"),
                ]
                title = f"[K metrics multiQ] layer={int(L)} head={int(h)} q=[{q0},{q1})"
                _print_metrics_table(metrics, col_names, title=title)
                print()

        return

    def _run_scan_pass(scan_layers, rope_overrides_local, return_sink_series=False):
        per_layer_acc = {
            L: {"k_norm": [], "v_norm": [], "postln_norm": [], "out_norm": [], "cos": [], "sink_attn": [], "sink_attn_heads": []}
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

                        q0 = max(0, min(int(args.sink_attn_q_start), min_len))
                        q1 = min_len if args.sink_attn_q_end is None else max(0, min(int(args.sink_attn_q_end), min_len))
                        if q1 <= q0:
                            q0, q1 = 0, min_len
                        sink_head = args.head if args.sink_attn_head is None else int(args.sink_attn_head)
                        # Compute sink-attn per head (robust to lower-attn cache)
                        if hasattr(model, "_lower_attn_cache") and (L in model._lower_attn_cache):
                            attn_full = model._lower_attn_cache[L]  # already cpu float
                        else:
                            attn_full = attns[L]
                        attn_full = attn_full.detach().float().cpu()  # [B, H, Q, K]
                        sink_probs = attn_full[0, :, q0:q1, sink_idx]  # [H, Q]
                        sink_attn_heads = sink_probs.mean(dim=1).numpy()  # [H]
                        if sink_head == -1:
                            sink_attn_val = float(sink_attn_heads.mean())
                        else:
                            sink_attn_val = float(sink_attn_heads[int(sink_head)])

                        per_layer_acc[L]["k_norm"].append(k_norm)
                        per_layer_acc[L]["v_norm"].append(v_norm)
                        per_layer_acc[L]["postln_norm"].append(postln_norm)
                        per_layer_acc[L]["out_norm"].append(out_norm)
                        per_layer_acc[L]["cos"].append(series[0][1])
                        per_layer_acc[L]["sink_attn"].append(sink_attn_val)
                        if args.plot_sink_attn_heads or args.plot_sink_attn_jump_bars:
                            per_layer_acc[L]["sink_attn_heads"].append(sink_attn_heads)

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
        
        sink_series = None
        if args.print_mode == "qkv":
            stats = []
            sink_attn_heads_by_layer = {}
            if return_sink_series:
                sink_series = {int(L): [float(x) for x in per_layer_acc[L]["sink_attn"]] for L in sorted(per_layer_acc.keys())}
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

                if (args.plot_sink_attn_heads or args.plot_sink_attn_jump_bars) and acc["sink_attn_heads"]:
                    Xh = np.stack(acc["sink_attn_heads"], axis=0)  # [N, H]
                    sink_attn_heads_by_layer[L] = Xh.mean(axis=0)  # [H]

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
            return stats, scan_maps, scan_titles, sink_attn_heads_by_layer, sink_series
        else:
            for L in sorted(per_layer_maps.keys()):
                if per_layer_maps[L]:
                    head_attn = np.mean(np.stack(per_layer_maps[L], axis=0), axis=0)
                    scan_maps.append(head_attn)
                    # DEBUG
                    src = "CACHE" if (hasattr(model, "_lower_attn_cache") and L in model._lower_attn_cache) else "ATTNS"
                    scan_titles.append(f"Layer {L} Head {args.head} [{src}]")
            return [], scan_maps, scan_titles, {}, None

    # -------------------------
    # User Experiment 2: output-point zero sweep (attn_out / mlp_out) before rank layer
    # -------------------------
    if args.zero_point_sweep:
        if args.print_mode != "qkv":
            raise ValueError("--zero-point-sweep requires '--print qkv' (default).")
        if args.lower_attn or args.ablate_sink_key or args.ablate_mlp_out:
            raise ValueError("--zero-point-sweep should not be combined with other interventions (--lower-attn/--ablate-sink-key/--ablate-mlp-out).")

        rank_L = int(args.zero_point_rank_layer)
        if not (1 <= rank_L < int(num_layers)):
            raise ValueError(f"--zero-point-rank-layer must be in [1, {int(num_layers)-1}] (got {rank_L}).")

        sweep_tok = parse_int_list(args.zero_point_tok_idxs) if args.zero_point_tok_idxs else [int(args.sink_idx)]
        if not sweep_tok:
            sweep_tok = [int(args.sink_idx)]

        max_sweep_L = min(int(args.zero_point_max_layer), rank_L - 1, int(num_layers) - 1)
        sweep_layers = list(range(0, max_sweep_L + 1))

        # Scan only what we need: layers up to rank_L so we can read sink_attn(rank_L) and sink_attn(rank_L-1).
        scan_layers = list(range(0, rank_L + 1))
        print(f"[zero-point-sweep] rank_L={rank_L} sweep_layers={sweep_layers} tok_idxs={sweep_tok} scan_layers={scan_layers}")

        # Baseline
        baseline_stats, _scan_maps, _scan_titles, _heads, baseline_series = _run_scan_pass(scan_layers, rope_overrides, return_sink_series=True)
        baseline_byL = {int(s["layer"]): s for s in (baseline_stats or [])}
        baseline_L = float(baseline_byL.get(rank_L, {}).get("sink_attn", 0.0))
        baseline_Lm1 = float(baseline_byL.get(rank_L - 1, {}).get("sink_attn", 0.0))
        baseline_jump = baseline_L - baseline_Lm1
        baseline_jump_by_prompt = None
        baseline_jump_std = 0.0
        if baseline_series is not None and (rank_L in baseline_series) and ((rank_L - 1) in baseline_series):
            sL = baseline_series[int(rank_L)]
            sLm1 = baseline_series[int(rank_L - 1)]
            if len(sL) == len(sLm1) and len(sL) > 0:
                baseline_jump_by_prompt = [float(a - b) for a, b in zip(sL, sLm1)]
                if len(baseline_jump_by_prompt) > 1:
                    baseline_jump_std = float(np.std(np.array(baseline_jump_by_prompt), ddof=1))

        results = []
        points = []
        for L in sweep_layers:
            points.append(("attn", int(L)))
            points.append(("mlp", int(L)))

        for kind, L in tqdm(points, desc="[zero-point-sweep] points", position=0, leave=False):
            if kind == "attn":
                handles = _install_attn_out_ablation_hooks(model, layers=[L], tok_idxs=sweep_tok, factor=0.0)
            else:
                handles = _install_mlp_out_ablation_hooks(model, layers=[L], tok_idxs=sweep_tok, factor=0.0)
            try:
                stats, _m, _t, _h, series = _run_scan_pass(scan_layers, rope_overrides, return_sink_series=True)
            finally:
                for hh in handles:
                    try:
                        hh.remove()
                    except Exception:
                        pass

            byL = {int(s["layer"]): s for s in (stats or [])}
            ab_L = float(byL.get(rank_L, {}).get("sink_attn", 0.0))
            ab_Lm1 = float(byL.get(rank_L - 1, {}).get("sink_attn", 0.0))
            ab_jump = ab_L - ab_Lm1
            delta_by_prompt = None
            delta_std = 0.0
            delta_sem = 0.0
            if baseline_jump_by_prompt is not None and series is not None and (rank_L in series) and ((rank_L - 1) in series):
                sL = series[int(rank_L)]
                sLm1 = series[int(rank_L - 1)]
                if len(sL) == len(sLm1) and len(sL) == len(baseline_jump_by_prompt) and len(sL) > 0:
                    ab_jump_by_prompt = [float(a - b) for a, b in zip(sL, sLm1)]
                    delta_by_prompt = [float(bj - aj) for bj, aj in zip(baseline_jump_by_prompt, ab_jump_by_prompt)]
                    if len(delta_by_prompt) > 1:
                        delta_std = float(np.std(np.array(delta_by_prompt), ddof=1))
                        delta_sem = float(delta_std / math.sqrt(len(delta_by_prompt)))

            tok_str = ",".join(str(int(t)) for t in sweep_tok)
            point = f"L{int(L)}_{kind}[tok={tok_str}]"
            results.append({
                "point": point,
                "layer": int(L),
                "kind": kind,
                "tok_idxs": [int(t) for t in sweep_tok],
                "rank_layer": int(rank_L),
                "baseline_L": baseline_L,
                "baseline_Lm1": baseline_Lm1,
                "baseline_jump": baseline_jump,
                "ablated_L": ab_L,
                "ablated_Lm1": ab_Lm1,
                "ablated_jump": ab_jump,
                "delta_jump": float(baseline_jump - ab_jump),
                "delta_jump_std": float(delta_std),
                "delta_jump_sem": float(delta_sem),
                "delta_jump_n": int(0 if delta_by_prompt is None else len(delta_by_prompt)),
            })

        report = {
            "experiment": "zero_point_sweep",
            "rank_layer": int(rank_L),
            "scan_layers": [int(L) for L in scan_layers],
            "sweep_layers": [int(L) for L in sweep_layers],
            "tok_idxs": [int(t) for t in sweep_tok],
            "n_prompts": int(len(tokenized)),
            "sink_idx": int(args.sink_idx),
            "sink_attn_head": int(args.head if args.sink_attn_head is None else args.sink_attn_head),
            "sink_attn_q_start": int(args.sink_attn_q_start),
            "sink_attn_q_end": None if args.sink_attn_q_end is None else int(args.sink_attn_q_end),
            "baseline": {
                "sink_attn_L": baseline_L,
                "sink_attn_Lm1": baseline_Lm1,
                "jump": baseline_jump,
                "jump_std": float(baseline_jump_std),
            },
            "results": results,
        }
        os.makedirs(args.outdir, exist_ok=True)
        out_json = os.path.join(args.outdir, "zero_point_sweep_report.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        err_mode = str(args.zero_point_errorbar)
        yerr = None
        yerr_label = None
        if err_mode in ("std", "sem"):
            key = "delta_jump_std" if err_mode == "std" else "delta_jump_sem"
            # yerr must align with sorting used in the plot, so recompute after sorting in _plot_zero_point_sweep.
            # We pass yerr=None here and instead attach a pre-sorted list below.
            rows_plot = results[:]
            if bool(args.zero_point_sort):
                rows_plot = sorted(rows_plot, key=lambda r: float(r.get("delta_jump", 0.0)), reverse=True)
            yerr = [float(r.get(key, 0.0)) for r in rows_plot]
            yerr_label = err_mode.upper()
        _plot_zero_point_sweep(
            results=results,
            outdir=args.outdir,
            fname="zero_point_sweep_delta_jump.png",
            title=f"Zero-point sweep: delta jump at L{rank_L} (prompts={len(tokenized)}, q>={int(args.sink_attn_q_start)}, head={int(args.head if args.sink_attn_head is None else args.sink_attn_head)})",
            sort=bool(args.zero_point_sort),
            yerr=yerr,
            yerr_label=yerr_label,
        )
        print(f"[zero-point-sweep] baseline jump(L{rank_L}-L{rank_L-1})={baseline_jump:.6f}")
        print(f"[zero-point-sweep] wrote {out_json}")
        print(f"[zero-point-sweep] wrote {os.path.join(args.outdir, 'zero_point_sweep_delta_jump.png')}")

        for pair in mask_handles:
            for h in pair:
                try: h.remove()
                except Exception: pass
        return

    # -------------------------
    # Follow-up: isolate L0_mlp(tok0) route into L1 (into attention vs residual bypass)
    # -------------------------
    if args.l0mlp_path_split:
        if args.print_mode != "qkv":
            raise ValueError("--l0mlp-path-split requires '--print qkv' (default).")
        if args.lower_attn or args.ablate_sink_key or args.ablate_mlp_out or args.zero_point_sweep:
            raise ValueError("--l0mlp-path-split should not be combined with other interventions in the same run.")

        rank_L = int(args.l0mlp_path_rank_layer)
        if not (1 <= rank_L < int(num_layers)):
            raise ValueError(f"--l0mlp-path-rank-layer must be in [1, {int(num_layers)-1}] (got {rank_L}).")
        if int(num_layers) < 2:
            raise ValueError("--l0mlp-path-split requires at least 2 layers.")
        tok0 = int(args.l0mlp_path_tok_idx)

        scan_layers = list(range(0, rank_L + 1))
        print(f"[l0mlp-path] tok={tok0} rank_L={rank_L} scan_layers={scan_layers}")

        # Baseline
        baseline_stats, _m, _t, _h, baseline_series = _run_scan_pass(scan_layers, rope_overrides, return_sink_series=True)
        baseline_byL = {int(s["layer"]): s for s in (baseline_stats or [])}
        baseline_L = float(baseline_byL.get(rank_L, {}).get("sink_attn", 0.0))
        baseline_Lm1 = float(baseline_byL.get(rank_L - 1, {}).get("sink_attn", 0.0))
        baseline_jump = baseline_L - baseline_Lm1

        baseline_jump_by_prompt = None
        if baseline_series is not None and (rank_L in baseline_series) and ((rank_L - 1) in baseline_series):
            sL = baseline_series[int(rank_L)]
            sLm1 = baseline_series[int(rank_L - 1)]
            if len(sL) == len(sLm1) and len(sL) > 0:
                baseline_jump_by_prompt = [float(a - b) for a, b in zip(sL, sLm1)]

        jobs = [
            ("no_into_l1_attn", "no_into_l1_attn"),
            ("no_bypass_l1_attn", "no_bypass_l1_attn"),
        ]
        results = []
        for label, mode in jobs:
            handles = _install_l0mlp_path_split_hooks(model, tok_idx=tok0, mode=mode)
            try:
                stats, _m2, _t2, _h2, series = _run_scan_pass(scan_layers, rope_overrides, return_sink_series=True)
            finally:
                for hh in handles:
                    try:
                        hh.remove()
                    except Exception:
                        pass

            byL = {int(s["layer"]): s for s in (stats or [])}
            ab_L = float(byL.get(rank_L, {}).get("sink_attn", 0.0))
            ab_Lm1 = float(byL.get(rank_L - 1, {}).get("sink_attn", 0.0))
            ab_jump = ab_L - ab_Lm1

            delta_std = 0.0
            delta_sem = 0.0
            if baseline_jump_by_prompt is not None and series is not None and (rank_L in series) and ((rank_L - 1) in series):
                sL = series[int(rank_L)]
                sLm1 = series[int(rank_L - 1)]
                if len(sL) == len(sLm1) and len(sL) == len(baseline_jump_by_prompt) and len(sL) > 1:
                    ab_jump_by_prompt = [float(a - b) for a, b in zip(sL, sLm1)]
                    delta_by_prompt = [float(bj - aj) for bj, aj in zip(baseline_jump_by_prompt, ab_jump_by_prompt)]
                    delta_std = float(np.std(np.array(delta_by_prompt), ddof=1))
                    delta_sem = float(delta_std / math.sqrt(len(delta_by_prompt)))

            results.append({
                "point": f"{label}[tok={tok0}]",
                "layer": 0,
                "kind": "path",
                "tok_idxs": [tok0],
                "rank_layer": int(rank_L),
                "baseline_L": baseline_L,
                "baseline_Lm1": baseline_Lm1,
                "baseline_jump": baseline_jump,
                "ablated_L": ab_L,
                "ablated_Lm1": ab_Lm1,
                "ablated_jump": ab_jump,
                "delta_jump": float(baseline_jump - ab_jump),
                "delta_jump_std": float(delta_std),
                "delta_jump_sem": float(delta_sem),
                "delta_jump_n": int(0 if baseline_jump_by_prompt is None else len(baseline_jump_by_prompt)),
            })

        report = {
            "experiment": "l0mlp_path_split",
            "tok_idx": int(tok0),
            "rank_layer": int(rank_L),
            "scan_layers": [int(L) for L in scan_layers],
            "n_prompts": int(len(tokenized)),
            "sink_idx": int(args.sink_idx),
            "sink_attn_head": int(args.head if args.sink_attn_head is None else args.sink_attn_head),
            "sink_attn_q_start": int(args.sink_attn_q_start),
            "sink_attn_q_end": None if args.sink_attn_q_end is None else int(args.sink_attn_q_end),
            "baseline": {
                "sink_attn_L": baseline_L,
                "sink_attn_Lm1": baseline_Lm1,
                "jump": baseline_jump,
            },
            "results": results,
        }
        os.makedirs(args.outdir, exist_ok=True)
        out_json = os.path.join(args.outdir, "l0mlp_path_split_report.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        # Plot with SEM error bars (most interpretable).
        yerr = [float(r.get("delta_jump_sem", 0.0)) for r in results]
        _plot_zero_point_sweep(
            results=results,
            outdir=args.outdir,
            fname="l0mlp_path_split_delta_jump.png",
            title=f"L0 MLP path-split (tok={tok0}): delta jump at L{rank_L} (prompts={len(tokenized)}, q>={int(args.sink_attn_q_start)}, head={int(args.head if args.sink_attn_head is None else args.sink_attn_head)})",
            sort=False,
            yerr=yerr,
            yerr_label="SEM",
        )
        print(f"[l0mlp-path] baseline jump(L{rank_L}-L{rank_L-1})={baseline_jump:.6f}")
        print(f"[l0mlp-path] wrote {out_json}")
        print(f"[l0mlp-path] wrote {os.path.join(args.outdir, 'l0mlp_path_split_delta_jump.png')}")
        return

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

        ablate_handles = []
        if args.ablate_sink_key:
            if args.lower_attn:
                raise ValueError("--ablate-sink-key cannot be combined with --lower-attn in the same run.")
            ablate_layers = parse_int_list(args.ablate_layers)
            ablate_heads = parse_int_list(args.ablate_heads)
            print(f"[ablate_sink_key] layers={ablate_layers} heads={ablate_heads} sink_idx={args.sink_idx} factor={args.ablate_factor} renorm={args.ablate_renorm}")
            ablate_handles = _install_sink_key_ablation_hooks(
                model,
                layers=ablate_layers,
                head_idxs=ablate_heads,
                sink_k_idx=int(args.sink_idx),
                factor=float(args.ablate_factor),
                renorm=bool(args.ablate_renorm),
                q_start=args.ablate_q_start,
                q_end=args.ablate_q_end,
            )

        mlp_ablate_handles = []
        if args.ablate_mlp_out:
            mlp_layers = parse_int_list(args.ablate_mlp_layers)
            mlp_tok = parse_int_list(args.ablate_mlp_tok_idxs)
            print(f"[ablate_mlp_out] layers={mlp_layers} tok_idxs={mlp_tok} factor={args.ablate_mlp_factor}")
            mlp_ablate_handles = _install_mlp_out_ablation_hooks(model, mlp_layers, mlp_tok, factor=args.ablate_mlp_factor)

        scan_stats = [] # aggregated per layer
        scan_maps, scan_titles = [], []
        summary = None

        if args.scan:
            # perturbed pass
            scan_layers = list(range(0, num_layers, args.scan_interval))
            scan_stats, scan_maps, scan_titles, sink_attn_heads_by_layer, _series = _run_scan_pass(scan_layers, rope_overrides, return_sink_series=False)

            if args.print_mode == "qkv" and len(scan_stats) > 0:
                scan_stats = sorted(scan_stats, key=lambda s: s["layer"])
                if args.dump_scan_stats:
                    out_path = _append_suffix(os.path.join(args.outdir, "scan_stats.json"), cur_suffix)
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(scan_stats, f, indent=2)
                    print(f"[dump] wrote scan_stats -> {out_path}")
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

            # registration metric: attention sink onset curve
            if args.print_mode == "qkv" and args.plot_sink_attn and len(scan_stats) > 0:
                _plot_progression(
                    scan_stats, args.outdir, key="sink_attn",
                    ylabel=f"mean attn prob to K[{args.sink_idx}]",
                    title=f"Sink-attention progression (sink_idx={args.sink_idx}, head={args.sink_attn_head if args.sink_attn_head is not None else args.head})",
                    fname="scan_sink_attn.png", suffix=cur_suffix,
                    mode="cos",
                )

            # registration metric: head-level decomposition
            if args.print_mode == "qkv" and (args.plot_sink_attn_heads or args.plot_sink_attn_jump_bars) and sink_attn_heads_by_layer:
                layers_avail = sorted(sink_attn_heads_by_layer.keys())
                rank_layer = int(args.sink_attn_head_rank_layer)
                if rank_layer not in sink_attn_heads_by_layer:
                    # pick closest available layer
                    rank_layer = min(layers_avail, key=lambda L: abs(L - rank_layer))
                prev_layer = max(layers_avail[0], rank_layer - 1)
                if prev_layer not in sink_attn_heads_by_layer and layers_avail:
                    prev_layer = layers_avail[max(0, layers_avail.index(rank_layer) - 1)]

                v_rank = sink_attn_heads_by_layer[rank_layer]
                if args.sink_attn_head_rank_metric == "jump" and prev_layer in sink_attn_heads_by_layer:
                    v_prev = sink_attn_heads_by_layer[prev_layer]
                    scores = v_rank - v_prev
                    metric_str = f"jump(L{rank_layer}-L{prev_layer})"
                else:
                    scores = v_rank
                    metric_str = f"value@L{rank_layer}"

                # Always print ranking (useful for debugging)
                topk = int(max(1, min(int(args.sink_attn_head_topk), scores.shape[0])))
                order = np.argsort(scores)[::-1]
                top_heads = [int(i) for i in order[:topk]]
                print(f"[sink_attn heads] rank_metric={metric_str} top{topk}={top_heads}")
                for h in top_heads:
                    print(f"  head {h:>2d}: score={float(scores[h]):.4f}  value@L{rank_layer}={float(v_rank[h]):.4f}")

                if args.plot_sink_attn_heads:
                    _plot_sink_attn_heads(
                        sink_attn_heads_by_layer,
                        args.outdir,
                        fname="scan_sink_attn_heads_topk.png",
                        top_heads=top_heads,
                        title=f"Sink-attn by head (sink_idx={args.sink_idx}, top{topk} by {metric_str})",
                        suffix=cur_suffix,
                    )

                if args.plot_sink_attn_jump_bars and args.sink_attn_head_rank_metric == "jump":
                    _plot_sink_attn_jump_bars(
                        scores,
                        args.outdir,
                        fname=f"scan_sink_attn_jump_bars_L{rank_layer}-L{prev_layer}.png",
                        title=f"Per-head sink-attn jump (sink_idx={args.sink_idx}, {metric_str})",
                        suffix=cur_suffix,
                        sort=bool(args.sink_attn_jump_sort),
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

                k = 3
                k_fracs = (0.25, 0.75, 0.99)

                col_names = ["tok0", "tok1", "tok8"]

                Z0 = mlp_data["Z"][0]; Z1 = mlp_data["Z"][1]; Z8 = mlp_data["Z"][8] 
                G0 = mlp_data["G"][0]; G1 = mlp_data["G"][1]; G8 = mlp_data["G"][8]
                U0 = mlp_data["U"][0]; U1 = mlp_data["U"][1]; U8 = mlp_data["U"][8]
                A0 = mlp_data["A"][0]; A1 = mlp_data["A"][1]; A8 = mlp_data["A"][8]
                X0 = mlp_data["X"][0]; X1 = mlp_data["X"][1]; X8 = mlp_data["X"][8]
                AttnIn0 = mlp_data["Attn_in"][0]
                AttnOut0 = mlp_data["Attn_out"][0]

                z0_topk_act_idx = _topk_activated_list_abs(Z0, k)
                w_ref = F.normalize(Wd[:, z0_topk_act_idx[0]], dim=0)
                wdown_cos_to_top1 = []
                for i in z0_topk_act_idx:
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
                    ("inter_z_k@frac_energy [.25, .75, .99]", z_kfrac_vals, ""),
                    (f"inter_z_topk_act_idx k={k}", z_topk_idx_vals, ""),
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

                probe_idx = [5723, 8518, 422]
                probe_cols = [f"idx_{i}" for i in probe_idx]
                norms_all, ranks_all = _row_norm_rank_desc(Wg)
                cos_tok0, cos_tok1, cos_tok8 = [], [], []
                w_norms, w_ranks = [], []
                cos_to_top1 = []
                frac_e_tok0, frac_e_tok1, frac_e_tok8 = [], [], []
                cos_attnin_tok0, cos_attnout_tok0 = [], []

                w_ref = F.normalize(Wg[probe_idx[0], :].to(dtype=torch.float32), dim=0)

                for ii in probe_idx:
                    wi = Wg[ii, :].to(dtype=torch.float32)
                    cos_tok0.append(_mean_cos_row_vs_X(wi, X0))
                    cos_tok1.append(_mean_cos_row_vs_X(wi, X1))
                    cos_tok8.append(_mean_cos_row_vs_X(wi, X8))
                    cos_attnin_tok0.append(_mean_cos_row_vs_X(wi, AttnIn0))
                    cos_attnout_tok0.append(_mean_cos_row_vs_X(wi, AttnOut0))
                    frac_e_tok0.append(_mean_frac_energy_on_dir(X0, wi))
                    frac_e_tok1.append(_mean_frac_energy_on_dir(X1, wi))
                    frac_e_tok8.append(_mean_frac_energy_on_dir(X8, wi))
                    w_norms.append(float(norms_all[ii].item()))
                    w_ranks.append(str(int(ranks_all[ii].item())))
                    
                    wi_normed = F.normalize(wi, dim=0)
                    cos_to_top1.append(float(torch.dot(wi_normed, w_ref).item()))

                metrics_g_vecs = [
                    ("cos(w_i, X_tok0)", cos_tok0, "float"),
                    ("cos(w_i, X_tok1)", cos_tok1, "float"),
                    ("cos(w_i, X_tok8)", cos_tok8, "float"),
                    ("cos(w_i, AttnIn_tok0)", cos_attnin_tok0, "float"),
                    ("cos(w_i, AttnOut_tok0)", cos_attnout_tok0, "float"),
                    ("frac_energy(X_tok0 -> w_i)", frac_e_tok0, "float"),
                    ("frac_energy(X_tok1 -> w_i)", frac_e_tok1, "float"),
                    ("frac_energy(X_tok8 -> w_i)", frac_e_tok8, "float"),
                    ("||w_i||", w_norms, "sci"),
                    ("rank(||w||) (1=largest)", w_ranks, ""),
                    ("cos(w_i, w_top1)", cos_to_top1, "float"),
                ]
                print()
                _print_metrics_table(metrics_g_vecs, probe_cols, title=f"[MLP gate row vectors] layer={L}")

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

        for pair in ablate_handles:
            for h in pair:
                try:
                    h.remove()
                except Exception:
                    pass

        for h in mlp_ablate_handles:
            try:
                h.remove()
            except Exception:
                pass

    for pair in mask_handles:
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