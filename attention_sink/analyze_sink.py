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

def _install_lower_attn_hooks(model, layers, factor=0.5, sink_k_idx=0, only_return_sink_value=False):
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

def _forward_attn_and_qkv(
    model, input_ids, position_ids, layer, need_qkv=False, 
    collect_row=False, row_head=0, row_qidx=None
):
    cache = {}
    if need_qkv:
        _capture_qkv(model, layer, cache)
    attns = prefill(model, input_ids, position_ids)
    hook = cache.pop("_hook", None)
    if hook is not None:
        hook.remove()

    q, k, v = cache.get("q", None), cache.get("k", None), cache.get("v", None)
    
    row_logits = None
    row_probs = None
    if collect_row and q is not None and k is not None and row_qidx is not None:
        head_dim = model.config.head_dim
        logits = torch.matmul(q[:, row_head], k[:, row_head].transpose(2, 1)) / math.sqrt(head_dim)
        row_logits = logits[:, row_qidx, :].detach().float().cpu() 
        probs = attns[layer][:, row_head]
        row_probs = probs[:, row_qidx, :].detach().float().cpu()
    
    return attns, q, k, v, row_logits, row_probs

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

        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if num_kv != num_heads:
            k = repeat_kv(k, num_groups)
            v = repeat_kv(v, num_groups)
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

def _plot_progression(stats_a, outdir, key, ylabel, title, fname, suffix="", stats_b=None, is_cos=False):
    layers = sorted({s["layer"] for s in stats_a})
    by_a = {s["layer"]: s for s in stats_a}
    y_a = [by_a[L][key] for L in layers]
    plt.figure(figsize=(7, 3.5))
    plt.plot(layers, y_a, marker="o", label="Perturbed" if stats_b else None)

    if stats_b:
        by_b = {s["layer"]: s for s in stats_b}
        y_b = [by_b[L][key] for L in layers]
        plt.plot(layers, y_b, marker="s", label="Baseline")

    plt.xlabel("Layer")
    plt.ylabel(ylabel)
    if len(y_a) > 0:
        plt.axhline(y_a[-1], color="gray", linestyle="--", linewidth=1.0)
    if is_cos:
        plt.ylim(0, 1)
    else:
        plt.ylim(0, 100)

    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(4))
    plt.title(title)
    if stats_b:
        plt.legend()
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
    # output
    p.add_argument("--outdir", default="results")
    args = p.parse_args()

    _disable_packed_sequence_splitting()
    tok, model = load_model(args.model, args.device, args.dtype, random_init=args.random_init)

    if args.compare and (args.print_mode == "heatmap" or args.lower_attn):
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

    rope_str = args.rope
    rope_overrides = parse_overrides(rope_str) if rope_str else None
    pert_suffix = _pert_suffix(args)

    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    job_summary_stats = []

    def _aggregate_qkv_for_layer(layer, head):
        cos_list = []
        k_norms, v_norms = [], []
        attn_maps = []
        row_logits_list, row_probs_list = [], []
        min_len = min(ids.shape[1] for (ids, _bp) in tokenized)
        last_q = min_len - 1
        if args.target_idx is None:
            target_q = last_q
        else:
            target_q = max(0, min(last_q, args.target_idx))
        sink_idx = max(0, min(last_q, args.sink_idx)) 
        sink_attn_values = []
        
        for input_ids, base_pos in tqdm(tokenized, desc="Aggregating QKV", position=1, leave=False):
            pos_ids_perturbed, _ = apply_perturbations(base_pos, rope_overrides=rope_overrides)
            attns, q, k, v, row_logits, row_probs = _forward_attn_and_qkv(
                model, input_ids, pos_ids_perturbed, layer, need_qkv=True,
                collect_row=True, row_head=head, row_qidx=target_q
            )
            if q is None or k is None:
                raise RuntimeError("Failed to capture Q/K.")
            series, k_norm = compute_cosine_series(q, k, head, k_sink_idx=sink_idx, q_positions=[target_q])
            v_norm = float(v[0, head, sink_idx].norm().item())

            k_norms.append(k_norm)
            v_norms.append(v_norm)
            cos_list.append(series[0][1])
            if hasattr(model, "_lower_attn_cache") and layer in model._lower_attn_cache:
                hm = pick_head(model._lower_attn_cache, layer, head)
            else:
                hm = pick_head(attns, layer, head)
            attn_maps.append(hm[:min_len, :min_len])

            sink_slice = hm[4:, sink_idx] # remove first token because attention is 1.0
            sink_attn_values.append(float(sink_slice.mean().item())) # average across queries, assuming same sink

            truncate_len = min_len
            if args.target_idx is not None:
                truncate_len = max(1, min(min_len, target_q + 1))
            if row_logits is not None:
                row_logits_np = row_logits.squeeze(0).numpy()[:truncate_len]
                row_logits_list.append(row_logits_np)
            if row_probs is not None:
                row_probs_np = row_probs.squeeze(0).numpy()[:truncate_len]
                row_probs_list.append(row_probs_np)

        cos_mean = float(np.mean(np.array(cos_list)))
        k_mean = float(np.mean(k_norms))
        v_mean = float(np.mean(v_norms))
        row_logits_mean = np.mean(np.stack(row_logits_list, axis=0), axis=0) if row_logits_list else None
        row_probs_mean = np.mean(np.stack(row_probs_list, axis=0), axis=0) if row_probs_list else None
        sink_mean = float(np.mean(np.array(sink_attn_values))) if sink_attn_values else None
        return {
            "k_norm": k_mean, "v_norm": v_mean, "cos": cos_mean, 
            "layer": layer, "head": head,
            "row_logits": row_logits_mean, "row_probs": row_probs_mean,
            "target_q": target_q, "sink_attn": sink_mean
        }

    def _aggregate_attn_score_for_layer(layer, head, tag=""):
        attn_maps = []
        min_len = min(ids.shape[1] for (ids, _bp) in tokenized)
        sink_attn_values = []
        sink_idx = max(0, min(min_len - 1, args.sink_idx)) 
        for input_ids, base_pos in tqdm(tokenized, desc="Aggregating attention scores", position=1, leave=False):
            pos_ids_perturbed, _ = apply_perturbations(base_pos, rope_overrides=rope_overrides)
            attns, _, _, _, _, _ = _forward_attn_and_qkv(model, input_ids, pos_ids_perturbed, layer, need_qkv=False)
            if hasattr(model, "_lower_attn_cache") and layer in model._lower_attn_cache:
                hm = pick_head(model._lower_attn_cache, layer, head)
            else:
                hm = pick_head(attns, layer, head)
            attn_maps.append(hm[:min_len, :min_len])

            sink_slice = hm[4:, sink_idx]
            sink_attn_values.append(float(sink_slice.mean().item()))

        head_attn = np.mean(np.stack(attn_maps, axis=0), axis=0)
        sink_mean = float(np.mean(np.array(sink_attn_values))) if sink_attn_values else None
        result = {"layer": layer, "head": head, "sink_attn": sink_mean, "heatmap": None, "title": None}
        if args.scan:
            m, t = _plot_heatmap_with_tag(head_attn, layer, head, args.outdir, tag=tag, return_map=True, suffix=pert_suffix)
            result["heatmap"] = m
            result["title"] = t
        else:
            _plot_heatmap_with_tag(head_attn, layer, head, args.outdir, tag=tag, return_map=False, suffix=pert_suffix)
        return result

    def _log_row_summary(prefix, stats_list):
        for stat in stats_list:
            logits = stat.get("row_logits")
            probs = stat.get("row_probs")
            if logits is None or probs is None:
                continue
            layer = stat["layer"]
            head = stat["head"]
            target_q = stat["target_q"]
            logits_list = np.round(np.asarray(logits), 2).tolist()
            probs_list = np.round(np.asarray(probs), 2).tolist()
            print(f"{prefix}[layer {layer} head {head}] mean logits[q={target_q}]: {logits_list}")
            print(f"{prefix}[layer {layer} head {head}] mean probs[q={target_q}]: {probs_list}")

    def _agg_baseline(layer, head):
        nonlocal rope_overrides
        tmp = rope_overrides
        rope_overrides = None
        try:
            return _aggregate_qkv_for_layer(layer, head)
        finally:
            rope_overrides = tmp

    stop_jobs = [None]
    if args.lower_attn:
        if args.stop_layers and args.only_stop_layer:
            raise ValueError("--stop-layers and --only-stop-layer cannot be used together.")
        if args.only_stop_layer:
            stop_jobs = [args.only_stop_layer]
        elif args.stop_layers:
            b, e = args.stop_layers
            b = max(0, int(b))
            e = min(num_layers, int(e))
            stop_jobs = list(range(b, e))
        else:
            stop_jobs = [num_layers - 1]

    for _job_stop in stop_jobs:
        args._current_stop_layer = _job_stop
        pert_suffix = _pert_suffix(args)

        lower_attn_handles = []
        if args.lower_attn:
            if args.only_stop_layer is not None:
                target_layers = [int(L) for L in args.only_stop_layer]
            else:
                target_layers = [L for L in range(num_layers) if L <= _job_stop]
            if target_layers:
                lower_attn_handles = _install_lower_attn_hooks(
                    model, layers=target_layers, factor=args.lower_factor, sink_k_idx=args.sink_idx,
                    only_return_sink_value=args.only_return_sink_value
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
            for L in tqdm(scan_layers, desc="Scanning layers", position=0, leave=False):
                if args.print_mode == "qkv":
                    stats = _aggregate_qkv_for_layer(L, args.head)                
                    scan_stats.append(stats)
                    if stats["layer"] == num_layers - 1 and summary is None:
                        summary = stats.get("sink_attn")
                else:
                    tag = ""
                    if rope_overrides:
                        ov_str = ",".join([f"{ti}={rp}" for ti, rp in rope_overrides])
                        tag = f"PERTURBED rope({ov_str}) "
                    stats = _aggregate_attn_score_for_layer(L, args.head, tag)
                    if stats["heatmap"] is not None and stats["title"] is not None:
                        scan_maps.append(stats["heatmap"])
                        scan_titles.append(stats["title"])
                    if stats["layer"] == num_layers - 1 and summary is None:
                        summary = stats.get("sink_attn")
            
            # baseline pass
            if args.compare and args.print_mode == "qkv":
                for L in tqdm(scan_layers, desc="Scanning baseline layers", position=0, leave=False):
                    scan_stats_base.append(_agg_baseline(L, args.head))

            if args.print_mode == "qkv" and len(scan_stats) > 0:
                scan_stats = sorted(scan_stats, key=lambda s: s["layer"])
                q_label = f"Q[{args.target_idx}]" if args.target_idx is not None else "Q[last]"
                if args.compare and len(scan_stats_base) > 0:
                    scan_stats_base = sorted(scan_stats_base, key=lambda s: s["layer"])
                    _plot_progression(
                        scan_stats, args.outdir, key="k_norm", ylabel=f"||K[{args.sink_idx}]||",
                        title=f"K-norm progression (token={args.sink_idx}) (compare)", fname="scan_knorm_compare.png", suffix=pert_suffix,
                        stats_b=scan_stats_base
                    )
                    _plot_progression(
                        scan_stats, args.outdir, key="v_norm", ylabel=f"||V[{args.sink_idx}]||",
                        title=f"V-norm progression (token={args.sink_idx}) (compare)", fname="scan_vnorm_compare.png", suffix=pert_suffix,
                        stats_b=scan_stats_base
                    )
                    _plot_progression(
                        scan_stats, args.outdir, key="cos", ylabel=f"cos({q_label}, K[{args.sink_idx}])",
                        title=f"Cosine to K[{args.sink_idx}] across layers (compare)", fname="scan_cosine_compare.png", suffix=pert_suffix,
                        stats_b=scan_stats_base, is_cos=True
                    )
                    # _log_row_summary("[Perturbed] ", scan_stats)
                    # _log_row_summary("[Baseline] ", scan_stats_base)
                else:
                    _plot_progression(
                        scan_stats, args.outdir, key="k_norm", ylabel=f"||K[{args.sink_idx}]||",
                        title=f"K-norm progression (token={args.sink_idx})", fname="scan_knorm.png", suffix=pert_suffix
                    )
                    _plot_progression(
                        scan_stats, args.outdir, key="v_norm", ylabel=f"||V[{args.sink_idx}]||",
                        title=f"V-norm progression (token={args.sink_idx})", fname="scan_vnorm.png", suffix=pert_suffix
                    )
                    _plot_progression(
                        scan_stats, args.outdir, key="cos", ylabel=f"cos({q_label}, K[{args.sink_idx}])",
                        title=f"Cosine to K[{args.sink_idx}] across layers", fname="scan_cosine.png", suffix=pert_suffix, is_cos=True
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
                if args.lower_attn:
                    segment = "lower_attn"
                    if args.only_stop_layer is not None:
                        segment += f" only_stop[{args.only_stop_layer}]"
                    elif args._current_stop_layer is not None:
                        segment += f" stop[{args._current_stop_layer}]"
                    grid_title += f" - {segment} (x{args.lower_factor})"

                _plot_heatmap_grid(
                    scan_maps, scan_titles, os.path.join(args.outdir, "scan_heatmaps.png"), rows=4,
                    suffix=pert_suffix, suptitle=grid_title,
                )
            
            if summary is not None:
                print(f"[Summary] sink attention at layer {num_layers - 1}: {summary:.4f} | stop at {args._current_stop_layer}")
                if args.lower_attn and args._current_stop_layer is not None:
                    job_summary_stats.append({"layer": int(args._current_stop_layer), "sink_attn": float(summary)})

        else:
            if args.print_mode == "qkv":
                stats = _aggregate_attn_score_for_layer(args.layer, args.head)
                stats_list = [stats]
                baseline_stats = []
                if args.compare:
                    baseline_stats = [_agg_baseline(args.layer, args.head)]
                q_label = f"Q[{args.target_idx}]" if args.target_idx is not None else "Q[last]"
                # _log_row_summary(f"[Perturbed] ", stats_list)
                if baseline_stats:
                    pass
                    # _log_row_summary(f"[Baseline] ", baseline_stats)
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

    if args.lower_attn and len(job_summary_stats) > 0:
        job_summary_stats = sorted(job_summary_stats, key=lambda s: s["layer"])
        prev_stop = getattr(args, "_current_stop_layer", None)
        args._current_stop_layer = None
        agg_suffix = _pert_suffix(args)
        args._current_stop_layer = prev_stop
        _plot_progression(
            job_summary_stats, args.outdir, key="sink_attn",
            ylabel="Sink attention score", title="Sink attention score progression for the last query token",
            fname="scan_sink_attn.png", suffix=agg_suffix, is_cos=True
        )

if __name__ == "__main__":
    main()