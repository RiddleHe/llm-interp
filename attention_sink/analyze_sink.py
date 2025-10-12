import argparse, os, numpy as np, torch, matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F

EPS = 1e-8
PRINT_CHOICES = ["heatmap", "log"]

def load_model(model_name, device, dtype):
    torch_dtype = {"auto": "auto", "bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dmap = "auto" if device == "auto" else None

    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch_dtype[dtype], device_map=dmap, attn_implementation="eager")
    model.eval()
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    return tok, model

def tokenize(tok, text):
    enc = tok(text, return_tensors="pt", add_special_tokens=True)
    return enc["input_ids"]

def _parse_qpos(arg, seq_len):
    if arg:
        return [max(0, min(seq_len - 1, int(x))) for x in arg.split(",")]
    last = seq_len - 1
    two_thirds = max(0, (2 * seq_len) // 3 - 1)
    one_third = max(0, seq_len // 3 - 1)
    q_positions = sorted({one_third, two_thirds, last})
    return q_positions

def run_cosine(tok, model, input_ids, position_ids, layer, head, qpos_arg):
    cache = {}
    _capture_qk(model, layer, cache)
    attns = prefill(model, input_ids, position_ids)
    hook = cache.pop("_hook", None)
    if hook is not None:
        hook.remove()

    q, k = cache["q"], cache["k"]
    seq_len = q.shape[2]
    q_positions = list(_parse_qpos(qpos_arg, seq_len))
    series, k_norm = compute_cosine_series(q, k, head, k_sink_idx=0, q_positions=q_positions)
    attn_mat = attns[layer][0, head].detach().float().cpu()

    print(f"Cosine (Q[q_idx], k[0]) layer={layer} head={head}")
    cos_vals = []
    for qi, cos in series:
        attn_to_sink = float(attn_mat[qi, 0].item())
        print(f"  q={qi}\tcos={cos:.6f}\tattn={attn_to_sink:.6f}")
        cos_vals.append(cos)
    return {
        "layer": layer, "head": head, "q_positions": q_positions,
        "cos": cos_vals, "k_norm": float(k_norm)
    }

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

def pick_head(attentions, layer_idx, head_idx):
    attn_l = attentions[layer_idx]
    return attn_l[0, head_idx].detach().float().cpu().numpy()

def _plot_attn(attn, title, out_path):
    plt.figure(figsize=(6, 4.5))
    im = plt.imshow(attn, aspect="auto", origin="lower", interpolation="nearest")
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel("K")
    plt.ylabel("Q")
    cbar = plt.colorbar(im)
    cbar.set_label("Attention prob")
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()

def _plot_k_norm_progression(stats, outdir):
    layers = sorted({s["layer"] for s in stats})
    layers_to_kn = {s["layer"]: s["k_norm"] for s in stats}
    y = [layers_to_kn[L] for L in layers]
    plt.figure(figsize=(7, 3.5))
    plt.plot(layers, y, marker="o")
    plt.xlabel("Layer")
    plt.ylabel("||K[0]||")
    plt.title(f"K-norm progression")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"scan_knorm.png"), dpi=300)
    plt.close()

def _plot_cosine_stack(stats, outdir):
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
    fig.savefig(os.path.join(outdir, f"scan_cosine.png"), dpi=300)
    plt.close(fig)

def run_heatmap(model, input_ids, position_ids, layer, head, outdir, tag="BASE"):
    attn = prefill(model, input_ids, position_ids)
    head_attn = pick_head(attn, layer, head)
    tag_safe = str(tag).lower().replace(" ", "_")
    _plot_attn(
        head_attn, 
        f"{tag} L{layer} H{head}", 
        os.path.join(outdir, f"{tag_safe}_L{layer}_H{head}.png")
    )

def parse_overrides(s):
    """'12=0,42=1' -> [(12,0),(42,1)]"""
    if not s: return []
    out = []
    for part in s.split(","):
        if not part: continue
        ti, rp = part.split("=")
        out.append((int(ti), int(rp)))
    return out

def _capture_qk(model, layer_idx, cache):
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
        q = attn.q_norm(q).transpose(1, 2).contiguous()
        k = attn.k_norm(k).transpose(1, 2).contiguous()

        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if num_kv != num_heads:
            k = k.repeat_interleave(num_groups, dim=1)
        cache["q"] = q.detach().float().cpu()
        cache["k"] = k.detach().float().cpu()

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
    p.add_argument("--mask", default=None, help="Casual mask type")
    p.add_argument("--outdir", default="figures")
    args = p.parse_args()

    tok, model = load_model(args.model, args.device, args.dtype)

    if args.prompt:
        text = args.prompt
    else:
        text = "To understand the failure of window attention, we find an interesting phenomenon of autoregressive LLMs: a surprisingly large amount of attention score is allocated to the initial tokens, irrespective of their relevance to the language modeling task"

    input_ids = tokenize(tok, text)
    base_pos = make_position_ids(input_ids.shape[1])
    os.makedirs(args.outdir, exist_ok=True)

    rope_str = args.rope
    rope_overrides = parse_overrides(rope_str) if rope_str else None

    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    scan_stats = []

    def handle_one(layer, head):
        pos_ids_perturbed, applied = apply_perturbations(base_pos, rope_overrides=rope_overrides)

        if args.print_mode == "log":
            stats = run_cosine(tok, model, input_ids, pos_ids_perturbed, layer, head, args.qpos)
            scan_stats.append(stats)

        elif args.print_mode == "heatmap":
            tag = "PERTURBED"
            if rope_overrides:
                ov_str = ",".join([f"{ti}={rp}" for ti, rp in rope_overrides])
                tag = f"PERTURBED rope({ov_str})"
            run_heatmap(model, input_ids, pos_ids_perturbed, layer, head, args.outdir, tag=tag)

    if args.scan:
        for L in range(0, num_layers, 4):
            handle_one(L, args.head)

        if args.print_mode == "log" and len(scan_stats) > 0:
            scan_stats = sorted(scan_stats, key=lambda s: s["layer"])
            _plot_k_norm_progression(scan_stats, args.outdir)
            _plot_cosine_stack(scan_stats, args.outdir)

    else:
        handle_one(args.layer, args.head)

if __name__ == "__main__":
    main()