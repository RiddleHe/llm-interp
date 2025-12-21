# Attention Sink Experiments

## Setup

```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install numpy matplotlib tqdm transformers
```

## Topics

The script uses CLI arguments to simulate a variety of attention sink related phenomena.

### Progression of vector norms and spreads across layers

```bash
python analyze_sink.py --scan --print qkv --prompt-file prompts.txt --outdir [outdir] --sink-idx [idx] --scan-interval [itvl]
```

### Key vectors in a specific layer

```bash
python analyze_sink.py --prompt-file prompts.txt --find-key-subspace --layer [layer_idx] --head [head_idx] --outdir [outdir]
```

The output will have the following features:

|                   | tok0 | tok1 | tok8 |
|-------------------|------|------|------|
|spread_radius      |      |      |      |
|mean_norm          |      |      |      |
|cos(Q, K)          |      |      |      |
|cos_centered(Q, K) |      |      |      |
|frac_Q_along_mean_K|      |      |      |

### Output activation in a specific layer

```bash
python analyze_sink.py --prompt-file prompts.txt --decompose-output --layer [layer_idx]
```

The output will have the following features:

|                   | tok0 | tok1 | tok8 |
|-------------------|------|------|------|
|norm_Y             |      |      |      |
|spread_Y           |      |      |      |
|norm_residual      |      |      |      |
|norm_attn          |      |      |      |
|norm_mlp           |      |      |      |
|spread_residual    |      |      |      |
|spread_attn        |      |      |      |
|spread_mlp         |      |      |      |