# Attention Sink Experiments

## Setup

```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install numpy matplotlib tqdm transformers accelerate
```

## Topics

The script uses CLI arguments to simulate a variety of attention sink related phenomena.

### Progression of vector norms and spreads across layers

```bash
python analyze_sink.py --scan --print qkv
```

### Output activation in a specific layer

```bash
python analyze_sink.py --decompose-output --layer [layer_idx]
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

### MLP module outputs in a specific layer

```bash
python analyze_sink.py --find-mlp-subspace --layer [layer_idx] --mlp z
```

### Ablating MLP outputs and see subsequent layer's attention scores

```bash
python analyze_sink.py --layer [layer_idx] --ablate-mlp-out [direction|magnitude]
```

### Ablating MLP input and see important dimensions

```bash
python analyze_sink.py --layer [layer_idx] --decompose-mlp-in
