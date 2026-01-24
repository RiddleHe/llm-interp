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
```

## Research philosophy

How to conduct experiments effectively on attention sinks has produced a lot of "scar tissues" in the last three months for someone who DIY all such things. 
I then come to the conclusion that to characterize the special mechanism of any group of geometric objects (eg. the MLP.6's inputs for token 0 vs for other tokens), they have to satisfy two conditions:

(A) there exists a **causal** factor relevant to our investigation (eg. a rank-2 subspace can ablate attention sinks);

(B) **only** this group of objects have a specific **geometric** relation to this causal factor (eg. only MLP.6's inputs for token 0 have a large projection onto this subspace above baseline).

Then, research just simplifies to: (1) use any probing technique (eg. reading off activations, doing gradient analysis) to find any interesting factor that is causally important and then (2) use linear algebra to find the group's relation to it.

This also makes me realize that I spend around two months doing trivial things that are relevant to neither (1) or (2), and the whole research can wrap up way sooner.