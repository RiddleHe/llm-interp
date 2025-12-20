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

Descriptions needed.

```bash
python analyze_sink.py --prompt-file prompts.txt --find-key-subspace --layer [layer_idx] --head [head_idx] --outdir [outdir]
```

### Output decomposition

Running the following command will print the total variance, direction concentration on top-3 PC, and the cosine similarity with the mean output vector of the residual, attention, and MLP outputs for token 0 (sink) vs token 1 and token 8.

```bash
python analyze_sink.py --prompt-file prompts.txt --decompose-output --layer [layer_idx]
```