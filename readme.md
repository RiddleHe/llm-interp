# LLM Brain Surgeon

A collection of lightweight, standalone scripts to visualize the reasoning traces of LLMs. (Help me pick a better repo name!)

To run any single one, just call `python <script_name>.py` in the subdir.

## Research topics

### Attention sinks

We want to understand when attention sinks occur, and what are some causal factors that contribute to its emergence.

You can now study the effect of RoPE, model weights, and the value of attention scores of sink token on the sink phenomenon.

```
cd attention_sink
python analyze_sink.py --scan \
    --print [heatmap, log] \
    --rope 0=12,1=13,2=14,3=15 \
    --random-init \
    --lower-attn \
```

### Deterministic attention

We want to understand when nondeterministic inference occurs even with temperature=0. 

You can now study the scenario where the variable can be batch size, prefill strategy, or attention implementation.

```
cd attention_nondeterministic
python analyze_attention_nondeterministic.py \
    --mode [split_kv, prefill] \
    --attn_implementation [flash_attention_2, sdpa] \
    --seq_lens [4096, 6144] \
    --batch_sizes [1, 2, 4, 8] \
    --use_fa2_repo --fa2_deterministic --fa2_split_size 1024
```

### Thinking budget

We want to study how models track and budget the number of thinking tokens in their CoT, especially for `ByteDance-Seed/Seed-OSS-36B-Instruct` by ByteDance.

You can now swap the RoPE of the budget token, regenerate it again in the decoding step, or do other fun things.
