# LLM Brain Surgeon

A collection of lightweight, standalone scripts to perform mechanistic interpretability observations on chosen LLMs. Topics include the geometric mechanism of attention sinks, the isolated simulations of non-deterministic attention outputs, etc.

In each subdir, which is headed by a research topic, a `readme.md` is (or will be) available for instructions on how to run the single python script.

## Research topics

### Attention sinks (actively developing)

We want to understand when attention sinks occur, and what are some causal factors that contribute to its emergence.

We find that token 0 manipulates a particular layer output to reside in a low-variance manifold, which it can then linearly transform into strong signals for alignment when doing QK computation.

### Deterministic attention

We want to understand when nondeterministic inference occurs even with temperature=0. 

We find that by changing the prefilling strategy or batch size, softmax outputs drift from the baseline.


### Thinking budget

We want to study how models track and budget the number of thinking tokens in their CoT, especially for `ByteDance-Seed/Seed-OSS-36B-Instruct` by ByteDance.

We find that special layers are activated before the budgeting occurs to only focus its attention on the last two tokens.
