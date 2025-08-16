import torch
import numpy as np 
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

model_name = "Qwen/Qwen3-8B"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name)
print(f"Model loaded.")

leetcode_prompt = """We can scramble a string s to get a string t using the following algorithm:

If the length of the string is 1, stop.
If the length of the string is > 1, do the following:
Split the string into two non-empty substrings at a random index, i.e., if the string is s, divide it to x and y where s = x + y.
Randomly decide to swap the two substrings or to keep them in the same order. i.e., after this step, s may become s = x + y or s = y + x.
Apply step 1 recursively on each of the two substrings x and y.
Given two strings s1 and s2 of the same length, return true if s2 is a scrambled string of s1, otherwise, return false."""

inputs = tokenizer(leetcode_prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=512,
        output_scores=True,
        return_dict_in_generate=True,
        do_sample=False,
        use_cache=True,
    )
logits = torch.stack(outputs.scores) # (seq_len, batch_size, vocab_size)
print(f"Logits generated, shape: {logits.shape}.")

token_entropies = []
conditional_entropies = []
for logit in tqdm(logits, desc="Calculating entropies"):
    probs = torch.softmax(logit[0], dim=-1)
    top_probs, _ = torch.topk(probs, k=100)
    top_probs = top_probs / top_probs.sum()

    entropy = -torch.sum(top_probs * torch.log(top_probs + 1e-10))
    token_entropies.append(entropy.item())
    conditional_entropies.append(entropy.item()) # model generation is condtioned on prev toks
print(f"Entropies calculated.")

cumulative_entropy_rate = np.cumsum(conditional_entropies) / np.arange(1, len(conditional_entropies) + 1)

generated_ids = outputs.sequences[0, inputs.input_ids.shape[1]:]
generated_tokens = tokenizer.convert_ids_to_tokens(generated_ids)
# generated_text = tokenizer.decode(generated_ids)

n_top_tokens = 8
n_gram_size = 8
top_entropy_indices = np.argsort(token_entropies)[-n_top_tokens:][::-1]
for rank, idx in enumerate(top_entropy_indices):
    token = generated_tokens[idx]
    token_text = tokenizer.decode(generated_ids[idx:idx+1], skip_special_tokens=False)
    entropy = token_entropies[idx]

    start_context = max(0, idx - n_gram_size // 2)
    end_context = min(len(generated_tokens), idx + n_gram_size // 2 + 1)
    context_tokens = generated_tokens[start_context:end_context]
    context_text = tokenizer.decode(generated_ids[start_context:end_context], skip_special_tokens=False)

    print(f"Token: {token_text} (Entropy: {entropy:.4f}, Rank: {rank+1})")
    print(f"Context: {context_text}")
    print("\n")

window_size = 16
sliding_window_entropies = []
window_positions = []
if len(token_entropies) >= window_size:
    for i in range(len(token_entropies) - window_size + 1):
        window_avg = np.mean(token_entropies[i:i+window_size])
        sliding_window_entropies.append(window_avg)
        window_positions.append((i, i + window_size))

plt.figure(figsize=(16, 6))
plt.plot(token_entropies, alpha=0.5, color='lightblue', linewidth=1, label='Token Entropy')

window_centers = [(start + end - 1) / 2 for start, end in window_positions]
plt.plot(
    window_centers,
    sliding_window_entropies,
    color='blue',
    linewidth=2,
    label=f'Sliding Window (Size={window_size})'
)

threshold = np.percentile(sliding_window_entropies, 90)
plt.axhline(
    y=threshold,
    color='red',
    linestyle='--',
    alpha=0.5,
    label=f'Threshold at 90th percentile: {threshold:.2f}'
)

high_entropy_regions = []
for i, (window_entropy, (start, end)) in enumerate(zip(sliding_window_entropies, window_positions)):
    if window_entropy > threshold:
        plt.axvspan(start, end, color='red', alpha=0.2)
        high_entropy_regions.append((start, end, float(window_entropy)))

merged_high_entropy_regions = []
if high_entropy_regions:
    merged_high_entropy_regions = [high_entropy_regions[0]]
    for start, end, entropy_val in high_entropy_regions[1:]: # use greedy merging
        last_start, last_end, last_entropy = merged_high_entropy_regions[-1]
        if start <= last_end:
            merged_end = max(last_end, end)
            merged_entropy = np.mean(token_entropies[last_start:merged_end])
            merged_high_entropy_regions[-1] = (last_start, merged_end, merged_entropy)
        else:
            merged_high_entropy_regions.append((start, end, entropy_val))


plt.xlabel('Token Position', fontsize=12)
plt.ylabel('Entropy', fontsize=12)
plt.title('Token-level and Sliding Window Entropy Analysis', fontsize=14, fontweight='bold')
plt.legend()

plt.grid(True, alpha=0.3)
plt.tight_layout()
os.makedirs('entropy', exist_ok=True)
plt.savefig('entropy/token_entropy_leetcode.png')

print(f"\nHigh Entropy Regions:\n")
for start, end, entropy_val in merged_high_entropy_regions:
    window_tokens = generated_tokens[start:end]
    window_text = tokenizer.decode(generated_ids[start:end], skip_special_tokens=False)
    print(f"Position [{start}:{end}] - Entropy: {entropy_val:.4f}")
    print(f"Text: {window_text}")
    print("\n")

print(f"Threshold: {threshold:.4f}")