import torch
import numpy as np 
from transformers import AutoModelForCausalLM
from scipy import stats
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()
print(f"Model loaded.")

wv_matrices = {}
for i in range(model.config.num_hidden_layers):
    wv = model.model.layers[i].self_attn.v_proj.weight.data
    wv_matrices[i] = wv.cpu().float() 

results = {}
all_effective_ranks = []
k = 128

print("Analyzing matrices...")
for layer_idx, matrix in tqdm(wv_matrices.items()):
    _, S, _ = torch.linalg.svd(matrix, full_matrices=False)
    S = S.numpy()

    numerical_rank = int(np.sum(S > 1e-6 * S[0]))
    stable_rank = np.sum(S**2) / (S[0]**2)
    participation_ratio = (np.sum(S)**2) / np.sum(S**2)

    indices = np.arange(1, min(len(S), 100) + 1)
    log_indices = np.log(indices)
    log_values = np.log(S[:100] + 1e-10)
    slope, intercept, r_value, _, _ = stats.linregress(log_indices, log_values)
    decay_coefficient = -slope

    total_frobenius_sq = np.sum(S**2)

    cumsum_energy = np.cumsum(S**2) / total_frobenius_sq
    k_90 = np.argmax(cumsum_energy >= 0.90) + 1

    results[layer_idx] = {
        'shape': list(matrix.shape),
        'numerical_rank': numerical_rank,
        'stable_rank': stable_rank,
        'participation_ratio': participation_ratio,
        'decay_coefficient': decay_coefficient,
        'k_90': k_90,
    }
    all_effective_ranks.append(k_90)

numerical_ranks = [r['numerical_rank'] for r in results.values()]
effective_ranks = [r['stable_rank'] for r in results.values()]
participation_ratios = [r['participation_ratio'] for r in results.values()]
decay_coefficients = [r['decay_coefficient'] for r in results.values()]
k_90_values = [r['k_90'] for r in results.values()]

print(f"V_proj shape: {model.model.layers[0].self_attn.v_proj.weight.shape}")

print(f"\nNumerical rank:")
print(f"  Min: {np.min(numerical_ranks)}")
print(f"  Max: {np.max(numerical_ranks)}")
print(f"  Mean: {np.mean(numerical_ranks):.2f}")
print(f"  Std: {np.std(numerical_ranks):.2f}")

print(f"\nEffective rank (stable rank):")
print(f"  Min: {np.min(effective_ranks):.2f}")
print(f"  Max: {np.max(effective_ranks):.2f}")
print(f"  Mean: {np.mean(effective_ranks):.2f}")
print(f"  Std: {np.std(effective_ranks):.2f}")

print(f"\nParticipation ratio:")
print(f"  Min: {np.min(participation_ratios):.2f}")
print(f"  Max: {np.max(participation_ratios):.2f}")
print(f"  Mean: {np.mean(participation_ratios):.2f}")
print(f"  Std: {np.std(participation_ratios):.2f}")

print(f"\nDecay coefficient:")
print(f"  Min: {np.min(decay_coefficients):.2f}")
print(f"  Max: {np.max(decay_coefficients):.2f}")
print(f"  Mean: {np.mean(decay_coefficients):.2f}")
print(f"  Std: {np.std(decay_coefficients):.2f}")

print(f"\nRank needed for 90% frobenius norm:")
print(f"  Min: {np.min(k_90_values)}")
print(f"  Max: {np.max(k_90_values)}")
print(f"  Mean: {np.mean(k_90_values):.2f}")
print(f"  Std: {np.std(k_90_values):.2f}")

layer_indices = list(range(len(k_90_values)))
correlation, p_value = stats.spearmanr(layer_indices, k_90_values)
print(f"\nLayer depth vs k_90 correlation: {correlation:.3f} (p={p_value:.3e})")

layer_idx = 15
matrix = wv_matrices[layer_idx]
_, S, _ = torch.linalg.svd(matrix.float(), full_matrices=False)
S = S.numpy()

stable_rank = np.sum(S**2) / (S[0]**2)
participation_ratio = (np.sum(S)**2) / np.sum(S**2)
k_90 = np.argmax(np.cumsum(S**2) / np.sum(S**2) >= 0.9) + 1

fig, ax = plt.subplots(figsize=(12, 6))
ratios = S / S[0]

ax.fill_between(
    range(len(S)), 
    0, 
    ratios, 
    where=(ratios > 0.1), 
    color='darkgreen', 
    alpha=0.6,
    label=f"Strong (>10% of max): {np.sum(ratios > 0.1)} dims"
)

ax.fill_between(
    range(len(S)),
    0,
    ratios,
    where=(ratios <= 0.1) & (ratios > 0.01),
    color='gold',
    alpha=0.6,
    label=f"Medium (1-10%): {np.sum((ratios <= 0.1) & (ratios > 0.01))} dims"
)

ax.fill_between(
    range(len(S)),
    0,
    ratios,
    where=(ratios <= 0.01),
    color='crimson',
    alpha=0.6,
    label=f"Weak (<1%): {np.sum(ratios <= 0.01)} dims"
)

ax.axhline(y=0.1, color='darkgreen', linestyle='--', alpha=0.5, linewidth=1)
ax.axhline(y=0.01, color='gold', linestyle='--', alpha=0.5, linewidth=1)
ax.axhline(y=0.001, color='crimson', linestyle='--', alpha=0.5, linewidth=1)

ax.set_xlabel('Dimension Index', fontsize=12)
ax.set_ylabel('Relative Strength', fontsize=12)
ax.set_title(f'Layer {layer_idx}', fontsize=14, fontweight='bold')
ax.set_yscale('log')
ax.set_xlim(0, len(S))
ax.set_ylim(1e-4, 1.5)
ax.legend(loc='upper right', fontsize=11)
ax.grid(True, alpha=0.3, which='both')

ax.text(
    0.02, 
    0.5, 
    f'k for 90% energy: {k_90}', 
    transform=ax.transAxes, 
    fontsize=11,
    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
)

strong_count = np.sum(ratios > 0.1)
weak_count = np.sum(ratios <= 0.01)
ax.text(
    0.02,
    0.35,
    f'Only {strong_count} strong dims\n-> Low stable rank\nBut {weak_count} weak dims\n-> High participation',
    transform=ax.transAxes,
    fontsize=10,
    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
)

plt.tight_layout()
plt.savefig(f'layer_{layer_idx}_rank_analysis.png', dpi=300)
plt.show()
