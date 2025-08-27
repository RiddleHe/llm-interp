import matplotlib.pyplot as plt
import numpy as np

def plot_budget_token_ranks(budget_token_ranks, start_idx, end_idx):
    budget_token_ranks = budget_token_ranks[start_idx:end_idx]

    fig, ax = plt.subplots(figsize=(12, 3))

    ranks_array = np.array(budget_token_ranks).reshape(1, -1)

    im = ax.imshow(ranks_array, aspect='auto', cmap='viridis_r', interpolation='nearest')

    for i in range(len(budget_token_ranks)):
        rank_val = budget_token_ranks[i]
        text_color = 'white' if rank_val > np.mean(budget_token_ranks) else 'black'
        ax.text(i, 0, f"{rank_val}", ha='center', va='center', color=text_color, fontsize=6)

    # ax.set_xticks(np.arange(start_idx, end_idx))
    ax.set_xticks(np.arange(0, len(budget_token_ranks)))
    ax.set_xticklabels(np.arange(start_idx, end_idx))

    ax.set_yticks([0])
    ax.set_yticklabels(['Budget reflection token'])
    
    ax.set_xlabel('Decoding step')
    ax.set_title('Budget reflection token rank across decoding steps')

    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.1)
    cbar.set_label('Rank (0=highest probability)')

    plt.tight_layout()
    plt.savefig(f"visualizations/budget_token_ranks_{start_idx}_{end_idx}.png")

def plot_hidden_token_ranks(hidden_token_ranks, step):
    fig, ax = plt.subplots(figsize=(12, 3))

    ranks_array = np.array(hidden_token_ranks).reshape(1, -1)

    im = ax.imshow(ranks_array, aspect='auto', cmap='viridis_r', interpolation='nearest')

    for i in range(len(hidden_token_ranks)):
        rank_val = hidden_token_ranks[i]
        text_color = 'white' if rank_val > np.mean(hidden_token_ranks) else 'black'
        ax.text(i, 0, f"{rank_val}", ha='center', va='center', color=text_color, fontsize=6)

    # ax.set_xticks(np.arange(start_idx, end_idx))
    ax.set_xticks(np.arange(0, len(hidden_token_ranks)))
    ax.set_xticklabels(np.arange(0, len(hidden_token_ranks)))

    ax.set_yticks([0])
    ax.set_yticklabels(['Budget reflection token'])
    
    ax.set_xlabel('Hidden state index')
    ax.set_title(f'Budget reflection token rank across hidden layers at step {step}')

    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.1)
    cbar.set_label('Rank (0=highest probability)')

    plt.tight_layout()
    plt.savefig(f"visualizations/budget_token_ranks_hidden_{step}.png")