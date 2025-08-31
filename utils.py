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

def plot_attention_weights(attention_dict, step):
    sorted_layers = sorted(attention_dict.keys())

    prev_pre_attention_matrix = np.stack(
        [attention_dict[layer_idx][0] for layer_idx in sorted_layers],
        axis=0
    )
    prev_attention_matrix = np.stack(
        [attention_dict[layer_idx][1] for layer_idx in sorted_layers],
        axis=0
    )
    curr_attention_matrix = np.stack(
        [attention_dict[layer_idx][2] for layer_idx in sorted_layers],
        axis=0
    )

    fig, axes = plt.subplots(1, 3, figsize=(24, len(sorted_layers)))

    for idx, (attention_matrix, ax, title_suffix) in enumerate(
        [
            ([prev_pre_attention_matrix, axes[0], f"Previous step {step -2}"]),
            ([prev_attention_matrix, axes[1], f"Previous step {step -1}"]),
            ([curr_attention_matrix, axes[2], f"Current step {step}"])
        ]
    ):
        im = ax.imshow(
            attention_matrix, cmap="hot",
            aspect="auto", interpolation="nearest"
        )

        ax.set_xlabel('Token position', fontsize=12)
        ax.set_ylabel('Layer', fontsize=12)
        ax.set_title(f'Attention Scores Heatmap - {title_suffix}', fontsize=14)

        ax.set_yticks(range(len(sorted_layers)))
        ax.set_yticklabels(sorted_layers)

        token_len = attention_matrix.shape[-1]
        ax.set_xticks(range(token_len))
        ax.set_xticklabels(range(- token_len + 1, 1)) # +1 to include current token

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Attention Score', rotation=270, labelpad=20)

        ax.set_xticks(np.arange(token_len) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(sorted_layers)) - 0.5, minor=True)

        ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.3, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"visualizations/budget_token_attention_weights_{step}.png")

def plot_key_norms(key_norms, step):
    sorted_layers = sorted(key_norms.keys())
    norm_matrix = np.stack(
        [key_norms[layer_idx][0] for layer_idx in sorted_layers],
        axis=0
    )

    fig, ax = plt.subplots(figsize=(12, len(sorted_layers)))

    im = ax.imshow(
        norm_matrix, cmap="hot",
        aspect="auto", interpolation="nearest"
    )

    ax.set_xlabel('Token position', fontsize=12)
    ax.set_ylabel('Layer', fontsize=12)
    ax.set_title(f'K vector norm for last 8 tokens at step {step}', fontsize=14)

    ax.set_yticks(range(len(sorted_layers)))
    ax.set_yticklabels(sorted_layers)

    token_len = norm_matrix.shape[-1]
    ax.set_xticks(range(token_len))
    ax.set_xticklabels(range(- token_len + 1, 1)) # include current token

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('K vector norm', rotation=270, labelpad=20)

    ax.set_xticks(np.arange(token_len) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(sorted_layers)) - 0.5, minor=True)

    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.3, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"visualizations/budget_token_k_norms_{step}.png")

def plot_heatmap_generic(matrix, step, layer_indices, title, cmap="hot"):
    sorted_layers = sorted(layer_indices)

    fig, ax = plt.subplots(figsize=(12, len(sorted_layers)))

    im = ax.imshow(
        matrix, cmap=cmap,
        aspect="auto", interpolation="nearest"
    )

    ax.set_xlabel('Token position', fontsize=12)
    ax.set_ylabel('Layer', fontsize=12)
    ax.set_title(f'{title} at step {step}', fontsize=14)

    ax.set_yticks(range(len(sorted_layers)))
    ax.set_yticklabels(sorted_layers)

    token_len = matrix.shape[-1]
    ax.set_xticks(range(token_len))
    ax.set_xticklabels(range(- token_len + 1, 1)) # including current token

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Magnitude', rotation=270, labelpad=20)

    ax.set_xticks(np.arange(token_len) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(sorted_layers)) - 0.5, minor=True)

    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.3, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"visualizations/budget_token_heatmap_{step}_{'_'.join(title.split(' '))}.png")