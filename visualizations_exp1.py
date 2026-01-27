"""
Visualization script for Experiment 1: Baseline population study.

Varying population size (N=2,4,6,...,20) without turnover.
"""
from pathlib import Path
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATION
# =============================================================================

PALETTE = 'flare'

EXPERIMENT_DIR = Path(__file__).parent.parent.parent.parent / "experiments" / "exp1_baseline" / "runs"

METRICS = {
    'acc': {'ylabel': 'Accuracy', 'ylim': (0, 1.05), 'title': 'Test Accuracy', 'smooth': 5},
    'vocab_usage': {'ylabel': 'Vocab Usage', 'ylim': (0, 1.05), 'title': 'Vocabulary Usage', 'smooth': None},
    'language_similarity': {'ylabel': 'Language Similarity', 'ylim': (0, 1.05), 'title': 'Language Similarity', 'smooth': None},
    'message_length_mean': {'ylabel': 'Message Length', 'ylim': None, 'title': 'Message Length', 'smooth': None},
    'topographic_similarity': {'ylabel': 'Topographic Similarity', 'ylim': (-0.1, 1.05), 'title': 'Topographic Similarity', 'smooth': None},
    'posdis': {'ylabel': 'Posdis', 'ylim': None, 'title': 'Positional Disentanglement', 'smooth': None},
    'bosdis': {'ylabel': 'Bosdis', 'ylim': None, 'title': 'Bag-of-Symbols Disentanglement', 'smooth': None},
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_training_metrics(run_dir: Path) -> pd.DataFrame:
    f = run_dir / 'training_metrics.csv'
    return pd.read_csv(f) if f.exists() else pd.DataFrame()


def load_snapshot_metrics(run_dir: Path) -> pd.DataFrame:
    f = run_dir / 'snapshot_metrics.csv'
    return pd.read_csv(f) if f.exists() else pd.DataFrame()


def get_metric_folder(metric: str) -> str:
    if metric == 'acc':
        return 'accuracy'
    if metric == 'message_length_mean':
        return 'message_length'
    return metric


def load_all_runs():
    acc_frames, lang_frames = [], []

    run_dirs = sorted(
        [d for d in EXPERIMENT_DIR.iterdir() if d.is_dir() and (d / 'opts.json').exists()],
        key=lambda x: int(x.name) if x.name.isdigit() else 0
    )

    for run_dir in run_dirs:
        with open(run_dir / 'opts.json') as f:
            opts = json.load(f)

        n_agents = opts.get('n_agents')
        seed = opts.get('random_seed')

        training_df = load_training_metrics(run_dir)
        if not training_df.empty:
            test_df = training_df[training_df['mode'] == 'test'].copy()
            if not test_df.empty:
                test_df['n_agents'] = n_agents
                test_df['seed'] = seed
                acc_frames.append(test_df[['epoch', 'acc', 'n_agents', 'seed']])

        snapshot_df = load_snapshot_metrics(run_dir)
        if not snapshot_df.empty:
            snapshot_df['n_agents'] = n_agents
            snapshot_df['seed'] = seed
            lang_frames.append(snapshot_df)

    acc_all = pd.concat(acc_frames, ignore_index=True) if acc_frames else pd.DataFrame()
    lang_all = pd.concat(lang_frames, ignore_index=True) if lang_frames else pd.DataFrame()
    return acc_all, lang_all


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_metric_curve(df, metric, ylabel, title, output_path, ylim=None, smooth_window=None):
    fig, ax = plt.subplots(figsize=(10, 6))

    population_sizes = sorted(df['n_agents'].unique())
    palette = sns.color_palette(PALETTE, n_colors=len(population_sizes))

    for i, n_agents in enumerate(population_sizes):
        subset = df[df['n_agents'] == n_agents]
        grouped = subset.groupby('epoch')[metric].agg(['mean', 'std']).reset_index()

        if smooth_window:
            grouped['mean'] = grouped['mean'].rolling(window=smooth_window, min_periods=1).mean()
            grouped['std'] = grouped['std'].rolling(window=smooth_window, min_periods=1).mean()

        ax.plot(grouped['epoch'], grouped['mean'], color=palette[i], label=f'N={n_agents}', linewidth=1.5)
        ax.fill_between(grouped['epoch'], grouped['mean'] - grouped['std'],
                       grouped['mean'] + grouped['std'], color=palette[i], alpha=0.2)

    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(ylim)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    sns.despine()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_final_boxplot(df, metric, ylabel, title, output_path, ylim=None):
    fig, ax = plt.subplots(figsize=(10, 6))

    final_df = df.groupby(['n_agents', 'seed']).last().reset_index()
    order = sorted(final_df['n_agents'].unique())
    sns.boxplot(data=final_df, x='n_agents', y=metric, order=order, ax=ax, palette=PALETTE)

    ax.set_xlabel('Population Size (N)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    sns.despine(left=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)

    output_dir = EXPERIMENT_DIR.parent / "figures"
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("EXPERIMENT 1: BASELINE POPULATION STUDY")
    print("=" * 60)

    print(f"\nLoading runs from {EXPERIMENT_DIR}...")
    acc_df, lang_df = load_all_runs()
    print(f"  Accuracy records: {len(acc_df)}, Language records: {len(lang_df)}")

    if acc_df.empty and lang_df.empty:
        print("No data found.")
        exit(1)

    count = 0

    if not acc_df.empty:
        folder = output_dir / 'accuracy'
        folder.mkdir(exist_ok=True)
        cfg = METRICS['acc']

        plot_metric_curve(acc_df, 'acc', cfg['ylabel'], f"{cfg['title']} vs Epoch",
                         folder / 'acc_by_epoch.png', cfg['ylim'], cfg['smooth'])
        print(f"    accuracy/acc_by_epoch.png")
        count += 1

        plot_final_boxplot(acc_df, 'acc', cfg['ylabel'], f"Final {cfg['title']} by Population Size",
                          folder / 'acc_final.png', cfg['ylim'])
        print(f"    accuracy/acc_final.png")
        count += 1

    if not lang_df.empty:
        for metric in ['vocab_usage', 'language_similarity', 'message_length_mean',
                       'topographic_similarity', 'posdis', 'bosdis']:
            if metric not in lang_df.columns:
                continue

            cfg = METRICS[metric]
            folder = output_dir / get_metric_folder(metric)
            folder.mkdir(exist_ok=True)

            plot_metric_curve(lang_df, metric, cfg['ylabel'], f"{cfg['title']} vs Epoch",
                             folder / f'{metric}_by_epoch.png', cfg['ylim'], cfg['smooth'])
            print(f"    {get_metric_folder(metric)}/{metric}_by_epoch.png")
            count += 1

            plot_final_boxplot(lang_df, metric, cfg['ylabel'], f"Final {cfg['title']} by Population Size",
                              folder / f'{metric}_final.png', cfg['ylim'])
            print(f"    {get_metric_folder(metric)}/{metric}_final.png")
            count += 1

    print(f"\nBaseline study: {count} plots saved to {output_dir}")
