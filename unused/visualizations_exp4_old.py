"""
Visualization script for Experiment 4: Plasticity conditions comparison.

Conditions:
- baseline: Normal learning (lr=0.003, temp=1.0)
- adults: Low plasticity (lr=0.0003, temp=0.1)
- children: High plasticity (lr=0.03, temp=3.0)
"""
from pathlib import Path
import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

NUM_GENERATIONS = 5
N_AGENTS = 10
RANDOM_BASELINE = 1 / 6
PALETTE = 'Set2'

EXP_DIR = Path(__file__).parent.parent.parent.parent / "experiments" / "exp4"

CONDITIONS = ['baseline', 'adults', 'children', 'age_based']
CONDITION_LABELS = {
    'baseline': 'Baseline',
    'adults': 'Adults (low plasticity)',
    'children': 'Children (high plasticity)',
    'age_based': 'Age-based plasticity',
}

STATIC_METRICS = {
    'vocab_usage': {'ylabel': 'Vocab Usage', 'ylim': (0, 1.05)},
    'language_similarity': {'ylabel': 'Language Similarity', 'ylim': (0, 1.05)},
    'topographic_similarity': {'ylabel': 'Topographic Similarity', 'ylim': (-0.1, 1.05)},
    'posdis': {'ylabel': 'Positional Disentanglement', 'ylim': None},
    'bosdis': {'ylabel': 'Bag-of-Symbols Disentanglement', 'ylim': None},
    'message_length_mean': {'ylabel': 'Message Length', 'ylim': None},
}

CROSSGEN_METRICS = {
    'crossgen_accuracy': {'ylabel': 'Accuracy', 'ylim': (0, 1.05), 'baseline': True},
    'crossgen_similarity': {'ylabel': 'Similarity', 'ylim': (0, 1.05), 'baseline': False},
    'crossgen_forward': {'ylabel': 'Forward Accuracy', 'ylim': (0, 1.05), 'baseline': True},
    'crossgen_backward': {'ylabel': 'Backward Accuracy', 'ylim': (0, 1.05), 'baseline': True},
    'crossgen_acc_youngest': {'ylabel': 'Youngest Agent Accuracy', 'ylim': (0, 1.05), 'baseline': True},
    'crossgen_sim_youngest': {'ylabel': 'Youngest Agent Similarity', 'ylim': (0, 1.05), 'baseline': False},
}

X_LABELS = {'epoch': 'Epoch', 'death_number': 'Death Number', 'generation': 'Generation'}


def load_snapshot_metrics(run_dir: Path) -> pd.DataFrame:
    f = run_dir / 'snapshot_metrics.csv'
    return pd.read_csv(f) if f.exists() else pd.DataFrame()


def load_training_metrics(run_dir: Path) -> pd.DataFrame:
    f = run_dir / 'training_metrics.csv'
    return pd.read_csv(f) if f.exists() else pd.DataFrame()


def truncate_to_generations(df: pd.DataFrame) -> pd.DataFrame:
    max_deaths = (NUM_GENERATIONS - 1) * N_AGENTS
    return df[df['death_number'] <= max_deaths].copy()


def load_all_runs(k_value: int):
    snapshot_frames, acc_frames = [], []

    for cond in CONDITIONS:
        cond_dir = EXP_DIR / cond
        if not cond_dir.exists():
            continue

        # Search all subdirs for matching k_value
        for subdir in cond_dir.iterdir():
            if not subdir.is_dir():
                continue
            opts_file = subdir / 'opts.json'
            if not opts_file.exists():
                continue

            with open(opts_file) as f:
                opts = json.load(f)

            if opts.get('kill_epoch') != k_value:
                continue

            snap_df = load_snapshot_metrics(subdir)
            if not snap_df.empty:
                snap_df = truncate_to_generations(snap_df)
                snap_df['condition'] = cond
                snap_df['generation'] = snap_df['death_number'] // N_AGENTS + 1
                snapshot_frames.append(snap_df)

            train_df = load_training_metrics(subdir)
            if not train_df.empty:
                test_df = train_df[train_df['mode'] == 'test'].copy()
                max_epoch = (NUM_GENERATIONS - 1) * N_AGENTS * k_value
                test_df = test_df[test_df['epoch'] <= max_epoch]
                test_df['condition'] = cond
                acc_frames.append(test_df[['epoch', 'acc', 'condition']])

    snapshot_all = pd.concat(snapshot_frames, ignore_index=True) if snapshot_frames else pd.DataFrame()
    acc_all = pd.concat(acc_frames, ignore_index=True) if acc_frames else pd.DataFrame()
    return snapshot_all, acc_all


def plot_metric(df: pd.DataFrame, metric: str, x_col: str, ylabel: str, title: str,
                output_path: Path, ylim=None, show_baseline=False):
    if metric not in df.columns or df[metric].isna().all():
        print(f"  Skipping {output_path.name}: no data for {metric}")
        return False

    fig, ax = plt.subplots(figsize=(10, 6))
    conditions = [c for c in CONDITIONS if c in df['condition'].unique()]
    colors = sns.color_palette(PALETTE, n_colors=len(conditions))

    for i, cond in enumerate(conditions):
        subset = df[df['condition'] == cond].dropna(subset=[metric])
        if subset.empty:
            continue
        grouped = subset.groupby(x_col)[metric].mean().reset_index()
        label = CONDITION_LABELS.get(cond, cond)
        ax.plot(grouped[x_col], grouped[metric], color=colors[i], label=label, linewidth=1.5)

    if show_baseline:
        ax.axhline(y=RANDOM_BASELINE, linestyle='--', color='gray', alpha=0.7, linewidth=1.5)

    ax.set_xlabel(X_LABELS.get(x_col, x_col))
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(ylim)
    x_max = df[x_col].max()
    ax.set_xlim(left=0 if x_col != 'generation' else 1, right=x_max)
    if x_col == 'generation':
        ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    sns.despine()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return True


def generate_k_figures(k_value: int):
    output_dir = EXP_DIR / "figures" / f"k{k_value}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading data for k={k_value}...")
    snap_df, acc_df = load_all_runs(k_value)
    print(f"  Snapshots: {len(snap_df)}, Accuracy: {len(acc_df)}")

    if snap_df.empty and acc_df.empty:
        print("  No data found.")
        return

    count = 0

    # Test accuracy plots
    if not acc_df.empty:
        for x_col in ['epoch']:
            fname = f"accuracy_by_{x_col}.png"
            title = f"Test Accuracy (k={k_value})"
            if plot_metric(acc_df, 'acc', x_col, 'Accuracy', title, output_dir / fname, (0, 1.05), True):
                print(f"    {fname}")
                count += 1

    # Add death_number and generation to acc_df via epoch mapping
    if not acc_df.empty and not snap_df.empty:
        epoch_to_death = snap_df.groupby('epoch')['death_number'].first().to_dict()
        acc_df['death_number'] = acc_df['epoch'].map(epoch_to_death)
        acc_df['generation'] = acc_df['death_number'].apply(lambda d: d // N_AGENTS + 1 if pd.notna(d) else np.nan)

        for x_col in ['death_number', 'generation']:
            subset = acc_df.dropna(subset=[x_col])
            if subset.empty:
                continue
            fname = f"accuracy_by_{x_col.replace('_number', '')}.png"
            title = f"Test Accuracy (k={k_value})"
            if plot_metric(subset, 'acc', x_col, 'Accuracy', title, output_dir / fname, (0, 1.05), True):
                print(f"    {fname}")
                count += 1

    # Static metrics from snapshots
    for metric, cfg in STATIC_METRICS.items():
        for x_col in ['epoch', 'death_number', 'generation']:
            fname = f"{metric}_by_{x_col.replace('_number', '')}.png"
            title = f"{cfg['ylabel']} (k={k_value})"
            if plot_metric(snap_df, metric, x_col, cfg['ylabel'], title, output_dir / fname, cfg['ylim']):
                print(f"    {fname}")
                count += 1

    # Cross-gen metrics
    for metric, cfg in CROSSGEN_METRICS.items():
        for x_col in ['epoch', 'death_number', 'generation']:
            fname = f"{metric}_by_{x_col.replace('_number', '')}.png"
            title = f"{cfg['ylabel']} vs Gen1 (k={k_value})"
            if plot_metric(snap_df, metric, x_col, cfg['ylabel'], title, output_dir / fname,
                           cfg['ylim'], cfg.get('baseline', False)):
                print(f"    {fname}")
                count += 1

    print(f"  Generated {count} plots for k={k_value}")


def plot_thesis_comparison(k5_snap, k10_snap, k5_acc, k10_acc, output_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    conditions = [c for c in CONDITIONS if c in k5_snap['condition'].unique()]
    colors = sns.color_palette(PALETTE, n_colors=len(conditions))

    # (a) Test accuracy k=5
    ax = axes[0, 0]
    for i, cond in enumerate(conditions):
        subset = k5_acc[k5_acc['condition'] == cond].dropna(subset=['acc'])
        if subset.empty:
            continue
        grouped = subset.groupby('epoch')['acc'].mean().reset_index()
        label = CONDITION_LABELS.get(cond, cond)
        ax.plot(grouped['epoch'], grouped['acc'], color=colors[i], label=label, linewidth=1.5)
    ax.axhline(y=RANDOM_BASELINE, linestyle='--', color='gray', alpha=0.7)
    ax.set_title('(a) Test Accuracy (k=5)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # (b) Test accuracy k=10
    ax = axes[0, 1]
    for i, cond in enumerate(conditions):
        subset = k10_acc[k10_acc['condition'] == cond].dropna(subset=['acc'])
        if subset.empty:
            continue
        grouped = subset.groupby('epoch')['acc'].mean().reset_index()
        ax.plot(grouped['epoch'], grouped['acc'], color=colors[i], linewidth=1.5)
    ax.axhline(y=RANDOM_BASELINE, linestyle='--', color='gray', alpha=0.7)
    ax.set_title('(b) Test Accuracy (k=10)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # (c) Cross-gen accuracy k=5
    ax = axes[1, 0]
    for i, cond in enumerate(conditions):
        subset = k5_snap[k5_snap['condition'] == cond].dropna(subset=['crossgen_accuracy'])
        if subset.empty:
            continue
        grouped = subset.groupby('generation')['crossgen_accuracy'].mean().reset_index()
        ax.plot(grouped['generation'], grouped['crossgen_accuracy'], color=colors[i], linewidth=1.5, marker='o', markersize=4)
    ax.axhline(y=RANDOM_BASELINE, linestyle='--', color='gray', alpha=0.7)
    ax.set_title('(c) Cross-Gen Accuracy (k=5)')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1.05)
    ax.set_xlim(1, NUM_GENERATIONS)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.grid(True, alpha=0.3)

    # (d) Cross-gen accuracy k=10
    ax = axes[1, 1]
    for i, cond in enumerate(conditions):
        subset = k10_snap[k10_snap['condition'] == cond].dropna(subset=['crossgen_accuracy'])
        if subset.empty:
            continue
        grouped = subset.groupby('generation')['crossgen_accuracy'].mean().reset_index()
        ax.plot(grouped['generation'], grouped['crossgen_accuracy'], color=colors[i], linewidth=1.5, marker='o', markersize=4)
    ax.axhline(y=RANDOM_BASELINE, linestyle='--', color='gray', alpha=0.7)
    ax.set_title('(d) Cross-Gen Accuracy (k=10)')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1.05)
    ax.set_xlim(1, NUM_GENERATIONS)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.grid(True, alpha=0.3)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(conditions),
               bbox_to_anchor=(0.5, -0.02), fontsize=10)

    sns.despine()
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_thesis_figures():
    output_dir = EXP_DIR / "figures" / "thesis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating thesis figures...")
    k5_snap, k5_acc = load_all_runs(5)
    k10_snap, k10_acc = load_all_runs(10)

    if k5_snap.empty or k10_snap.empty:
        print("  Insufficient data for thesis figures")
        return

    plot_thesis_comparison(k5_snap, k10_snap, k5_acc, k10_acc, output_dir / 'exp4_comparison.png')
    print("  Saved exp4_comparison.png")


def print_summary(snap_df: pd.DataFrame, acc_df: pd.DataFrame, k_value: int):
    print(f"\n{'='*60}")
    print(f"SUMMARY FOR k={k_value}")
    print('='*60)

    if not acc_df.empty:
        print(f"\nFinal Test Accuracy:")
        for cond in CONDITIONS:
            subset = acc_df[acc_df['condition'] == cond]
            if subset.empty:
                continue
            final_acc = subset[subset['epoch'] == subset['epoch'].max()]['acc'].values
            if len(final_acc) > 0:
                print(f"  {CONDITION_LABELS.get(cond, cond):30s}: {final_acc[0]*100:.1f}%")

    if not snap_df.empty:
        final_df = snap_df[snap_df['generation'] == NUM_GENERATIONS]
        if not final_df.empty:
            print(f"\nFinal Cross-Gen Metrics (Gen {NUM_GENERATIONS}):")
            for cond in CONDITIONS:
                subset = final_df[final_df['condition'] == cond]
                if subset.empty:
                    continue
                cg_acc = subset['crossgen_accuracy'].mean()
                cg_sim = subset['crossgen_similarity'].mean()
                print(f"  {CONDITION_LABELS.get(cond, cond):30s}: acc={cg_acc:.3f}, sim={cg_sim:.3f}")


if __name__ == "__main__":
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)

    print("=" * 60)
    print("EXPERIMENT 4: PLASTICITY CONDITIONS COMPARISON")
    print("=" * 60)

    for k in [5, 10]:
        print(f"\n{'-'*60}")
        print(f"TURNOVER RATE k={k}")
        print('-'*60)
        generate_k_figures(k)
        snap_df, acc_df = load_all_runs(k)
        print_summary(snap_df, acc_df, k)

    print(f"\n{'-'*60}")
    print("THESIS FIGURES")
    print('-'*60)
    generate_thesis_figures()

    print(f"\n{'='*60}")
    print("Complete!")
    print("=" * 60)
