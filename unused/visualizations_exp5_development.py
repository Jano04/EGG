"""
Visualization script for Experiment 5: Plasticity conditions comparison.
"""
from pathlib import Path
import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

RANDOM_BASELINE = 1 / 6
N_AGENTS = 10  # Number of agents per generation in exp5
EXP_DIR = Path(__file__).parent.parent.parent.parent / "experiments" / "exp5"

ALL_CONDITIONS = [
    'baseline', 'adults', 'adults_v6',
    'children_v1', 'children_v2', 'children_v3', 'children_v4', 'children_v5', 'children_v6',
    'age_based_v1', 'age_based_v2', 'age_based_v3', 'age_based_v4',
    'age_based_stagger_v1', 'age_based_stagger_v2', 'age_based_stagger_v3', 'age_based_stagger_v4',
    'age_based_stagger_adults_v1', 'age_based_stagger_adults_v2', 'age_based_stagger_adults_v3',
    'age_based_stagger_adults_v4', 'age_based_stagger_adults_v5', 'age_based_stagger_adults_v6',
]

SHARED_CONDITIONS = [
    'baseline',
    'adults', 'adults_v6',
    'children_v1', 'children_v2', 'children_v3', 'children_v4', 'children_v5', 'children_v6',
    'age_based_stagger_adults_v1', 'age_based_stagger_adults_v2', 'age_based_stagger_adults_v3',
    'age_based_stagger_adults_v4', 'age_based_stagger_adults_v5', 'age_based_stagger_adults_v6',
]

VERSION_GROUPS = {
    'v1': ['baseline', 'adults', 'children_v1', 'age_based_v1', 'age_based_stagger_v1', 'age_based_stagger_adults_v1'],
    'v2': ['baseline', 'adults', 'children_v2', 'age_based_v2', 'age_based_stagger_v2', 'age_based_stagger_adults_v2'],
    'v3': ['baseline', 'adults', 'children_v3', 'age_based_v3', 'age_based_stagger_v3', 'age_based_stagger_adults_v3'],
    'v4': ['baseline', 'adults', 'children_v4', 'age_based_v4', 'age_based_stagger_v4', 'age_based_stagger_adults_v4'],
    'v5': ['baseline', 'adults', 'children_v5', 'age_based_stagger_adults_v5'],
    'v6': ['baseline', 'adults_v6', 'children_v6', 'age_based_stagger_adults_v6'],
}

CONDITION_LABELS = {
    'baseline': 'Baseline',
    'adults': 'Adults', 'adults_v6': 'Adults',
    'children_v1': 'Children', 'children_v2': 'Children', 'children_v3': 'Children',
    'children_v4': 'Children', 'children_v5': 'Children', 'children_v6': 'Children',
    'age_based_v1': 'Age-based (young)', 'age_based_v2': 'Age-based (young)', 'age_based_v3': 'Age-based (young)',
    'age_based_v4': 'Age-based (young)',
    'age_based_stagger_v1': 'Age-based (mixed)', 'age_based_stagger_v2': 'Age-based (mixed)', 'age_based_stagger_v3': 'Age-based (mixed)',
    'age_based_stagger_v4': 'Age-based (mixed)',
    'age_based_stagger_adults_v1': 'Age-based (adult)', 'age_based_stagger_adults_v2': 'Age-based (adult)', 'age_based_stagger_adults_v3': 'Age-based (adult)',
    'age_based_stagger_adults_v4': 'Age-based (adult)', 'age_based_stagger_adults_v5': 'Age-based (adult)', 'age_based_stagger_adults_v6': 'Age-based',
}

SHARED_LABELS = {
    'baseline': 'Baseline',
    'adults': 'Adults', 'adults_v6': 'Adults v6',
    'children_v1': 'Children v1', 'children_v2': 'Children v2', 'children_v3': 'Children v3',
    'children_v4': 'Children v4', 'children_v5': 'Children v5', 'children_v6': 'Children v6',
    'age_based_stagger_adults_v1': 'Age-based v1', 'age_based_stagger_adults_v2': 'Age-based v2',
    'age_based_stagger_adults_v3': 'Age-based v3', 'age_based_stagger_adults_v4': 'Age-based v4',
    'age_based_stagger_adults_v5': 'Age-based v5', 'age_based_stagger_adults_v6': 'Age-based v6',
}

SHARED_COLORS = {
    'baseline': '#555555',
    'adults': '#ffb366', 'adults_v6': '#cc5500',
    'children_v1': '#c5e8c5', 'children_v2': '#98d898', 'children_v3': '#6bc86b',
    'children_v4': '#3eb83e', 'children_v5': '#28a028', 'children_v6': '#147014',
    'age_based_stagger_adults_v1': '#c5c5f0', 'age_based_stagger_adults_v2': '#9898e0',
    'age_based_stagger_adults_v3': '#6b6bd0', 'age_based_stagger_adults_v4': '#4040c0',
    'age_based_stagger_adults_v5': '#2828a8', 'age_based_stagger_adults_v6': '#141480',
}

COLORS_VERSION = {
    'baseline': '#1f77b4',
    'adults': '#ff7f0e', 'adults_v6': '#ff7f0e',
    'children_v1': '#2ca02c', 'children_v2': '#2ca02c', 'children_v3': '#2ca02c',
    'children_v4': '#2ca02c', 'children_v5': '#2ca02c', 'children_v6': '#2ca02c',
    'age_based_v1': '#d62728', 'age_based_v2': '#d62728', 'age_based_v3': '#d62728',
    'age_based_v4': '#d62728',
    'age_based_stagger_v1': '#9467bd', 'age_based_stagger_v2': '#9467bd', 'age_based_stagger_v3': '#9467bd',
    'age_based_stagger_v4': '#9467bd',
    'age_based_stagger_adults_v1': '#8c564b', 'age_based_stagger_adults_v2': '#8c564b', 'age_based_stagger_adults_v3': '#8c564b',
    'age_based_stagger_adults_v4': '#8c564b', 'age_based_stagger_adults_v5': '#8c564b', 'age_based_stagger_adults_v6': '#8c564b',
}

LANGUAGE_METRICS = {
    'vocab_usage': {'ylabel': 'Vocab Usage', 'ylim': (0, 1.05), 'title': 'Vocabulary Usage'},
    'language_similarity': {'ylabel': 'Similarity', 'ylim': (0, 1.05), 'title': 'Language Similarity'},
    'topographic_similarity': {'ylabel': 'Topsim', 'ylim': (0, 1.05), 'title': 'Topographic Similarity'},
    'posdis': {'ylabel': 'Posdis', 'ylim': (0, 1.05), 'title': 'Positional Disentanglement'},
    'bosdis': {'ylabel': 'Bosdis', 'ylim': (0, 1.05), 'title': 'Bag-of-Symbols Disentanglement'},
    'message_length_mean': {'ylabel': 'Length', 'ylim': None, 'title': 'Message Length'},
}


def find_seed_dirs(cond_dir: Path):
    """Find all seed directories for a condition. Returns list of (seed_id, path) tuples."""
    seed_dirs = []
    if (cond_dir / 'snapshot_metrics.csv').exists():
        opts_file = cond_dir / 'opts.json'
        if opts_file.exists():
            with open(opts_file) as f:
                opts = json.load(f)
            seed = opts.get('random_seed', 0)
        else:
            seed = 0
        seed_dirs.append((seed, cond_dir))
    for subdir in sorted(cond_dir.iterdir()):
        if subdir.is_dir() and subdir.name.startswith('s') and subdir.name[1:].isdigit():
            if (subdir / 'snapshot_metrics.csv').exists():
                opts_file = subdir / 'opts.json'
                if opts_file.exists():
                    with open(opts_file) as f:
                        opts = json.load(f)
                    seed = opts.get('random_seed', int(subdir.name[1:]))
                else:
                    seed = int(subdir.name[1:])
                seed_dirs.append((seed, subdir))
    return seed_dirs


def load_all_runs():
    snapshot_frames, acc_frames = [], []
    for cond in ALL_CONDITIONS:
        cond_dir = EXP_DIR / cond
        if not cond_dir.exists():
            continue
        snap_file = cond_dir / 'snapshot_metrics.csv'
        if snap_file.exists():
            df = pd.read_csv(snap_file)
            df['condition'] = cond
            snapshot_frames.append(df)
        train_file = cond_dir / 'training_metrics.csv'
        if train_file.exists():
            df = pd.read_csv(train_file)
            df = df[df['mode'] == 'test'][['epoch', 'acc']].copy()
            df['condition'] = cond
            acc_frames.append(df)
    snapshot_all = pd.concat(snapshot_frames, ignore_index=True) if snapshot_frames else pd.DataFrame()
    acc_all = pd.concat(acc_frames, ignore_index=True) if acc_frames else pd.DataFrame()
    return snapshot_all, acc_all


def load_runs_with_seeds(conditions):
    """Load data from all conditions, including seed subdirectories."""
    snapshot_frames, acc_frames = [], []
    for cond in conditions:
        cond_dir = EXP_DIR / cond
        if not cond_dir.exists():
            continue
        seed_dirs = find_seed_dirs(cond_dir)
        for seed, seed_dir in seed_dirs:
            snap_file = seed_dir / 'snapshot_metrics.csv'
            if snap_file.exists():
                df = pd.read_csv(snap_file)
                df['condition'] = cond
                df['seed'] = seed
                snapshot_frames.append(df)
            train_file = seed_dir / 'training_metrics.csv'
            if train_file.exists():
                df = pd.read_csv(train_file)
                df = df[df['mode'] == 'test'][['epoch', 'acc']].copy()
                df['condition'] = cond
                df['seed'] = seed
                acc_frames.append(df)
    snapshot_all = pd.concat(snapshot_frames, ignore_index=True) if snapshot_frames else pd.DataFrame()
    acc_all = pd.concat(acc_frames, ignore_index=True) if acc_frames else pd.DataFrame()
    return snapshot_all, acc_all


def has_multiple_seeds(df, condition):
    """Check if a condition has multiple seeds in the dataframe."""
    if 'seed' not in df.columns:
        return False
    subset = df[df['condition'] == condition]
    return subset['seed'].nunique() > 1


def plot_line(df, conditions, x_col, y_col, output_path, title, ylabel, labels, colors, ylim=(0, 1.05), show_baseline=True):
    if y_col not in df.columns or df[y_col].isna().all():
        return
    available = [c for c in conditions if c in df['condition'].unique()]
    if not available:
        return
    plot_df = df[df['death_number'] >= 0].copy() if x_col == 'death_number' else df
    fig, ax = plt.subplots(figsize=(10, 6))
    for cond in available:
        subset = plot_df[plot_df['condition'] == cond].dropna(subset=[y_col])
        if not subset.empty:
            ax.plot(subset[x_col], subset[y_col], color=colors[cond],
                    label=labels.get(cond, cond), linewidth=1.5, marker='o', markersize=3)
    if show_baseline:
        ax.axhline(y=RANDOM_BASELINE, linestyle='--', color='gray', alpha=0.7, linewidth=1.5)
    ax.set_xlabel('Epoch' if x_col == 'epoch' else 'Death Number')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(ylim)
    ax.set_xlim(0, plot_df[x_col].max())
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    sns.despine()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_line_with_seeds(df, conditions, x_col, y_col, output_path, title, ylabel, labels, colors, ylim=(0, 1.05), show_baseline=True):
    """Plot with mean line and std shading when multiple seeds exist."""
    if y_col not in df.columns or df[y_col].isna().all():
        return
    available = [c for c in conditions if c in df['condition'].unique()]
    if not available:
        return
    plot_df = df[df['death_number'] >= 0].copy() if x_col == 'death_number' else df.copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    for cond in available:
        subset = plot_df[plot_df['condition'] == cond].dropna(subset=[y_col])
        if subset.empty:
            continue
        if has_multiple_seeds(subset, cond):
            grouped = subset.groupby(x_col)[y_col].agg(['mean', 'std']).reset_index()
            ax.plot(grouped[x_col], grouped['mean'], color=colors[cond],
                    label=labels.get(cond, cond), linewidth=1.5)
            ax.fill_between(grouped[x_col],
                            grouped['mean'] - grouped['std'],
                            grouped['mean'] + grouped['std'],
                            color=colors[cond], alpha=0.2)
        else:
            ax.plot(subset[x_col], subset[y_col], color=colors[cond],
                    label=labels.get(cond, cond), linewidth=1.5, marker='o', markersize=3)
    if show_baseline:
        ax.axhline(y=RANDOM_BASELINE, linestyle='--', color='gray', alpha=0.7, linewidth=1.5)
    ax.set_xlabel('Epoch' if x_col == 'epoch' else 'Death Number')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(ylim)
    ax.set_xlim(0, plot_df[x_col].max())
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    sns.despine()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_accuracy(df, conditions, output_path, title, labels, colors):
    available = [c for c in conditions if c in df['condition'].unique()]
    if not available:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for cond in available:
        subset = df[df['condition'] == cond]
        if not subset.empty:
            ax.plot(subset['epoch'], subset['acc'], color=colors[cond],
                    label=labels.get(cond, cond), linewidth=1.5)
    ax.axhline(y=RANDOM_BASELINE, linestyle='--', color='gray', alpha=0.7, linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, df['epoch'].max())
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    sns.despine()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_accuracy_with_seeds(df, conditions, output_path, title, labels, colors):
    """Plot accuracy with mean line and std shading when multiple seeds exist."""
    available = [c for c in conditions if c in df['condition'].unique()]
    if not available:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for cond in available:
        subset = df[df['condition'] == cond]
        if subset.empty:
            continue
        if has_multiple_seeds(subset, cond):
            grouped = subset.groupby('epoch')['acc'].agg(['mean', 'std']).reset_index()
            ax.plot(grouped['epoch'], grouped['mean'], color=colors[cond],
                    label=labels.get(cond, cond), linewidth=1.5)
            ax.fill_between(grouped['epoch'],
                            grouped['mean'] - grouped['std'],
                            grouped['mean'] + grouped['std'],
                            color=colors[cond], alpha=0.2)
        else:
            ax.plot(subset['epoch'], subset['acc'], color=colors[cond],
                    label=labels.get(cond, cond), linewidth=1.5)
    ax.axhline(y=RANDOM_BASELINE, linestyle='--', color='gray', alpha=0.7, linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, df['epoch'].max())
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    sns.despine()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_shared_figures(acc_df, snap_df, conditions, output_dir, labels, colors):
    output_dir.mkdir(parents=True, exist_ok=True)
    if not acc_df.empty:
        plot_accuracy(acc_df, conditions, output_dir / 'accuracy_by_epoch.png', 'Test Accuracy vs Epoch', labels, colors)
    if not snap_df.empty:
        plot_line(snap_df, conditions, 'death_number', 'crossgen_accuracy', output_dir / 'crossgen_accuracy_by_death.png', 'Cross-Gen Accuracy (vs Gen 1) by Death', 'Accuracy', labels, colors)
        plot_line(snap_df, conditions, 'death_number', 'crossgen_similarity', output_dir / 'crossgen_similarity_by_death.png', 'Cross-Gen Similarity (vs Gen 1) by Death', 'Similarity', labels, colors)
        plot_line(snap_df, conditions, 'death_number', 'crossgen_prev_accuracy', output_dir / 'crossgen_prev_accuracy_by_death.png', 'Cross-Gen Accuracy (vs Prev Gen) by Death', 'Accuracy', labels, colors)
        plot_line(snap_df, conditions, 'death_number', 'crossgen_prev_similarity', output_dir / 'crossgen_prev_similarity_by_death.png', 'Cross-Gen Similarity (vs Prev Gen) by Death', 'Similarity', labels, colors)


def generate_version_figures(acc_df, snap_df, conditions, output_dir, labels, colors):
    output_dir.mkdir(parents=True, exist_ok=True)
    if not acc_df.empty:
        plot_accuracy(acc_df, conditions, output_dir / 'accuracy_by_epoch.png', 'Test Accuracy vs Epoch', labels, colors)
    if not snap_df.empty:
        plot_line(snap_df, conditions, 'death_number', 'crossgen_accuracy', output_dir / 'crossgen_accuracy_by_death.png', 'Cross-Gen Accuracy (vs Gen 1) by Death', 'Accuracy', labels, colors)
        plot_line(snap_df, conditions, 'death_number', 'crossgen_similarity', output_dir / 'crossgen_similarity_by_death.png', 'Cross-Gen Similarity (vs Gen 1) by Death', 'Similarity', labels, colors)
        plot_line(snap_df, conditions, 'death_number', 'crossgen_prev_accuracy', output_dir / 'crossgen_prev_accuracy_by_death.png', 'Cross-Gen Accuracy (vs Prev Gen) by Death', 'Accuracy', labels, colors)
        plot_line(snap_df, conditions, 'death_number', 'crossgen_prev_similarity', output_dir / 'crossgen_prev_similarity_by_death.png', 'Cross-Gen Similarity (vs Prev Gen) by Death', 'Similarity', labels, colors)
        for metric, cfg in LANGUAGE_METRICS.items():
            plot_line(snap_df, conditions, 'death_number', metric,
                      output_dir / f'{metric}_by_death.png',
                      f"{cfg['title']} by Death", cfg['ylabel'],
                      labels, colors, ylim=cfg['ylim'], show_baseline=False)


def generate_version_figures_with_seeds(acc_df, snap_df, conditions, output_dir, labels, colors):
    """Generate version figures with seed shading support."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if not acc_df.empty:
        plot_accuracy_with_seeds(acc_df, conditions, output_dir / 'accuracy_by_epoch.png', 'Test Accuracy vs Epoch', labels, colors)
    if not snap_df.empty:
        plot_line_with_seeds(snap_df, conditions, 'death_number', 'crossgen_accuracy', output_dir / 'crossgen_accuracy_by_death.png', 'Cross-Gen Accuracy (vs Gen 1) by Death', 'Accuracy', labels, colors)
        plot_line_with_seeds(snap_df, conditions, 'death_number', 'crossgen_similarity', output_dir / 'crossgen_similarity_by_death.png', 'Cross-Gen Similarity (vs Gen 1) by Death', 'Similarity', labels, colors)
        plot_line_with_seeds(snap_df, conditions, 'death_number', 'crossgen_prev_accuracy', output_dir / 'crossgen_prev_accuracy_by_death.png', 'Cross-Gen Accuracy (vs Prev Gen) by Death', 'Accuracy', labels, colors)
        plot_line_with_seeds(snap_df, conditions, 'death_number', 'crossgen_prev_similarity', output_dir / 'crossgen_prev_similarity_by_death.png', 'Cross-Gen Similarity (vs Prev Gen) by Death', 'Similarity', labels, colors)
        for metric, cfg in LANGUAGE_METRICS.items():
            plot_line_with_seeds(snap_df, conditions, 'death_number', metric,
                                 output_dir / f'{metric}_by_death.png',
                                 f"{cfg['title']} by Death", cfg['ylabel'],
                                 labels, colors, ylim=cfg['ylim'], show_baseline=False)


def plot_crossgen_combined(df, conditions, output_path, labels, colors):
    """Plot cross-gen accuracy and similarity (vs Gen 1) side by side, per death."""
    if df.empty:
        return
    available = [c for c in conditions if c in df['condition'].unique()]
    if not available:
        return
    plot_df = df[df['death_number'] >= 0].copy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    handles, legend_labels = [], []
    x_max = plot_df['death_number'].max()

    for cond in available:
        subset = plot_df[plot_df['condition'] == cond].dropna(subset=['crossgen_accuracy'])
        if subset.empty:
            continue
        if has_multiple_seeds(subset, cond):
            grouped = subset.groupby('death_number')['crossgen_accuracy'].agg(['mean', 'std']).reset_index()
            line, = ax1.plot(grouped['death_number'], grouped['mean'], color=colors[cond],
                             label=labels.get(cond, cond), linewidth=1.5)
            ax1.fill_between(grouped['death_number'],
                             grouped['mean'] - grouped['std'],
                             grouped['mean'] + grouped['std'],
                             color=colors[cond], alpha=0.2)
        else:
            line, = ax1.plot(subset['death_number'], subset['crossgen_accuracy'], color=colors[cond],
                             label=labels.get(cond, cond), linewidth=1.5)
        handles.append(line)
        legend_labels.append(labels.get(cond, cond))

    ax1.axhline(y=RANDOM_BASELINE, linestyle='--', color='gray', alpha=0.7, linewidth=1.5, zorder=0)
    ax1.set_xlabel('Death Number')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('(a) Cross-Generational Accuracy')
    ax1.set_ylim(0, 1.05)
    ax1.set_xlim(0, x_max)
    ax1.grid(True, alpha=0.3)

    for cond in available:
        subset = plot_df[plot_df['condition'] == cond].dropna(subset=['crossgen_similarity'])
        if subset.empty:
            continue
        if has_multiple_seeds(subset, cond):
            grouped = subset.groupby('death_number')['crossgen_similarity'].agg(['mean', 'std']).reset_index()
            ax2.plot(grouped['death_number'], grouped['mean'], color=colors[cond], linewidth=1.5)
            ax2.fill_between(grouped['death_number'],
                             grouped['mean'] - grouped['std'],
                             grouped['mean'] + grouped['std'],
                             color=colors[cond], alpha=0.2)
        else:
            ax2.plot(subset['death_number'], subset['crossgen_similarity'], color=colors[cond], linewidth=1.5)

    ax2.axhline(y=RANDOM_BASELINE, linestyle='--', color='gray', alpha=0.7, linewidth=1.5, zorder=0)
    ax2.set_xlabel('Death Number')
    ax2.set_ylabel('Similarity')
    ax2.set_title('(b) Cross-Generational Similarity')
    ax2.set_ylim(0, 1.05)
    ax2.set_xlim(0, x_max)
    ax2.grid(True, alpha=0.3)

    fig.legend(handles, legend_labels, loc='lower center', ncol=len(legend_labels),
               fontsize=10, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    sns.despine()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_acc_langsim_combined(acc_df, snap_df, conditions, output_path, labels, colors):
    """Plot test accuracy and language similarity side by side, over epochs."""
    if acc_df.empty and snap_df.empty:
        return
    available = [c for c in conditions if c in acc_df['condition'].unique() or c in snap_df['condition'].unique()]
    if not available:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    handles, legend_labels = [], []
    x_max = max(acc_df['epoch'].max() if not acc_df.empty else 0,
                snap_df['epoch'].max() if not snap_df.empty else 0)

    # Panel (a): Test Accuracy
    for cond in available:
        subset = acc_df[acc_df['condition'] == cond]
        if subset.empty:
            continue
        if has_multiple_seeds(subset, cond):
            grouped = subset.groupby('epoch')['acc'].agg(['mean', 'std']).reset_index()
            line, = ax1.plot(grouped['epoch'], grouped['mean'], color=colors[cond],
                             label=labels.get(cond, cond), linewidth=1.5)
            ax1.fill_between(grouped['epoch'],
                             grouped['mean'] - grouped['std'],
                             grouped['mean'] + grouped['std'],
                             color=colors[cond], alpha=0.2)
        else:
            line, = ax1.plot(subset['epoch'], subset['acc'], color=colors[cond],
                             label=labels.get(cond, cond), linewidth=1.5)
        handles.append(line)
        legend_labels.append(labels.get(cond, cond))

    ax1.axhline(y=RANDOM_BASELINE, linestyle='--', color='gray', alpha=0.7, linewidth=1.5, zorder=0)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('(a) Test Accuracy')
    ax1.set_ylim(0, 1.05)
    ax1.set_xlim(0, x_max)
    ax1.grid(True, alpha=0.3)

    # Panel (b): Language Similarity
    for cond in available:
        subset = snap_df[snap_df['condition'] == cond].dropna(subset=['language_similarity'])
        if subset.empty:
            continue
        if has_multiple_seeds(subset, cond):
            grouped = subset.groupby('epoch')['language_similarity'].agg(['mean', 'std']).reset_index()
            ax2.plot(grouped['epoch'], grouped['mean'], color=colors[cond], linewidth=1.5)
            ax2.fill_between(grouped['epoch'],
                             grouped['mean'] - grouped['std'],
                             grouped['mean'] + grouped['std'],
                             color=colors[cond], alpha=0.2)
        else:
            ax2.plot(subset['epoch'], subset['language_similarity'], color=colors[cond], linewidth=1.5)

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Similarity')
    ax2.set_title('(b) Language Similarity')
    ax2.set_ylim(0, 1.05)
    ax2.set_xlim(0, x_max)
    ax2.grid(True, alpha=0.3)

    fig.legend(handles, legend_labels, loc='lower center', ncol=len(legend_labels),
               fontsize=10, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    sns.despine()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_crossgen_prev_combined(df, conditions, output_path, labels, colors):
    """Plot cross-gen accuracy and similarity (vs Previous) side by side, per generation."""
    if df.empty:
        return
    available = [c for c in conditions if c in df['condition'].unique()]
    if not available:
        return
    plot_df = df[df['death_number'] >= N_AGENTS].copy()
    if plot_df.empty:
        return
    plot_df['generation'] = plot_df['death_number'] // N_AGENTS + 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    handles, legend_labels = [], []
    max_gen = int(plot_df['generation'].max())

    for cond in available:
        subset = plot_df[plot_df['condition'] == cond].dropna(subset=['crossgen_prev_accuracy'])
        if subset.empty:
            continue
        if has_multiple_seeds(subset, cond):
            grouped = subset.groupby('generation')['crossgen_prev_accuracy'].agg(['mean', 'std']).reset_index()
            line, = ax1.plot(grouped['generation'], grouped['mean'], color=colors[cond],
                             label=labels.get(cond, cond), linewidth=1.5, marker='o', markersize=4)
            ax1.fill_between(grouped['generation'],
                             grouped['mean'] - grouped['std'],
                             grouped['mean'] + grouped['std'],
                             color=colors[cond], alpha=0.2)
        else:
            grouped = subset.groupby('generation')['crossgen_prev_accuracy'].mean().reset_index()
            line, = ax1.plot(grouped['generation'], grouped['crossgen_prev_accuracy'], color=colors[cond],
                             label=labels.get(cond, cond), linewidth=1.5, marker='o', markersize=4)
        handles.append(line)
        legend_labels.append(labels.get(cond, cond))

    ax1.axhline(y=RANDOM_BASELINE, linestyle='--', color='gray', alpha=0.7, linewidth=1.5, zorder=0)
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('(a) Cross-Generational Accuracy (vs Previous)')
    ax1.set_ylim(0, 1.05)
    ax1.set_xlim(2, max_gen)
    ax1.xaxis.set_major_locator(MultipleLocator(1))
    ax1.grid(True, alpha=0.3)

    for cond in available:
        subset = plot_df[plot_df['condition'] == cond].dropna(subset=['crossgen_prev_similarity'])
        if subset.empty:
            continue
        if has_multiple_seeds(subset, cond):
            grouped = subset.groupby('generation')['crossgen_prev_similarity'].agg(['mean', 'std']).reset_index()
            ax2.plot(grouped['generation'], grouped['mean'], color=colors[cond], linewidth=1.5, marker='o', markersize=4)
            ax2.fill_between(grouped['generation'],
                             grouped['mean'] - grouped['std'],
                             grouped['mean'] + grouped['std'],
                             color=colors[cond], alpha=0.2)
        else:
            grouped = subset.groupby('generation')['crossgen_prev_similarity'].mean().reset_index()
            ax2.plot(grouped['generation'], grouped['crossgen_prev_similarity'], color=colors[cond],
                     linewidth=1.5, marker='o', markersize=4)

    ax2.axhline(y=RANDOM_BASELINE, linestyle='--', color='gray', alpha=0.7, linewidth=1.5, zorder=0)
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Similarity')
    ax2.set_title('(b) Cross-Generational Similarity (vs Previous)')
    ax2.set_ylim(0, 1.05)
    ax2.set_xlim(2, max_gen)
    ax2.xaxis.set_major_locator(MultipleLocator(1))
    ax2.grid(True, alpha=0.3)

    fig.legend(handles, legend_labels, loc='lower center', ncol=len(legend_labels),
               fontsize=10, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    sns.despine()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def version_has_seeds(version, conditions):
    """Check if any condition in this version has seed subdirectories."""
    for cond in conditions:
        cond_dir = EXP_DIR / cond
        if not cond_dir.exists():
            continue
        seed_dirs = find_seed_dirs(cond_dir)
        if len(seed_dirs) > 1:
            return True
    return False


def print_summary(snap_df, acc_df):
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nFinal Test Accuracy:")
    for cond in SHARED_CONDITIONS:
        subset = acc_df[acc_df['condition'] == cond]
        if subset.empty:
            continue
        final_acc = subset[subset['epoch'] == subset['epoch'].max()]['acc'].values
        if len(final_acc) > 0:
            print(f"  {SHARED_LABELS.get(cond, cond):20s}: {final_acc[0]*100:.1f}%")

    print("\nFinal Cross-Gen Metrics (vs Gen 1):")
    for cond in SHARED_CONDITIONS:
        subset = snap_df[snap_df['condition'] == cond]
        if subset.empty:
            continue
        final = subset[subset['death_number'] == subset['death_number'].max()]
        if final.empty:
            continue
        cg_acc = final['crossgen_accuracy'].values[0] if 'crossgen_accuracy' in final.columns else float('nan')
        cg_sim = final['crossgen_similarity'].values[0] if 'crossgen_similarity' in final.columns else float('nan')
        print(f"  {SHARED_LABELS.get(cond, cond):20s}: acc={cg_acc:.3f}, sim={cg_sim:.3f}")


if __name__ == "__main__":
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)

    print("=" * 80)
    print("EXPERIMENT 5: PLASTICITY CONDITIONS COMPARISON")
    print("=" * 80)

    print(f"\nLoading data from {EXP_DIR}...")
    snap_df, acc_df = load_all_runs()
    print(f"  Snapshots: {len(snap_df)}, Accuracy records: {len(acc_df)}")

    if snap_df.empty and acc_df.empty:
        print("No data found.")
        exit(1)

    found = sorted(set(snap_df['condition'].unique()) | set(acc_df['condition'].unique()))
    print(f"  Conditions found ({len(found)}): {found}")

    output_dir = EXP_DIR / "figures"

    print("\nGenerating shared figures (baseline, adults, children, age-based)...")
    generate_shared_figures(acc_df, snap_df, SHARED_CONDITIONS, output_dir, SHARED_LABELS, SHARED_COLORS)
    print(f"  Saved to {output_dir}")

    for version, conditions in VERSION_GROUPS.items():
        if version_has_seeds(version, conditions):
            print(f"\nGenerating version figures with seeds ({version})...")
            snap_df_seeds, acc_df_seeds = load_runs_with_seeds(conditions)
            n_seeds = snap_df_seeds['seed'].nunique() if 'seed' in snap_df_seeds.columns else 1
            print(f"  Loaded {len(snap_df_seeds)} snapshots, {len(acc_df_seeds)} accuracy records ({n_seeds} seeds)")
            generate_version_figures_with_seeds(acc_df_seeds, snap_df_seeds, conditions, output_dir / version, CONDITION_LABELS, COLORS_VERSION)
            # Generate combined figures for v6
            if version == 'v6':
                plot_acc_langsim_combined(acc_df_seeds, snap_df_seeds, conditions, output_dir / version / 'acc_langsim_combined.png', CONDITION_LABELS, COLORS_VERSION)
                plot_crossgen_combined(snap_df_seeds, conditions, output_dir / version / 'crossgen_combined.png', CONDITION_LABELS, COLORS_VERSION)
                plot_crossgen_prev_combined(snap_df_seeds, conditions, output_dir / version / 'crossgen_prev_combined.png', CONDITION_LABELS, COLORS_VERSION)
        else:
            print(f"\nGenerating version figures ({version})...")
            generate_version_figures(acc_df, snap_df, conditions, output_dir / version, CONDITION_LABELS, COLORS_VERSION)
        print(f"  Saved to {output_dir / version}")

    print_summary(snap_df, acc_df)
    print("=" * 80)
