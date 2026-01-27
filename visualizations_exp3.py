"""
Visualization script for Experiment 3: Plasticity conditions comparison.

Compares four plasticity conditions:
- baseline: Fixed temperature and learning rate (plasticity disabled)
- adults: Uniform low plasticity for all agents
- children: Uniform high plasticity for all agents
- age_based: Plasticity decreases with agent age
"""
from pathlib import Path
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# =============================================================================
# CONFIGURATION
# =============================================================================

N_AGENTS = 10
RANDOM_BASELINE = 1 / 6
PALETTE = 'Set2'

EXP_DIR = Path(__file__).parent.parent.parent.parent / "experiments" / "exp3_plasticity"

CONDITIONS = ['baseline', 'adults', 'children', 'age_based']
CONDITION_LABELS = {
    'baseline': 'Baseline',
    'adults': 'Adults',
    'children': 'Children',
    'age_based': 'Age-based',
}

# =============================================================================
# DATA LOADING
# =============================================================================

def find_seed_dirs(cond_dir: Path):
    seed_dirs = []
    if (cond_dir / 'snapshot_metrics.csv').exists():
        opts_file = cond_dir / 'opts.json'
        seed = 0
        if opts_file.exists():
            with open(opts_file) as f:
                seed = json.load(f).get('random_seed', 0)
        seed_dirs.append((seed, cond_dir))
    for subdir in sorted(cond_dir.iterdir()):
        if subdir.is_dir() and subdir.name.startswith('s') and subdir.name[1:].isdigit():
            if (subdir / 'snapshot_metrics.csv').exists():
                opts_file = subdir / 'opts.json'
                seed = int(subdir.name[1:])
                if opts_file.exists():
                    with open(opts_file) as f:
                        seed = json.load(f).get('random_seed', seed)
                seed_dirs.append((seed, subdir))
    return seed_dirs


def load_all_runs():
    snapshot_frames, acc_frames = [], []
    for cond in CONDITIONS:
        cond_dir = EXP_DIR / cond
        if not cond_dir.exists():
            continue
        for seed, seed_dir in find_seed_dirs(cond_dir):
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


# =============================================================================
# PLOTTING
# =============================================================================

def plot_accuracy_langsim(acc_df, snap_df, output_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    available = [c for c in CONDITIONS if c in acc_df['condition'].unique()]
    colors = sns.color_palette(PALETTE, n_colors=len(available))
    handles, labels = [], []

    for i, cond in enumerate(available):
        subset = acc_df[acc_df['condition'] == cond]
        grouped = subset.groupby('epoch')['acc'].agg(['mean', 'std']).reset_index()
        line, = ax1.plot(grouped['epoch'], grouped['mean'], color=colors[i],
                         label=CONDITION_LABELS[cond], linewidth=1.5)
        ax1.fill_between(grouped['epoch'], grouped['mean'] - grouped['std'],
                         grouped['mean'] + grouped['std'], color=colors[i], alpha=0.2)
        handles.append(line)
        labels.append(CONDITION_LABELS[cond])

    ax1.axhline(y=RANDOM_BASELINE, linestyle='--', color='gray', alpha=0.7, linewidth=1.5)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('(a) Test Accuracy')
    ax1.set_ylim(0, 1.05)
    ax1.set_xlim(0, acc_df['epoch'].max())
    ax1.grid(True, alpha=0.3)

    for i, cond in enumerate(available):
        subset = snap_df[snap_df['condition'] == cond].dropna(subset=['language_similarity'])
        if subset.empty:
            continue
        grouped = subset.groupby('epoch')['language_similarity'].agg(['mean', 'std']).reset_index()
        ax2.plot(grouped['epoch'], grouped['mean'], color=colors[i], linewidth=1.5)
        ax2.fill_between(grouped['epoch'], grouped['mean'] - grouped['std'],
                         grouped['mean'] + grouped['std'], color=colors[i], alpha=0.2)

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Similarity')
    ax2.set_title('(b) Language Similarity')
    ax2.set_ylim(0, 1.05)
    ax2.set_xlim(0, snap_df['epoch'].max())
    ax2.grid(True, alpha=0.3)

    fig.legend(handles, labels, loc='lower center', ncol=len(labels),
               bbox_to_anchor=(0.5, -0.02), fontsize=10)
    sns.despine()
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_crossgen(snap_df, output_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    plot_df = snap_df[snap_df['death_number'] >= 0].copy()
    available = [c for c in CONDITIONS if c in plot_df['condition'].unique()]
    colors = sns.color_palette(PALETTE, n_colors=len(available))
    handles, labels = [], []
    x_max = plot_df['death_number'].max()

    for i, cond in enumerate(available):
        subset = plot_df[plot_df['condition'] == cond].dropna(subset=['crossgen_accuracy'])
        if subset.empty:
            continue
        grouped = subset.groupby('death_number')['crossgen_accuracy'].agg(['mean', 'std']).reset_index()
        line, = ax1.plot(grouped['death_number'], grouped['mean'], color=colors[i],
                         label=CONDITION_LABELS[cond], linewidth=1.5)
        ax1.fill_between(grouped['death_number'], grouped['mean'] - grouped['std'],
                         grouped['mean'] + grouped['std'], color=colors[i], alpha=0.2)
        handles.append(line)
        labels.append(CONDITION_LABELS[cond])

    ax1.axhline(y=RANDOM_BASELINE, linestyle='--', color='gray', alpha=0.7, linewidth=1.5)
    ax1.set_xlabel('Death Number')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('(a) Cross-Generational Accuracy')
    ax1.set_ylim(0, 1.05)
    ax1.set_xlim(0, x_max)
    ax1.grid(True, alpha=0.3)

    for i, cond in enumerate(available):
        subset = plot_df[plot_df['condition'] == cond].dropna(subset=['crossgen_similarity'])
        if subset.empty:
            continue
        grouped = subset.groupby('death_number')['crossgen_similarity'].agg(['mean', 'std']).reset_index()
        ax2.plot(grouped['death_number'], grouped['mean'], color=colors[i], linewidth=1.5)
        ax2.fill_between(grouped['death_number'], grouped['mean'] - grouped['std'],
                         grouped['mean'] + grouped['std'], color=colors[i], alpha=0.2)

    ax2.axhline(y=RANDOM_BASELINE, linestyle='--', color='gray', alpha=0.7, linewidth=1.5)
    ax2.set_xlabel('Death Number')
    ax2.set_ylabel('Similarity')
    ax2.set_title('(b) Cross-Generational Similarity')
    ax2.set_ylim(0, 1.05)
    ax2.set_xlim(0, x_max)
    ax2.grid(True, alpha=0.3)

    fig.legend(handles, labels, loc='lower center', ncol=len(labels),
               bbox_to_anchor=(0.5, -0.02), fontsize=10)
    sns.despine()
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_crossgen_prev(snap_df, output_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    plot_df = snap_df[snap_df['death_number'] >= N_AGENTS].copy()
    plot_df['generation'] = plot_df['death_number'] // N_AGENTS + 1
    available = [c for c in CONDITIONS if c in plot_df['condition'].unique()]
    colors = sns.color_palette(PALETTE, n_colors=len(available))
    handles, labels = [], []
    max_gen = int(plot_df['generation'].max())

    for i, cond in enumerate(available):
        subset = plot_df[plot_df['condition'] == cond].dropna(subset=['crossgen_prev_accuracy'])
        if subset.empty:
            continue
        grouped = subset.groupby('generation')['crossgen_prev_accuracy'].agg(['mean', 'std']).reset_index()
        line, = ax1.plot(grouped['generation'], grouped['mean'], color=colors[i],
                         label=CONDITION_LABELS[cond], linewidth=1.5, marker='o', markersize=4)
        ax1.fill_between(grouped['generation'], grouped['mean'] - grouped['std'],
                         grouped['mean'] + grouped['std'], color=colors[i], alpha=0.2)
        handles.append(line)
        labels.append(CONDITION_LABELS[cond])

    ax1.axhline(y=RANDOM_BASELINE, linestyle='--', color='gray', alpha=0.7, linewidth=1.5)
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('(a) Cross-Generational Accuracy (vs Previous)')
    ax1.set_ylim(0, 1.05)
    ax1.set_xlim(2, max_gen)
    ax1.xaxis.set_major_locator(MultipleLocator(1))
    ax1.grid(True, alpha=0.3)

    for i, cond in enumerate(available):
        subset = plot_df[plot_df['condition'] == cond].dropna(subset=['crossgen_prev_similarity'])
        if subset.empty:
            continue
        grouped = subset.groupby('generation')['crossgen_prev_similarity'].agg(['mean', 'std']).reset_index()
        ax2.plot(grouped['generation'], grouped['mean'], color=colors[i],
                 linewidth=1.5, marker='o', markersize=4)
        ax2.fill_between(grouped['generation'], grouped['mean'] - grouped['std'],
                         grouped['mean'] + grouped['std'], color=colors[i], alpha=0.2)

    ax2.axhline(y=RANDOM_BASELINE, linestyle='--', color='gray', alpha=0.7, linewidth=1.5)
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Similarity')
    ax2.set_title('(b) Cross-Generational Similarity (vs Previous)')
    ax2.set_ylim(0, 1.05)
    ax2.set_xlim(2, max_gen)
    ax2.xaxis.set_major_locator(MultipleLocator(1))
    ax2.grid(True, alpha=0.3)

    fig.legend(handles, labels, loc='lower center', ncol=len(labels),
               bbox_to_anchor=(0.5, -0.02), fontsize=10)
    sns.despine()
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def print_summary(snap_df, acc_df):
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\nFinal Test Accuracy:")
    for cond in CONDITIONS:
        subset = acc_df[acc_df['condition'] == cond]
        if subset.empty:
            continue
        final = subset.groupby('seed').apply(lambda x: x.loc[x['epoch'].idxmax(), 'acc'])
        print(f"  {CONDITION_LABELS[cond]:12s}: {final.mean()*100:.1f}% +/- {final.std()*100:.1f}%")

    print("\nFinal Cross-Gen Metrics:")
    final_df = snap_df.loc[snap_df.groupby(['condition', 'seed'])['death_number'].idxmax()]
    for cond in CONDITIONS:
        subset = final_df[final_df['condition'] == cond]
        if subset.empty:
            continue
        cg_acc = subset['crossgen_accuracy']
        cg_sim = subset['crossgen_similarity']
        lang_sim = subset['language_similarity']
        print(f"  {CONDITION_LABELS[cond]:12s}: acc={cg_acc.mean():.3f}+/-{cg_acc.std():.3f}, "
              f"sim={cg_sim.mean():.3f}+/-{cg_sim.std():.3f}, "
              f"lang={lang_sim.mean():.3f}+/-{lang_sim.std():.3f}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)

    output_dir = EXP_DIR / "figures"
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("EXPERIMENT 3: PLASTICITY CONDITIONS")
    print("=" * 60)

    print(f"\nLoading runs from {EXP_DIR}...")
    snap_df, acc_df = load_all_runs()
    print(f"  Snapshots: {len(snap_df)}, Accuracy: {len(acc_df)}")

    if snap_df.empty and acc_df.empty:
        print("No data found.")
        exit(1)

    found = sorted(set(snap_df['condition'].unique()) | set(acc_df['condition'].unique()))
    print(f"  Conditions: {found}")

    print("\nGenerating figures...")
    if not acc_df.empty and not snap_df.empty:
        plot_accuracy_langsim(acc_df, snap_df, output_dir / 'exp3_accuracy_langsim.png')
        print("  exp3_accuracy_langsim.png")

    if not snap_df.empty:
        plot_crossgen(snap_df, output_dir / 'exp3_crossgen.png')
        print("  exp3_crossgen.png")

        if 'crossgen_prev_accuracy' in snap_df.columns:
            plot_crossgen_prev(snap_df, output_dir / 'exp3_crossgen_prev.png')
            print("  exp3_crossgen_prev.png")

    print_summary(snap_df, acc_df)
    print(f"\nFigures saved to {output_dir}")
