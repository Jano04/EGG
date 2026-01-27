"""
Visualization script for Experiment 3: Plasticity study.

Compares plasticity conditions:
- adults: Low plasticity (temp=0.1, lr=0.0001) for all agents
- fixed (children): High plasticity (temp=1.0, lr=0.001) for all agents
- early/mid/late: Age-based plasticity with different critical points
- none: Baseline from exp2 (no plasticity manipulation)
"""
from pathlib import Path
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np

NUM_GENERATIONS = 5
RANDOM_BASELINE = 1 / 6
PALETTE = 'Set2'

EXPERIMENT_DIR = Path(__file__).parent.parent.parent.parent / "experiments" / "exp3_plasticity"
EXP2_POPULATION_DIR = Path(__file__).parent.parent.parent.parent / "experiments" / "exp2_turnover" / "population"

CONDITIONS = ['adults', 'fixed', 'early', 'mid', 'late']
CONDITION_LABELS = {
    'none': 'Baseline',
    'adults': 'Adults',
    'fixed': 'Children',
    'early': 'Early CP',
    'mid': 'Mid CP',
    'late': 'Late CP',
}
CONDITION_ORDER = ['none', 'adults', 'fixed', 'early', 'mid', 'late']


def load_training_metrics(run_dir: Path) -> pd.DataFrame:
    metrics_file = run_dir / 'training_metrics.csv'
    if metrics_file.exists():
        return pd.read_csv(metrics_file)
    return pd.DataFrame()


def load_snapshot_metrics(run_dir: Path) -> pd.DataFrame:
    metrics_file = run_dir / 'snapshot_metrics.csv'
    if metrics_file.exists():
        return pd.read_csv(metrics_file)
    return pd.DataFrame()


def truncate_to_generations(df: pd.DataFrame, n_agents: int) -> pd.DataFrame:
    max_deaths = (NUM_GENERATIONS - 1) * n_agents
    return df[df['death_number'] <= max_deaths].copy()


def load_all_runs():
    snapshot_frames = []
    acc_frames = []

    for condition in CONDITIONS:
        condition_dir = EXPERIMENT_DIR / condition
        if not condition_dir.exists():
            continue

        for seed_dir in sorted(condition_dir.iterdir()):
            if not seed_dir.is_dir():
                continue

            opts_file = seed_dir / 'opts.json'
            if not opts_file.exists():
                continue

            with open(opts_file) as f:
                opts = json.load(f)

            seed = opts.get('random_seed', 0)
            n_agents = opts.get('n_agents', 16)
            run_id = f"{condition}_{seed_dir.name}"

            snapshot_df = load_snapshot_metrics(seed_dir)
            if not snapshot_df.empty:
                snapshot_df = truncate_to_generations(snapshot_df, n_agents)
                snapshot_df['condition'] = condition
                snapshot_df['seed'] = seed
                snapshot_df['n_agents'] = n_agents
                snapshot_df['run_id'] = run_id
                snapshot_df['generation'] = snapshot_df['death_number'] / n_agents + 1
                snapshot_frames.append(snapshot_df)

            training_df = load_training_metrics(seed_dir)
            if not training_df.empty:
                test_df = training_df[training_df['mode'] == 'test'].copy()
                if not test_df.empty:
                    max_epoch = (NUM_GENERATIONS - 1) * n_agents * opts.get('kill_epoch', 10)
                    test_df = test_df[test_df['epoch'] <= max_epoch]
                    test_df['condition'] = condition
                    test_df['seed'] = seed
                    test_df['run_id'] = run_id
                    acc_frames.append(test_df[['epoch', 'acc', 'condition', 'seed', 'run_id']])

    n16_dir = EXP2_POPULATION_DIR / 'n16'
    if n16_dir.exists():
        for seed_dir in sorted(n16_dir.iterdir()):
            if not seed_dir.is_dir():
                continue

            opts_file = seed_dir / 'opts.json'
            if not opts_file.exists():
                continue

            with open(opts_file) as f:
                opts = json.load(f)

            seed = opts.get('random_seed', 0)
            n_agents = opts.get('n_agents', 16)
            run_id = f"none_{seed_dir.name}"

            snapshot_df = load_snapshot_metrics(seed_dir)
            if not snapshot_df.empty:
                warmup_row = snapshot_df[snapshot_df['death_number'] == 0]
                if not warmup_row.empty:
                    warmup_epoch = warmup_row['epoch'].iloc[0]
                    snapshot_df = snapshot_df[snapshot_df['death_number'] > 0].copy()
                    snapshot_df['epoch'] = snapshot_df['epoch'] - warmup_epoch
                    snapshot_df['death_number'] = snapshot_df['death_number'] - 1

                snapshot_df = truncate_to_generations(snapshot_df, n_agents)
                snapshot_df['condition'] = 'none'
                snapshot_df['seed'] = seed
                snapshot_df['n_agents'] = n_agents
                snapshot_df['run_id'] = run_id
                snapshot_df['generation'] = snapshot_df['death_number'] / n_agents + 1
                snapshot_frames.append(snapshot_df)

            training_df = load_training_metrics(seed_dir)
            if not training_df.empty:
                test_df = training_df[training_df['mode'] == 'test'].copy()
                if not test_df.empty:
                    snapshot_check = load_snapshot_metrics(seed_dir)
                    if not snapshot_check.empty:
                        warmup_row = snapshot_check[snapshot_check['death_number'] == 0]
                        if not warmup_row.empty:
                            warmup_epoch = warmup_row['epoch'].iloc[0]
                            test_df = test_df[test_df['epoch'] >= warmup_epoch].copy()
                            test_df['epoch'] = test_df['epoch'] - warmup_epoch
                    max_epoch = (NUM_GENERATIONS - 1) * n_agents * opts.get('kill_epoch', 10)
                    test_df = test_df[test_df['epoch'] <= max_epoch]
                    test_df['condition'] = 'none'
                    test_df['seed'] = seed
                    test_df['run_id'] = run_id
                    acc_frames.append(test_df[['epoch', 'acc', 'condition', 'seed', 'run_id']])

    snapshot_all = pd.concat(snapshot_frames, ignore_index=True) if snapshot_frames else pd.DataFrame()
    acc_all = pd.concat(acc_frames, ignore_index=True) if acc_frames else pd.DataFrame()

    return snapshot_all, acc_all


def plot_accuracy_langsim(acc_df, snapshot_df, output_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    conditions = [c for c in CONDITION_ORDER if c in acc_df['condition'].unique() or c in snapshot_df['condition'].unique()]
    palette = sns.color_palette(PALETTE, n_colors=len(conditions))

    if not acc_df.empty:
        x_max = acc_df['epoch'].max()

        for i, cond in enumerate(conditions):
            subset = acc_df[acc_df['condition'] == cond].dropna(subset=['acc'])
            if subset.empty:
                continue
            grouped = subset.groupby('epoch')['acc'].agg(['mean', 'std']).reset_index()
            grouped['mean'] = grouped['mean'].rolling(window=10, min_periods=1).mean()
            grouped['std'] = grouped['std'].rolling(window=10, min_periods=1).mean()

            label = CONDITION_LABELS.get(cond, cond)
            ax1.plot(grouped['epoch'], grouped['mean'], color=palette[i], label=label, linewidth=1.5)
            ax1.fill_between(grouped['epoch'],
                            grouped['mean'] - grouped['std'],
                            grouped['mean'] + grouped['std'],
                            color=palette[i], alpha=0.2)

        ax1.axhline(y=RANDOM_BASELINE, linestyle='--', color='gray', alpha=0.7, linewidth=1.5, zorder=0)

        ax1.set_title('(a) Test Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1.05)
        ax1.set_xlim(0, x_max)
        ax1.grid(True, alpha=0.3)

    if not snapshot_df.empty:
        x_max = snapshot_df['epoch'].max()

        for i, cond in enumerate(conditions):
            subset = snapshot_df[snapshot_df['condition'] == cond].dropna(subset=['language_similarity'])
            if subset.empty:
                continue
            grouped = subset.groupby('epoch')['language_similarity'].agg(['mean', 'std']).reset_index()

            ax2.plot(grouped['epoch'], grouped['mean'], color=palette[i], linewidth=1.5)
            ax2.fill_between(grouped['epoch'],
                            grouped['mean'] - grouped['std'],
                            grouped['mean'] + grouped['std'],
                            color=palette[i], alpha=0.2)

        ax2.set_title('(b) Language Similarity')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Similarity')
        ax2.set_ylim(0, 1.05)
        ax2.set_xlim(0, x_max)
        ax2.grid(True, alpha=0.3)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(conditions),
               bbox_to_anchor=(0.5, -0.02), fontsize=10)

    sns.despine()
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_crossgen(snapshot_df, output_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    if snapshot_df.empty:
        print("  No snapshot data for crossgen figure")
        plt.close()
        return

    conditions = [c for c in CONDITION_ORDER if c in snapshot_df['condition'].unique()]
    palette = sns.color_palette(PALETTE, n_colors=len(conditions))

    gen_df = snapshot_df[snapshot_df['generation'] == snapshot_df['generation'].astype(int)].copy()
    x_max = NUM_GENERATIONS

    if 'crossgen_accuracy' in gen_df.columns:
        for i, cond in enumerate(conditions):
            subset = gen_df[gen_df['condition'] == cond].dropna(subset=['crossgen_accuracy'])
            if subset.empty:
                continue
            grouped = subset.groupby('generation')['crossgen_accuracy'].agg(['mean', 'std']).reset_index()

            label = CONDITION_LABELS.get(cond, cond)
            ax1.plot(grouped['generation'], grouped['mean'], color=palette[i], label=label, linewidth=1.5, marker='o', markersize=4)
            ax1.fill_between(grouped['generation'],
                            grouped['mean'] - grouped['std'],
                            grouped['mean'] + grouped['std'],
                            color=palette[i], alpha=0.2)

        ax1.axhline(y=RANDOM_BASELINE, linestyle='--', color='gray', alpha=0.7, linewidth=1.5, zorder=0)

        ax1.set_title('(a) Cross-Generational Accuracy')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1.05)
        ax1.set_xlim(1, x_max)
        ax1.xaxis.set_major_locator(MultipleLocator(1))
        ax1.grid(True, alpha=0.3)

    if 'crossgen_similarity' in gen_df.columns:
        for i, cond in enumerate(conditions):
            subset = gen_df[gen_df['condition'] == cond].dropna(subset=['crossgen_similarity'])
            if subset.empty:
                continue
            grouped = subset.groupby('generation')['crossgen_similarity'].agg(['mean', 'std']).reset_index()

            ax2.plot(grouped['generation'], grouped['mean'], color=palette[i], linewidth=1.5, marker='o', markersize=4)
            ax2.fill_between(grouped['generation'],
                            grouped['mean'] - grouped['std'],
                            grouped['mean'] + grouped['std'],
                            color=palette[i], alpha=0.2)

        ax2.set_title('(b) Cross-Generational Similarity')
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Similarity')
        ax2.set_ylim(0, 1.05)
        ax2.set_xlim(1, x_max)
        ax2.xaxis.set_major_locator(MultipleLocator(1))
        ax2.grid(True, alpha=0.3)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(conditions),
               bbox_to_anchor=(0.5, -0.02), fontsize=10)

    sns.despine()
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def compute_summary_stats(snapshot_df, acc_df):
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS FOR THESIS")
    print("=" * 70)

    print(f"\n--- Final Test Accuracy (generation {NUM_GENERATIONS}) ---")
    for cond in CONDITION_ORDER:
        if cond not in acc_df['condition'].unique():
            continue
        subset = acc_df[acc_df['condition'] == cond]
        final_accs = []
        for run_id in subset['run_id'].unique():
            run_data = subset[subset['run_id'] == run_id]
            final_acc = run_data[run_data['epoch'] == run_data['epoch'].max()]['acc'].values
            if len(final_acc) > 0:
                final_accs.append(final_acc[0])
        if final_accs:
            print(f"  {CONDITION_LABELS.get(cond, cond):15s}: {np.mean(final_accs)*100:.1f}% +/- {np.std(final_accs)*100:.1f}%")

    print(f"\n--- Final Cross-Gen Metrics (generation {NUM_GENERATIONS}) ---")
    final_df = snapshot_df.loc[snapshot_df.groupby('run_id')['death_number'].idxmax()]

    for cond in CONDITION_ORDER:
        if cond not in final_df['condition'].unique():
            continue
        subset = final_df[final_df['condition'] == cond]
        cg_acc = subset['crossgen_accuracy'].mean()
        cg_acc_std = subset['crossgen_accuracy'].std()
        cg_sim = subset['crossgen_similarity'].mean()
        cg_sim_std = subset['crossgen_similarity'].std()
        lang_sim = subset['language_similarity'].mean()
        lang_sim_std = subset['language_similarity'].std()
        print(f"  {CONDITION_LABELS.get(cond, cond):15s}: acc={cg_acc:.3f}+/-{cg_acc_std:.3f}, sim={cg_sim:.3f}+/-{cg_sim_std:.3f}, lang={lang_sim:.3f}+/-{lang_sim_std:.3f}")


if __name__ == "__main__":
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)

    output_dir = EXPERIMENT_DIR / "figures"
    output_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print(f"EXPERIMENT 3: PLASTICITY STUDY ({NUM_GENERATIONS} generations)")
    print("=" * 70)

    print(f"\nLoading runs from {EXPERIMENT_DIR}...")
    snapshot_df, acc_df = load_all_runs()

    print(f"Loaded: {len(snapshot_df)} snapshot records, {len(acc_df)} accuracy records")

    if snapshot_df.empty and acc_df.empty:
        print("No data found for plasticity experiment.")
        exit(1)

    if not snapshot_df.empty:
        print(f"Conditions found: {sorted(snapshot_df['condition'].unique())}")
        print(f"Runs per condition: {snapshot_df.groupby('condition')['run_id'].nunique().to_dict()}")

    print("\n--- Generating Thesis Figures ---")

    print("Generating exp3_accuracy_langsim.png...")
    plot_accuracy_langsim(acc_df, snapshot_df, output_dir / 'exp3_accuracy_langsim.png')
    print("  Saved")

    print("Generating exp3_crossgen.png...")
    plot_crossgen(snapshot_df, output_dir / 'exp3_crossgen.png')
    print("  Saved")

    compute_summary_stats(snapshot_df, acc_df)

    print(f"\n" + "=" * 70)
    print(f"All figures saved to {output_dir}")
    print("=" * 70)
