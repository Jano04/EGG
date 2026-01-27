"""
Visualization script for Experiment 2: Turnover studies.

Sub-experiments:
A) Rate study: N=10 agents, varying turnover rate (k=1,2,5,10,20)
B) Population study: Varying population size (N=2,4,8,16), fixed k=10
"""
from pathlib import Path
import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# =============================================================================
# CONFIGURATION
# =============================================================================

NUM_GENERATIONS = 5
RANDOM_BASELINE = 1 / 6
PALETTE = 'flare'

EXP2_DIR = Path(__file__).parent.parent.parent.parent / "experiments" / "exp2_turnover"
RATE_DIR = EXP2_DIR / "rate"
POPULATION_DIR = EXP2_DIR / "population"

CROSSGEN_METRICS = {
    'crossgen_accuracy': {'ylabel': 'Accuracy', 'ylim': (0, 1.05), 'baseline': True, 'title': 'Cross-Gen Accuracy (vs Gen1)'},
    'crossgen_similarity': {'ylabel': 'Similarity', 'ylim': (0, 1.05), 'baseline': False, 'title': 'Cross-Gen Similarity (vs Gen1)'},
}

CROSSGEN_PREV_METRICS = {
    'crossgen_prev_accuracy': {'ylabel': 'Accuracy', 'ylim': (0, 1.05), 'baseline': True, 'title': 'Cross-Gen Accuracy (vs Previous)'},
    'crossgen_prev_similarity': {'ylabel': 'Similarity', 'ylim': (0, 1.05), 'baseline': False, 'title': 'Cross-Gen Similarity (vs Previous)'},
}

LANGUAGE_METRICS = {
    'vocab_usage': {'ylabel': 'Vocab Usage', 'ylim': (0, 1.05), 'baseline': False, 'title': 'Vocabulary Usage'},
    'language_similarity': {'ylabel': 'Language Similarity', 'ylim': (0, 1.05), 'baseline': False, 'title': 'Language Similarity'},
    'message_length_mean': {'ylabel': 'Message Length', 'ylim': None, 'baseline': False, 'title': 'Message Length'},
    'topographic_similarity': {'ylabel': 'Topographic Similarity', 'ylim': (-0.1, 1.05), 'baseline': False, 'title': 'Topographic Similarity'},
    'posdis': {'ylabel': 'Posdis', 'ylim': None, 'baseline': False, 'title': 'Positional Disentanglement'},
    'bosdis': {'ylabel': 'Bosdis', 'ylim': None, 'baseline': False, 'title': 'Bag-of-Symbols Disentanglement'},
}

ACCURACY_METRIC = {
    'acc': {'ylabel': 'Accuracy', 'ylim': (0, 1.05), 'baseline': True, 'title': 'Test Accuracy'},
}

DRIFT_RATE_METRICS = {
    'drift_per_epoch': {
        'ylabel': 'Accuracy Drift Rate (per epoch)',
        'ylim': None,
        'baseline': False,
        'title': 'Language Drift Rate (Accuracy)'
    },
    'sim_drift_per_epoch': {
        'ylabel': 'Similarity Drift Rate (per epoch)',
        'ylim': None,
        'baseline': False,
        'title': 'Language Drift Rate (Similarity)'
    },
    'drift_per_death': {
        'ylabel': 'Accuracy Drift Rate (per death)',
        'ylim': None,
        'baseline': False,
        'title': 'Language Drift per Turnover Event (Accuracy)'
    },
    'sim_drift_per_death': {
        'ylabel': 'Similarity Drift Rate (per death)',
        'ylim': None,
        'baseline': False,
        'title': 'Language Drift per Turnover Event (Similarity)'
    },
}

X_AXIS_LABELS = {
    'generation': 'Generation',
    'death_number': 'Death Number',
    'epoch': 'Epoch',
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def truncate_to_generations(df: pd.DataFrame, n_agents: int) -> pd.DataFrame:
    max_deaths = (NUM_GENERATIONS - 1) * n_agents
    return df[df['death_number'] <= max_deaths].copy()


def compute_generation(death_number: int, n_agents: int) -> int:
    return death_number // n_agents + 1


def add_drift_per_epoch(df: pd.DataFrame) -> pd.DataFrame:
    """Add drift rate columns computed per-run from Gen 1 reference."""
    result_frames = []
    group_cols = [c for c in ['kill_epoch', 'n_agents', 'seed'] if c in df.columns]

    for _, run_df in df.groupby(group_cols):
        run_df = run_df.copy()
        gen1 = run_df[run_df['death_number'] == 0]

        if gen1.empty:
            run_df['drift_per_epoch'] = np.nan
            run_df['sim_drift_per_epoch'] = np.nan
            run_df['drift_per_death'] = np.nan
            run_df['sim_drift_per_death'] = np.nan
            result_frames.append(run_df)
            continue

        gen1_epoch = gen1['epoch'].iloc[0]
        gen1_acc = gen1['crossgen_accuracy'].iloc[0] if pd.notna(gen1['crossgen_accuracy'].iloc[0]) else np.nan
        gen1_sim = gen1['crossgen_similarity'].iloc[0] if pd.notna(gen1['crossgen_similarity'].iloc[0]) else np.nan

        def calc_drift(row, gen1_val, col_name, normalizer):
            if row['death_number'] == 0 or pd.isna(row[col_name]) or pd.isna(gen1_val):
                return np.nan
            divisor = normalizer(row)
            return (gen1_val - row[col_name]) / divisor if divisor > 0 else np.nan

        epoch_norm = lambda r: r['epoch'] - gen1_epoch
        death_norm = lambda r: r['death_number']

        run_df['drift_per_epoch'] = run_df.apply(lambda r: calc_drift(r, gen1_acc, 'crossgen_accuracy', epoch_norm), axis=1)
        run_df['sim_drift_per_epoch'] = run_df.apply(lambda r: calc_drift(r, gen1_sim, 'crossgen_similarity', epoch_norm), axis=1)
        run_df['drift_per_death'] = run_df.apply(lambda r: calc_drift(r, gen1_acc, 'crossgen_accuracy', death_norm), axis=1)
        run_df['sim_drift_per_death'] = run_df.apply(lambda r: calc_drift(r, gen1_sim, 'crossgen_similarity', death_norm), axis=1)
        result_frames.append(run_df)

    return pd.concat(result_frames, ignore_index=True) if result_frames else df


def load_snapshot_metrics(run_dir: Path) -> pd.DataFrame:
    f = run_dir / 'snapshot_metrics.csv'
    return pd.read_csv(f) if f.exists() else pd.DataFrame()


def load_training_metrics(run_dir: Path) -> pd.DataFrame:
    f = run_dir / 'training_metrics.csv'
    return pd.read_csv(f) if f.exists() else pd.DataFrame()


def get_metric_folder(metric: str) -> str:
    if metric == 'acc':
        return 'accuracy'
    if metric == 'message_length_mean':
        return 'message_length'
    return metric


# =============================================================================
# GENERIC PLOTTING FUNCTION
# =============================================================================

def plot_metric(df: pd.DataFrame, metric: str, x_col: str, group_col: str,
                group_label: str, ylabel: str, title: str, output_path: Path,
                ylim: tuple = None, show_baseline: bool = False) -> bool:
    if metric not in df.columns or df[metric].isna().all():
        print(f"    Skipping {output_path.name}: no data for {metric}")
        return False

    fig, ax = plt.subplots(figsize=(10, 6))
    groups = sorted(df[group_col].unique(), reverse=(group_col == 'kill_epoch'))
    colors = sns.color_palette(PALETTE, n_colors=len(groups))
    x_max = df[x_col].max()

    for i, g in enumerate(groups):
        subset = df[df[group_col] == g].dropna(subset=[metric])
        if subset.empty:
            continue
        grouped = subset.groupby(x_col)[metric].agg(['mean', 'std']).reset_index()
        ax.plot(grouped[x_col], grouped['mean'], color=colors[i],
                label=f'{group_label}={g}', linewidth=1.5)
        ax.fill_between(grouped[x_col], grouped['mean'] - grouped['std'],
                        grouped['mean'] + grouped['std'], color=colors[i], alpha=0.2)

    if show_baseline:
        ax.axhline(y=RANDOM_BASELINE, linestyle='--', color='gray', alpha=0.7, linewidth=1.5)

    ax.set_xlabel(X_AXIS_LABELS.get(x_col, x_col))
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(ylim)
    ax.set_xlim(left=0 if x_col != 'generation' else 1, right=x_max)
    if x_col == 'generation':
        ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.legend(title=group_label, bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    sns.despine()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return True


def plot_warmup_boxplot(df: pd.DataFrame, output_path: Path) -> bool:
    gen1 = df[df['death_number'] == 0].copy()
    if gen1.empty:
        print(f"    Skipping {output_path.name}: no warmup data")
        return False

    fig, ax = plt.subplots(figsize=(8, 5))
    n_values = sorted(gen1['n_agents'].unique())
    colors = sns.color_palette(PALETTE, n_colors=len(n_values))

    data = [gen1[gen1['n_agents'] == n]['epoch'].values for n in n_values]
    bp = ax.boxplot(data, patch_artist=True, labels=[f'N={n}' for n in n_values])
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('Epochs to 95% Accuracy')
    ax.set_xlabel('Population Size')
    ax.set_title('Warmup Duration by Population Size')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    sns.despine()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return True


def plot_drift_barplot(df: pd.DataFrame, metric: str, group_col: str,
                       group_label: str, ylabel: str, title: str,
                       output_path: Path, n_agents: int) -> bool:
    """Plot final drift rate as bar chart comparing conditions."""
    final_deaths = (NUM_GENERATIONS - 1) * n_agents
    final_df = df[df['death_number'] == final_deaths].copy()

    if final_df.empty or metric not in final_df.columns:
        print(f"    Skipping {output_path.name}: no data at generation {NUM_GENERATIONS}")
        return False

    final_df = final_df.dropna(subset=[metric])
    if final_df.empty:
        print(f"    Skipping {output_path.name}: no valid {metric} values")
        return False

    fig, ax = plt.subplots(figsize=(8, 5))
    groups = sorted(final_df[group_col].unique(), reverse=(group_col == 'kill_epoch'))
    colors = sns.color_palette(PALETTE, n_colors=len(groups))

    means = [final_df[final_df[group_col] == g][metric].mean() for g in groups]
    stds = [final_df[final_df[group_col] == g][metric].std() for g in groups]
    x_pos = range(len(groups))

    ax.bar(x_pos, means, yerr=stds, capsize=5, color=colors, alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{group_label}={g}' for g in groups])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    sns.despine()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return True


def plot_population_drift_barplot(df: pd.DataFrame, metric: str, ylabel: str,
                                   title: str, output_path: Path) -> bool:
    """Plot final drift rate bar chart for population study (varying n_agents)."""
    final_data = []
    for n in df['n_agents'].unique():
        final_deaths = (NUM_GENERATIONS - 1) * int(n)
        subset = df[(df['n_agents'] == n) & (df['death_number'] == final_deaths)]
        if not subset.empty:
            final_data.append(subset)

    if not final_data:
        print(f"    Skipping {output_path.name}: no final generation data")
        return False

    final_df = pd.concat(final_data, ignore_index=True)
    final_df = final_df.dropna(subset=[metric])
    if final_df.empty:
        return False

    fig, ax = plt.subplots(figsize=(8, 5))
    groups = sorted(final_df['n_agents'].unique())
    colors = sns.color_palette(PALETTE, n_colors=len(groups))

    means = [final_df[final_df['n_agents'] == n][metric].mean() for n in groups]
    stds = [final_df[final_df['n_agents'] == n][metric].std() for n in groups]
    x_pos = range(len(groups))

    ax.bar(x_pos, means, yerr=stds, capsize=5, color=colors, alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'N={int(n)}' for n in groups])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    sns.despine()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    return True


# =============================================================================
# DATA LOADING
# =============================================================================

def load_rate_runs():
    runs_dir = RATE_DIR / 'runs'
    snapshot_frames, training_frames = [], []
    n_agents = 10

    if not runs_dir.exists():
        return pd.DataFrame(), pd.DataFrame()

    for run_dir in sorted(runs_dir.iterdir()):
        opts_file = run_dir / 'opts.json'
        if not run_dir.is_dir() or not opts_file.exists():
            continue

        with open(opts_file) as f:
            opts = json.load(f)
        k = opts.get('kill_epoch')
        seed = opts.get('random_seed', 0)
        if k == 0 or k is None:
            continue

        snap_df = load_snapshot_metrics(run_dir)
        train_df = load_training_metrics(run_dir)

        if not snap_df.empty:
            snap_df = truncate_to_generations(snap_df, n_agents)
            snap_df['kill_epoch'] = k
            snap_df['seed'] = seed
            snap_df['n_agents'] = n_agents
            snap_df['generation'] = snap_df['death_number'].apply(lambda d: compute_generation(d, n_agents))

            if not train_df.empty:
                test_acc = train_df[train_df['mode'] == 'test'][['epoch', 'acc']]
                snap_df = snap_df.merge(test_acc, on='epoch', how='left')

            snapshot_frames.append(snap_df)

        if not train_df.empty:
            test_df = train_df[train_df['mode'] == 'test'].copy()
            test_df['kill_epoch'] = k
            test_df['seed'] = seed
            training_frames.append(test_df[['epoch', 'acc', 'kill_epoch', 'seed']])

    snapshot_all = pd.concat(snapshot_frames, ignore_index=True) if snapshot_frames else pd.DataFrame()
    if not snapshot_all.empty:
        snapshot_all = add_drift_per_epoch(snapshot_all)
    training_all = pd.concat(training_frames, ignore_index=True) if training_frames else pd.DataFrame()
    return snapshot_all, training_all


def load_population_runs():
    snapshot_frames, training_frames = [], []

    if not POPULATION_DIR.exists():
        return pd.DataFrame(), pd.DataFrame()

    for pop_dir in sorted(POPULATION_DIR.iterdir()):
        if not pop_dir.is_dir() or not pop_dir.name.startswith('n'):
            continue

        for seed_dir in sorted(pop_dir.iterdir()):
            opts_file = seed_dir / 'opts.json'
            if not seed_dir.is_dir() or not opts_file.exists():
                continue

            with open(opts_file) as f:
                opts = json.load(f)
            n_agents = opts.get('n_agents')
            k = opts.get('kill_epoch', 10)
            seed = opts.get('random_seed', 0)

            snap_df = load_snapshot_metrics(seed_dir)
            train_df = load_training_metrics(seed_dir)

            if not snap_df.empty:
                snap_df = truncate_to_generations(snap_df, n_agents)
                snap_df['n_agents'] = n_agents
                snap_df['kill_epoch'] = k
                snap_df['seed'] = seed
                snap_df['generation'] = snap_df['death_number'].apply(lambda d: compute_generation(d, n_agents))

                if not train_df.empty:
                    test_acc = train_df[train_df['mode'] == 'test'][['epoch', 'acc']]
                    snap_df = snap_df.merge(test_acc, on='epoch', how='left')

                snapshot_frames.append(snap_df)

            if not train_df.empty:
                test_df = train_df[train_df['mode'] == 'test'].copy()
                test_df['n_agents'] = n_agents
                test_df['seed'] = seed
                training_frames.append(test_df[['epoch', 'acc', 'n_agents', 'seed']])

    snapshot_all = pd.concat(snapshot_frames, ignore_index=True) if snapshot_frames else pd.DataFrame()
    if not snapshot_all.empty:
        snapshot_all = add_drift_per_epoch(snapshot_all)
    training_all = pd.concat(training_frames, ignore_index=True) if training_frames else pd.DataFrame()
    return snapshot_all, training_all


# =============================================================================
# THESIS COMBINED FIGURES
# =============================================================================

def plot_thesis_warmup_accuracy(pop_snap_df: pd.DataFrame, rate_train_df: pd.DataFrame,
                                 output_path: Path) -> bool:
    """Figure 1: (a) Warmup duration boxplot, (b) Test accuracy over epochs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # (a) Warmup duration by population size
    gen1 = pop_snap_df[pop_snap_df['death_number'] == 0].copy()
    if not gen1.empty:
        n_values = sorted(gen1['n_agents'].unique())
        colors = sns.color_palette(PALETTE, n_colors=len(n_values))
        data = [gen1[gen1['n_agents'] == n]['epoch'].values for n in n_values]
        bp = ax1.boxplot(data, patch_artist=True, labels=[f'N={int(n)}' for n in n_values])
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax1.set_ylabel('Epochs to 95% Accuracy')
        ax1.set_xlabel('Population Size')
        ax1.set_title('(a) Warmup Duration')
        ax1.grid(True, alpha=0.3, axis='y')

    # (b) Test accuracy over epochs (rate study)
    if not rate_train_df.empty:
        k_values = sorted(rate_train_df['kill_epoch'].unique(), reverse=True)
        colors = sns.color_palette(PALETTE, n_colors=len(k_values))
        max_epoch = rate_train_df['epoch'].max()

        for i, k in enumerate(k_values):
            subset = rate_train_df[rate_train_df['kill_epoch'] == k]
            grouped = subset.groupby('epoch')['acc'].agg(['mean', 'std']).reset_index()
            ax2.plot(grouped['epoch'], grouped['mean'], color=colors[i],
                    label=f'k={k}', linewidth=1.5)
            ax2.fill_between(grouped['epoch'],
                            grouped['mean'] - grouped['std'],
                            grouped['mean'] + grouped['std'],
                            color=colors[i], alpha=0.2)

        ax2.axhline(y=RANDOM_BASELINE, linestyle='--', color='gray', alpha=0.7, linewidth=1.5)
        ax2.annotate('Random', xy=(max_epoch, RANDOM_BASELINE), xytext=(5, 0),
                    textcoords='offset points', va='center', fontsize=9, color='gray')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('(b) Test Accuracy (Rate Study)')
        ax2.set_ylim(0, 1.05)
        ax2.set_xlim(0, max_epoch)
        ax2.grid(True, alpha=0.3)

    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels),
               bbox_to_anchor=(0.5, -0.02), fontsize=9, title='Turnover Rate')

    sns.despine()
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {output_path.name}")
    return True


def plot_thesis_rate_crossgen(snap_df: pd.DataFrame, output_path: Path) -> bool:
    """Figure 2: (a) Cross-gen accuracy, (b) Cross-gen similarity by generation."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    if snap_df.empty:
        plt.close()
        return False

    k_values = sorted(snap_df['kill_epoch'].unique(), reverse=True)
    colors = sns.color_palette(PALETTE, n_colors=len(k_values))
    x_max = snap_df['generation'].max()

    # (a) Cross-generational accuracy
    for i, k in enumerate(k_values):
        subset = snap_df[snap_df['kill_epoch'] == k].dropna(subset=['crossgen_accuracy'])
        if subset.empty:
            continue
        grouped = subset.groupby('generation')['crossgen_accuracy'].agg(['mean', 'std']).reset_index()
        ax1.plot(grouped['generation'], grouped['mean'], color=colors[i],
                label=f'k={k}', linewidth=1.5)
        ax1.fill_between(grouped['generation'],
                        grouped['mean'] - grouped['std'],
                        grouped['mean'] + grouped['std'],
                        color=colors[i], alpha=0.2)

    ax1.axhline(y=RANDOM_BASELINE, linestyle='--', color='gray', alpha=0.7, linewidth=1.5)
    ax1.annotate('Random', xy=(x_max, RANDOM_BASELINE), xytext=(5, 0),
                textcoords='offset points', va='center', fontsize=9, color='gray')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('(a) Cross-Generational Accuracy')
    ax1.set_ylim(0, 1.05)
    ax1.set_xlim(1, x_max)
    ax1.xaxis.set_major_locator(MultipleLocator(1))
    ax1.grid(True, alpha=0.3)

    # (b) Cross-generational similarity
    for i, k in enumerate(k_values):
        subset = snap_df[snap_df['kill_epoch'] == k].dropna(subset=['crossgen_similarity'])
        if subset.empty:
            continue
        grouped = subset.groupby('generation')['crossgen_similarity'].agg(['mean', 'std']).reset_index()
        ax2.plot(grouped['generation'], grouped['mean'], color=colors[i], linewidth=1.5)
        ax2.fill_between(grouped['generation'],
                        grouped['mean'] - grouped['std'],
                        grouped['mean'] + grouped['std'],
                        color=colors[i], alpha=0.2)

    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Similarity')
    ax2.set_title('(b) Cross-Generational Similarity')
    ax2.set_ylim(0, 1.05)
    ax2.set_xlim(1, x_max)
    ax2.xaxis.set_major_locator(MultipleLocator(1))
    ax2.grid(True, alpha=0.3)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels),
               bbox_to_anchor=(0.5, -0.02), fontsize=9, title='Turnover Rate')

    sns.despine()
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {output_path.name}")
    return True


def plot_thesis_pop_crossgen(snap_df: pd.DataFrame, output_path: Path) -> bool:
    """Figure 3: (a) Cross-gen accuracy, (b) Cross-gen similarity by generation."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    if snap_df.empty:
        plt.close()
        return False

    n_values = sorted(snap_df['n_agents'].unique())
    colors = sns.color_palette(PALETTE, n_colors=len(n_values))
    x_max = snap_df['generation'].max()

    # (a) Cross-generational accuracy
    for i, n in enumerate(n_values):
        subset = snap_df[snap_df['n_agents'] == n].dropna(subset=['crossgen_accuracy'])
        if subset.empty:
            continue
        grouped = subset.groupby('generation')['crossgen_accuracy'].agg(['mean', 'std']).reset_index()
        ax1.plot(grouped['generation'], grouped['mean'], color=colors[i],
                label=f'N={int(n)}', linewidth=1.5)
        ax1.fill_between(grouped['generation'],
                        grouped['mean'] - grouped['std'],
                        grouped['mean'] + grouped['std'],
                        color=colors[i], alpha=0.2)

    ax1.axhline(y=RANDOM_BASELINE, linestyle='--', color='gray', alpha=0.7, linewidth=1.5)
    ax1.annotate('Random', xy=(x_max, RANDOM_BASELINE), xytext=(5, 0),
                textcoords='offset points', va='center', fontsize=9, color='gray')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('(a) Cross-Generational Accuracy')
    ax1.set_ylim(0, 1.05)
    ax1.set_xlim(1, x_max)
    ax1.xaxis.set_major_locator(MultipleLocator(1))
    ax1.grid(True, alpha=0.3)

    # (b) Cross-generational similarity
    for i, n in enumerate(n_values):
        subset = snap_df[snap_df['n_agents'] == n].dropna(subset=['crossgen_similarity'])
        if subset.empty:
            continue
        grouped = subset.groupby('generation')['crossgen_similarity'].agg(['mean', 'std']).reset_index()
        ax2.plot(grouped['generation'], grouped['mean'], color=colors[i], linewidth=1.5)
        ax2.fill_between(grouped['generation'],
                        grouped['mean'] - grouped['std'],
                        grouped['mean'] + grouped['std'],
                        color=colors[i], alpha=0.2)

    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Similarity')
    ax2.set_title('(b) Cross-Generational Similarity')
    ax2.set_ylim(0, 1.05)
    ax2.set_xlim(1, x_max)
    ax2.xaxis.set_major_locator(MultipleLocator(1))
    ax2.grid(True, alpha=0.3)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels),
               bbox_to_anchor=(0.5, -0.02), fontsize=9, title='Population Size')

    sns.despine()
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {output_path.name}")
    return True


def generate_thesis_figures():
    """Generate combined figures for thesis Section 5.3."""
    output_dir = EXP2_DIR / "thesis_figures"
    output_dir.mkdir(exist_ok=True)

    print("Loading data for thesis figures...")
    rate_snap_df, rate_train_df = load_rate_runs()
    pop_snap_df, _ = load_population_runs()
    print(f"  Rate: {len(rate_snap_df)} snapshots, {len(rate_train_df)} training")
    print(f"  Population: {len(pop_snap_df)} snapshots")

    print("\nGenerating thesis figures...")
    plot_thesis_warmup_accuracy(pop_snap_df, rate_train_df, output_dir / 'exp2_warmup_accuracy.png')
    plot_thesis_rate_crossgen(rate_snap_df, output_dir / 'exp2_rate_crossgen.png')
    plot_thesis_pop_crossgen(pop_snap_df, output_dir / 'exp2_pop_crossgen.png')

    print(f"\nThesis figures saved to {output_dir}")


# =============================================================================
# INDIVIDUAL FIGURE GENERATION (for detailed analysis)
# =============================================================================

def generate_rate_figures():
    output_dir = RATE_DIR / "figures"
    output_dir.mkdir(exist_ok=True)

    print("Loading rate study data...")
    snap_df, train_df = load_rate_runs()
    print(f"  Snapshots: {len(snap_df)}, Training: {len(train_df)}")

    if snap_df.empty and train_df.empty:
        print("No data found.")
        return

    count = 0

    for metric, cfg in CROSSGEN_METRICS.items():
        folder = output_dir / get_metric_folder(metric)
        folder.mkdir(exist_ok=True)
        for x_col, suffix in [('generation', 'by_gen'), ('death_number', 'by_death'), ('epoch', 'by_epoch')]:
            fname = f"{metric}_{suffix}.png"
            title = f"{cfg['title']} vs {X_AXIS_LABELS[x_col]}"
            if plot_metric(snap_df, metric, x_col, 'kill_epoch', 'k',
                          cfg['ylabel'], title, folder / fname, cfg['ylim'], cfg['baseline']):
                print(f"    {get_metric_folder(metric)}/{fname}")
                count += 1

    for metric, cfg in CROSSGEN_PREV_METRICS.items():
        folder = output_dir / get_metric_folder(metric)
        folder.mkdir(exist_ok=True)
        fname = f"{metric}_by_gen.png"
        title = f"{cfg['title']} vs Generation"
        if plot_metric(snap_df, metric, 'generation', 'kill_epoch', 'k',
                      cfg['ylabel'], title, folder / fname, cfg['ylim'], cfg['baseline']):
            print(f"    {get_metric_folder(metric)}/{fname}")
            count += 1

    for metric, cfg in LANGUAGE_METRICS.items():
        folder = output_dir / get_metric_folder(metric)
        folder.mkdir(exist_ok=True)
        for x_col, suffix in [('generation', 'by_gen'), ('death_number', 'by_death'), ('epoch', 'by_epoch')]:
            fname = f"{metric}_{suffix}.png"
            title = f"{cfg['title']} vs {X_AXIS_LABELS[x_col]}"
            if plot_metric(snap_df, metric, x_col, 'kill_epoch', 'k',
                          cfg['ylabel'], title, folder / fname, cfg['ylim'], cfg['baseline']):
                print(f"    {get_metric_folder(metric)}/{fname}")
                count += 1

    folder = output_dir / 'accuracy'
    folder.mkdir(exist_ok=True)

    for x_col, suffix in [('generation', 'by_gen'), ('death_number', 'by_death')]:
        fname = f"acc_{suffix}.png"
        title = f"Test Accuracy vs {X_AXIS_LABELS[x_col]}"
        if plot_metric(snap_df, 'acc', x_col, 'kill_epoch', 'k',
                      'Accuracy', title, folder / fname, (0, 1.05), True):
            print(f"    accuracy/{fname}")
            count += 1

    if not train_df.empty:
        fname = "acc_by_epoch.png"
        title = "Test Accuracy vs Epoch"
        if plot_metric(train_df, 'acc', 'epoch', 'kill_epoch', 'k',
                      'Accuracy', title, folder / fname, (0, 1.05), True):
            print(f"    accuracy/{fname}")
            count += 1

    # Drift rate plots
    for metric, cfg in DRIFT_RATE_METRICS.items():
        folder = output_dir / 'drift_rate'
        folder.mkdir(exist_ok=True)

        fname = f"{metric}_by_gen.png"
        title = f"{cfg['title']} vs Generation"
        if plot_metric(snap_df, metric, 'generation', 'kill_epoch', 'k',
                      cfg['ylabel'], title, folder / fname, cfg['ylim'], cfg['baseline']):
            print(f"    drift_rate/{fname}")
            count += 1

        fname = f"{metric}_final_bar.png"
        title = f"Final {cfg['title']} (Gen {NUM_GENERATIONS})"
        if plot_drift_barplot(snap_df, metric, 'kill_epoch', 'k',
                             cfg['ylabel'], title, folder / fname, n_agents=10):
            print(f"    drift_rate/{fname}")
            count += 1

    print(f"Rate study: {count} plots saved")


def generate_population_figures():
    output_dir = POPULATION_DIR / "figures"
    output_dir.mkdir(exist_ok=True)

    print("Loading population study data...")
    snap_df, train_df = load_population_runs()
    print(f"  Snapshots: {len(snap_df)}, Training: {len(train_df)}")

    if snap_df.empty and train_df.empty:
        print("No data found.")
        return

    count = 0

    for metric, cfg in CROSSGEN_METRICS.items():
        folder = output_dir / get_metric_folder(metric)
        folder.mkdir(exist_ok=True)
        for x_col, suffix in [('generation', 'by_gen'), ('death_number', 'by_death'), ('epoch', 'by_epoch')]:
            fname = f"{metric}_{suffix}.png"
            title = f"{cfg['title']} vs {X_AXIS_LABELS[x_col]}"
            if plot_metric(snap_df, metric, x_col, 'n_agents', 'N',
                          cfg['ylabel'], title, folder / fname, cfg['ylim'], cfg['baseline']):
                print(f"    {get_metric_folder(metric)}/{fname}")
                count += 1

    for metric, cfg in CROSSGEN_PREV_METRICS.items():
        folder = output_dir / get_metric_folder(metric)
        folder.mkdir(exist_ok=True)
        fname = f"{metric}_by_gen.png"
        title = f"{cfg['title']} vs Generation"
        if plot_metric(snap_df, metric, 'generation', 'n_agents', 'N',
                      cfg['ylabel'], title, folder / fname, cfg['ylim'], cfg['baseline']):
            print(f"    {get_metric_folder(metric)}/{fname}")
            count += 1

    for metric, cfg in LANGUAGE_METRICS.items():
        folder = output_dir / get_metric_folder(metric)
        folder.mkdir(exist_ok=True)
        for x_col, suffix in [('generation', 'by_gen'), ('death_number', 'by_death'), ('epoch', 'by_epoch')]:
            fname = f"{metric}_{suffix}.png"
            title = f"{cfg['title']} vs {X_AXIS_LABELS[x_col]}"
            if plot_metric(snap_df, metric, x_col, 'n_agents', 'N',
                          cfg['ylabel'], title, folder / fname, cfg['ylim'], cfg['baseline']):
                print(f"    {get_metric_folder(metric)}/{fname}")
                count += 1

    folder = output_dir / 'accuracy'
    folder.mkdir(exist_ok=True)

    for x_col, suffix in [('generation', 'by_gen'), ('death_number', 'by_death')]:
        fname = f"acc_{suffix}.png"
        title = f"Test Accuracy vs {X_AXIS_LABELS[x_col]}"
        if plot_metric(snap_df, 'acc', x_col, 'n_agents', 'N',
                      'Accuracy', title, folder / fname, (0, 1.05), True):
            print(f"    accuracy/{fname}")
            count += 1

    if not train_df.empty:
        fname = "acc_by_epoch.png"
        title = "Test Accuracy vs Epoch"
        if plot_metric(train_df, 'acc', 'epoch', 'n_agents', 'N',
                      'Accuracy', title, folder / fname, (0, 1.05), True):
            print(f"    accuracy/{fname}")
            count += 1

    folder = output_dir / 'warmup'
    folder.mkdir(exist_ok=True)
    fname = "warmup_duration.png"
    if plot_warmup_boxplot(snap_df, folder / fname):
        print(f"    warmup/{fname}")
        count += 1

    # Drift rate plots
    for metric, cfg in DRIFT_RATE_METRICS.items():
        folder = output_dir / 'drift_rate'
        folder.mkdir(exist_ok=True)

        fname = f"{metric}_by_gen.png"
        title = f"{cfg['title']} vs Generation"
        if plot_metric(snap_df, metric, 'generation', 'n_agents', 'N',
                      cfg['ylabel'], title, folder / fname, cfg['ylim'], cfg['baseline']):
            print(f"    drift_rate/{fname}")
            count += 1

        fname = f"{metric}_final_bar.png"
        title = f"Final {cfg['title']} (Gen {NUM_GENERATIONS})"
        if plot_population_drift_barplot(snap_df, metric, cfg['ylabel'], title, folder / fname):
            print(f"    drift_rate/{fname}")
            count += 1

    print(f"Population study: {count} plots saved")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)

    print("=" * 60)
    print("EXPERIMENT 2: TURNOVER STUDIES")
    print(f"Truncated to {NUM_GENERATIONS} generations")
    print("=" * 60)

    print("\n" + "-" * 60)
    print("THESIS FIGURES")
    print("-" * 60)
    generate_thesis_figures()

    print("\n" + "-" * 60)
    print("RATE STUDY (N=10, varying k)")
    print("-" * 60)
    generate_rate_figures()

    print("\n" + "-" * 60)
    print("POPULATION STUDY (varying N, k=10)")
    print("-" * 60)
    generate_population_figures()

    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)
