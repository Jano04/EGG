"""
Post-hoc analysis: compute cross-gen metrics from saved snapshots.

Usage:
    # Single experiment:
    python -m egg.zoo.aging.analysis experiments/exp2_turnover/rate/runs/k5_s0

    # All experiments under a directory (recursive):
    python -m egg.zoo.aging.analysis experiments/exp2_turnover --recursive --similarity-only

Cross-gen metrics compare snapshots at generation boundaries (death_number % n_agents == 0):
- crossgen_* : Compare Gen N to Gen 1 (cumulative drift from founding)
- crossgen_prev_* : Compare Gen N to Gen N-1 (drift velocity per generation)
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import List

import pandas as pd
import torch

from egg.zoo.aging.metrics import Snapshot, compute_cross_gen_similarity


CROSSGEN_COLUMNS = ['crossgen_accuracy', 'crossgen_similarity']
CROSSGEN_PREV_COLUMNS = ['crossgen_prev_accuracy', 'crossgen_prev_similarity']


def find_experiment_dirs(root: Path) -> List[Path]:
    """Find all directories containing a 'snapshots' subdirectory."""
    exp_dirs = []
    for snapshots_dir in root.rglob("snapshots"):
        if snapshots_dir.is_dir() and list(snapshots_dir.glob("snapshot_*.pkl")):
            exp_dirs.append(snapshots_dir.parent)
    return sorted(exp_dirs)


def load_snapshots(snapshot_dir: Path) -> List[Snapshot]:
    """Load all snapshots from directory, sorted by death_number."""
    snapshots = []
    for filepath in sorted(snapshot_dir.glob("snapshot_*.pkl")):
        with open(filepath, "rb") as f:
            snap = pickle.load(f)
            snapshots.append(snap)

    snapshots.sort(key=lambda s: s.death_number)
    return snapshots


def load_or_create_metrics_csv(exp_dir: Path, snapshots: List[Snapshot]) -> pd.DataFrame:
    """Load existing metrics CSV or create empty one from snapshots."""
    metrics_file = exp_dir / 'snapshot_metrics.csv'

    if metrics_file.exists():
        df = pd.read_csv(metrics_file)
        for col in CROSSGEN_PREV_COLUMNS:
            if col not in df.columns:
                df[col] = None
        return df

    rows = []
    for snap in snapshots:
        row = {
            'epoch': snap.epoch,
            'death_number': snap.death_number,
            'vocab_usage': '',
            'message_length_mean': '',
            'message_length_std': '',
            'language_similarity': '',
            'topographic_similarity': '',
            'posdis': '',
            'bosdis': '',
        }
        for col in CROSSGEN_COLUMNS + CROSSGEN_PREV_COLUMNS:
            row[col] = ''
        rows.append(row)

    return pd.DataFrame(rows)


def get_generation_snapshots(snapshots: List[Snapshot], n_agents: int) -> List[Snapshot]:
    """Filter snapshots to generation boundaries only (death_number % n_agents == 0)."""
    return [s for s in snapshots if s.death_number % n_agents == 0]


def compute_crossgen_similarity_from_snapshots(
    snapshots: List[Snapshot],
    df: pd.DataFrame,
    n_agents: int,
    quiet: bool = False,
) -> pd.DataFrame:
    """Compute cross-gen similarity (vs Gen1) at generation boundaries."""
    gen_snaps = get_generation_snapshots(snapshots, n_agents)
    if not gen_snaps:
        return df

    gen1 = gen_snaps[0]

    for snap in gen_snaps:
        sim_result = compute_cross_gen_similarity(gen1, snap)
        similarity = sim_result['similarity']
        gen = snap.death_number // n_agents + 1

        mask = df['death_number'] == snap.death_number
        df.loc[mask, 'crossgen_similarity'] = similarity

        if not quiet:
            print(f"  Gen {gen} (death #{snap.death_number}): similarity={similarity:.4f}")

    return df


def compute_crossgen_prev_similarity_from_snapshots(
    snapshots: List[Snapshot],
    df: pd.DataFrame,
    n_agents: int,
    quiet: bool = False,
) -> pd.DataFrame:
    """Compute cross-gen similarity (vs previous generation) at generation boundaries."""
    gen_snaps = get_generation_snapshots(snapshots, n_agents)
    if len(gen_snaps) < 2:
        return df

    for i, snap in enumerate(gen_snaps):
        if i == 0:
            continue

        prev_snap = gen_snaps[i - 1]
        sim_result = compute_cross_gen_similarity(prev_snap, snap)
        similarity = sim_result['similarity']
        gen = snap.death_number // n_agents + 1

        mask = df['death_number'] == snap.death_number
        df.loc[mask, 'crossgen_prev_similarity'] = similarity

        if not quiet:
            print(f"  Gen {gen} (death #{snap.death_number}): prev_similarity={similarity:.4f}")

    return df


def _make_loss_fn():
    """Create the loss function that computes accuracy."""
    import torch.nn.functional as F

    def loss(sender_input, message, receiver_input, receiver_output, labels, aux_input):
        acc = (receiver_output.argmax(dim=1) == labels).detach().float()
        loss_val = F.cross_entropy(receiver_output, labels, reduction="none")
        return loss_val, {"acc": acc}

    return loss


def _create_game(opts, device):
    """Create game architecture for accuracy computation."""
    from egg.zoo.aging.archs import FullAgent, MultiPairPopulationGame
    from egg.zoo.aging.features import VectorsLoader

    data_loader = VectorsLoader(
        perceptual_dimensions=opts['perceptual_dimensions'],
        n_distractors=opts['n_distractors'],
        batch_size=opts.get('batch_size', 1024),
        train_samples=opts['train_samples'],
        validation_samples=opts['validation_samples'],
        test_samples=opts['test_samples'],
        shuffle_train_data=opts.get('shuffle_train_data', False),
        seed=opts['data_seed'],
    )
    _, validation_data, _ = data_loader.get_iterators(is_distributed=False)

    total_agents = opts.get('total_agents', opts['n_agents'])
    all_agents = [
        FullAgent(
            n_features=len(opts['perceptual_dimensions']),
            sender_hidden=opts['sender_hidden'],
            receiver_hidden=opts['receiver_hidden'],
        )
        for _ in range(total_agents)
    ]

    game = MultiPairPopulationGame(
        agents=all_agents[:opts['n_agents']],
        loss_fn=_make_loss_fn(),
        vocab_size=opts['vocab_size'],
        sender_embedding=opts['sender_embedding'],
        sender_hidden=opts['sender_hidden'],
        receiver_embedding=opts['receiver_embedding'],
        receiver_hidden=opts['receiver_hidden'],
        sender_cell=opts.get('sender_cell', 'lstm'),
        receiver_cell=opts.get('receiver_cell', 'lstm'),
        max_len=opts['max_len'],
        total_agents=total_agents,
        all_agents=all_agents,
    ).to(device)

    return game, validation_data


def compute_crossgen_accuracy_from_snapshots(
    snapshots: List[Snapshot],
    df: pd.DataFrame,
    opts: dict,
    n_agents: int,
    quiet: bool = False,
) -> pd.DataFrame:
    """Compute cross-gen accuracy (vs Gen1) at generation boundaries."""
    from egg.zoo.aging.metrics import compute_cross_gen_accuracy

    gen_snaps = get_generation_snapshots(snapshots, n_agents)
    if not gen_snaps:
        return df

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    game, validation_data = _create_game(opts, device)

    gen1 = gen_snaps[0]

    for snap in gen_snaps:
        acc_result = compute_cross_gen_accuracy(gen1, snap, game, validation_data, device)
        accuracy = acc_result['mean']
        gen = snap.death_number // n_agents + 1

        mask = df['death_number'] == snap.death_number
        df.loc[mask, 'crossgen_accuracy'] = accuracy

        if not quiet:
            print(f"  Gen {gen} (death #{snap.death_number}): accuracy={accuracy:.4f}")

    return df


def compute_crossgen_prev_accuracy_from_snapshots(
    snapshots: List[Snapshot],
    df: pd.DataFrame,
    opts: dict,
    n_agents: int,
    quiet: bool = False,
) -> pd.DataFrame:
    """Compute cross-gen accuracy (vs previous generation) at generation boundaries."""
    from egg.zoo.aging.metrics import compute_cross_gen_accuracy

    gen_snaps = get_generation_snapshots(snapshots, n_agents)
    if len(gen_snaps) < 2:
        return df

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    game, validation_data = _create_game(opts, device)

    for i, snap in enumerate(gen_snaps):
        if i == 0:
            continue

        prev_snap = gen_snaps[i - 1]
        acc_result = compute_cross_gen_accuracy(prev_snap, snap, game, validation_data, device)
        accuracy = acc_result['mean']
        gen = snap.death_number // n_agents + 1

        mask = df['death_number'] == snap.death_number
        df.loc[mask, 'crossgen_prev_accuracy'] = accuracy

        if not quiet:
            print(f"  Gen {gen} (death #{snap.death_number}): prev_accuracy={accuracy:.4f}")

    return df


def process_experiment(exp_dir: Path, similarity_only: bool, prev_only: bool, quiet: bool = False) -> bool:
    """Process a single experiment directory. Returns True on success."""
    snapshot_dir = exp_dir / 'snapshots'
    metrics_file = exp_dir / 'snapshot_metrics.csv'
    opts_file = exp_dir / 'opts.json'

    if not snapshot_dir.exists():
        if not quiet:
            print(f"Error: snapshot directory not found: {snapshot_dir}")
        return False

    if not opts_file.exists():
        if not quiet:
            print(f"Error: opts.json not found: {opts_file}")
        return False

    with open(opts_file) as f:
        opts = json.load(f)
    n_agents = opts.get('n_agents', 10)

    if not quiet:
        print(f"Loading snapshots from {snapshot_dir}...")
    snapshots = load_snapshots(snapshot_dir)
    gen_snaps = get_generation_snapshots(snapshots, n_agents)
    if not quiet:
        print(f"  Found {len(snapshots)} snapshots, {len(gen_snaps)} at generation boundaries")

    if not snapshots:
        if not quiet:
            print("No snapshots found, nothing to compute.")
        return False

    df = load_or_create_metrics_csv(exp_dir, snapshots)

    if not prev_only:
        if not quiet:
            print("\nComputing cross-gen similarity (vs Gen1)...")
        df = compute_crossgen_similarity_from_snapshots(snapshots, df, n_agents, quiet)

    if not quiet:
        print("\nComputing cross-gen similarity (vs previous)...")
    df = compute_crossgen_prev_similarity_from_snapshots(snapshots, df, n_agents, quiet)

    if not similarity_only:
        if not prev_only:
            if not quiet:
                print("\nComputing cross-gen accuracy (vs Gen1)...")
            df = compute_crossgen_accuracy_from_snapshots(snapshots, df, opts, n_agents, quiet)

        if not quiet:
            print("\nComputing cross-gen accuracy (vs previous)...")
        df = compute_crossgen_prev_accuracy_from_snapshots(snapshots, df, opts, n_agents, quiet)

    if not quiet:
        print(f"\nSaving to {metrics_file}...")
    df.to_csv(metrics_file, index=False)
    if not quiet:
        print("Done.")
    return True


def main():
    parser = argparse.ArgumentParser(description='Post-hoc cross-gen analysis from snapshots')
    parser.add_argument('exp_dir', type=str, help='Experiment directory (or root for --recursive)')
    parser.add_argument('--recursive', '-r', action='store_true',
                        help='Find and process all experiments under exp_dir')
    parser.add_argument('--similarity-only', action='store_true',
                        help='Only compute similarity (skip accuracy)')
    parser.add_argument('--prev-only', action='store_true',
                        help='Only compute prev metrics (skip Gen1 comparison)')
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)

    if args.recursive:
        print(f"Finding experiments under {exp_dir}...")
        exp_dirs = find_experiment_dirs(exp_dir)
        print(f"Found {len(exp_dirs)} experiments\n")

        for i, d in enumerate(exp_dirs, 1):
            print(f"[{i}/{len(exp_dirs)}] {d}")
            process_experiment(d, args.similarity_only, args.prev_only, quiet=True)
            print("  Done.")

        print(f"\nProcessed {len(exp_dirs)} experiments.")
    else:
        if not (exp_dir / 'snapshots').exists():
            print(f"Error: snapshot directory not found: {exp_dir / 'snapshots'}")
            sys.exit(1)
        process_experiment(exp_dir, args.similarity_only, args.prev_only, quiet=False)


if __name__ == '__main__':
    main()
