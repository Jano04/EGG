# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function

import argparse
from typing import List, Tuple, Dict, Optional, Any

import torch
import torch.nn.functional as F
import torch.utils.data

import egg.core as core
from egg.zoo.aging.archs import (
    FullAgent,
    MultiPairPopulationGame,
    PerAgentOptimizer,
)
from egg.zoo.aging.callbacks import (
    DistributedSamplerEpochSetter,
    OptsSaver,
    AgeTrackerCallback,
    LanguageAnalysisCallback,
    TrainingMetricsLogger,
)
from egg.zoo.aging.features import VectorsLoader


def get_params(params: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--perceptual_dimensions",
        type=str,
        default="[5, 5, 5, 5]",
        help="Number of features for every perceptual dimension",
    )

    parser.add_argument(
        "--n_distractors",
        type=int,
        default=5,
        help="Number of distractor objects for the receiver (default: 5)",
    )
    parser.add_argument(
        "--train_samples",
        type=float,
        default=10240,
        help="Number of tuples in training data (default: 10240)",
    )
    parser.add_argument(
        "--validation_samples",
        type=float,
        default=1e3,
        help="Number of tuples in validation data (default: 1e3)",
    )
    parser.add_argument(
        "--test_samples",
        type=float,
        default=1e3,
        help="Number of tuples in test data (default: 1e3)",
    )
    parser.add_argument(
        "--data_seed",
        type=int,
        default=111,
        help="Seed for random creation of train, validation and test tuples (default: 111)",
    )
    parser.add_argument(
        "--shuffle_train_data",
        action="store_true",
        default=False,
        help="Shuffle train data before every epoch (default: False)",
    )

    parser.add_argument(
        "--sender_hidden",
        type=int,
        default=128,
        help="Size of the hidden layer of Sender (default: 128)",
    )
    parser.add_argument(
        "--receiver_hidden",
        type=int,
        default=128,
        help="Size of the hidden layer of Receiver (default: 128)",
    )

    parser.add_argument(
        "--sender_embedding",
        type=int,
        default=64,
        help="Dimensionality of the embedding hidden layer for Sender (default: 64)",
    )
    parser.add_argument(
        "--receiver_embedding",
        type=int,
        default=64,
        help="Dimensionality of the embedding hidden layer for Receiver (default: 64)",
    )

    parser.add_argument(
        "--sender_cell",
        type=str,
        default="lstm",
        help="Type of the cell used for Sender {rnn, gru, lstm} (default: lstm)",
    )
    parser.add_argument(
        "--receiver_cell",
        type=str,
        default="lstm",
        help="Type of the cell used for Receiver {rnn, gru, lstm} (default: lstm)",
    )

    # Population-specific arguments
    parser.add_argument(
        "--n_agents",
        type=int,
        default=2,
        help="Number of agents in the population (default: 2)",
    )
    parser.add_argument(
        "--compute_cross_gen_online",
        action="store_true",
        default=False,
        help="Compute cross-generation accuracy/similarity during training (expensive). Default: compute post-hoc from snapshots. (default: False)",
    )

    parser.add_argument(
        "--warmup_threshold",
        type=float,
        default=0.0,
        help="Accuracy threshold (0-1) to complete warmup and start aging. 0 disables warmup. (default: 0.0)",
    )
    parser.add_argument(
        "--kill_epoch",
        type=int,
        default=0,
        help="Replace oldest agent every N epochs after aging starts. 0 disables turnover. (default: 0)",
    )
    parser.add_argument(
        "--stop_after_warmup",
        action="store_true",
        default=False,
        help="Stop training immediately after warmup completes. Use for creating warmup checkpoints. (default: False)",
    )
    parser.add_argument(
        "--death_policy",
        type=str,
        default="oldest",
        choices=['oldest', 'random', 'performance_based', 'age_weighted'],
        help="Policy for selecting which agent dies (default: oldest)",
    )
    parser.add_argument(
        "--death_age_exponent",
        type=float,
        default=1.0,
        help="Exponent for age_weighted policy: p(death) ~ age^k (default: 1.0)",
    )
    parser.add_argument(
        "--plasticity_max_age",
        type=int,
        default=100,
        help="Age at which plasticity reaches minimum. (default: 100)",
    )
    parser.add_argument(
        "--total_agents",
        type=int,
        default=None,
        help="Total agent slots for pre-allocation. If None, equals n_agents (default: None)",
    )
    parser.add_argument(
        "--enable_plasticity",
        action="store_true",
        default=False,
        help="Enable age-based plasticity (temperature and LR scaling) (default: False)",
    )
    parser.add_argument(
        "--no_plasticity",
        action="store_false",
        dest="enable_plasticity",
        help="Disable age-based plasticity",
    )
    parser.add_argument(
        "--temp_min",
        type=float,
        default=0.1,
        help="Temperature for oldest agents (default: 0.1)",
    )
    parser.add_argument(
        "--temp_max",
        type=float,
        default=1.0,
        help="Temperature for youngest agents (default: 1.0)",
    )
    parser.add_argument(
        "--lr_min",
        type=float,
        default=1e-4,
        help="Learning rate for oldest agents (default: 1e-4)",
    )
    parser.add_argument(
        "--lr_max",
        type=float,
        default=1e-3,
        help="Learning rate for youngest agents (default: 1e-3)",
    )
    parser.add_argument(
        "--plasticity_function",
        type=str,
        default="sigmoid",
        help="Plasticity function: 'sigmoid' or 'linear' (default: sigmoid)",
    )
    parser.add_argument(
        "--plasticity_steepness",
        type=float,
        default=10.0,
        help="Sigmoid steepness (default: 10.0)",
    )
    parser.add_argument(
        "--plasticity_critical_point",
        type=float,
        default=0.5,
        help="Sigmoid critical point as fraction of max_age (default: 0.5)",
    )
    parser.add_argument(
        "--snapshot_freq",
        type=int,
        default=0,
        help="Take periodic snapshots every N epochs. 0 disables. (default: 0)",
    )
    parser.add_argument(
        "--stagger_ages",
        nargs='*',
        default=None,
        help="Stagger initial ages. Use without args for auto [0,k,2k,...], or provide specific ages like --stagger_ages 0 10 20 30",
    )

    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=False,
        help="Enable CUDNN determinism for full reproducibility (slower)",
    )
    parser.add_argument(
        "--fresh_start",
        action="store_true",
        default=False,
        help="Reset epoch counter to 0 when loading checkpoint. Use for turnover experiments that load a warmup checkpoint. (default: False)",
    )
    args = core.init(parser, params)

    # Override EGG defaults with game-specific defaults (only when user didn't specify)
    game_defaults = {
        'lr': 1e-3,
        'vocab_size': 20,
        'max_len': 10,
        'batch_size': 1024,
        'n_epochs': 500,
        'checkpoint_freq': 50,
    }
    for name, value in game_defaults.items():
        if not any(p == f'--{name}' or p.startswith(f'--{name}=') for p in params):
            setattr(args, name, value)

    # Auto-set tensorboard_dir to checkpoint_dir/tensorboard if not explicitly specified
    if args.tensorboard and args.checkpoint_dir:
        if not any(p == '--tensorboard_dir' or p.startswith('--tensorboard_dir=') for p in params):
            args.tensorboard_dir = f"{args.checkpoint_dir}/tensorboard"

    check_args(args)
    print(args)

    return args


def check_args(args: argparse.Namespace) -> None:
    args.train_samples, args.validation_samples, args.test_samples = (
        int(args.train_samples),
        int(args.validation_samples),
        int(args.test_samples),
    )

    try:
        args.perceptual_dimensions = eval(args.perceptual_dimensions)
    except SyntaxError:
        print(
            "The format of the # of perceptual dimensions param is not correct. Please change it to string representing a list of int. Correct format: '[int, ..., int]' "
        )
        exit(1)

    args.n_features = len(args.perceptual_dimensions)

    # Population-specific validation
    if args.n_agents < 2:
        print("Error: n_agents must be at least 2")
        exit(1)

    # Set default total_agents if not specified
    if args.total_agents is None:
        args.total_agents = args.n_agents

    if args.total_agents < args.n_agents:
        print(f"Error: total_agents ({args.total_agents}) must be >= n_agents ({args.n_agents})")
        exit(1)

    if args.warmup_threshold < 0 or args.warmup_threshold > 1:
        print("Error: warmup_threshold must be between 0 and 1")
        exit(1)

    if args.kill_epoch < 0:
        print("Error: kill_epoch must be non-negative")
        exit(1)

    if args.plasticity_max_age <= 0:
        print("Error: plasticity_max_age must be positive")
        exit(1)


def loss(
    _sender_input: torch.Tensor,
    _message: torch.Tensor,
    _receiver_input: torch.Tensor,
    receiver_output: torch.Tensor,
    labels: torch.Tensor,
    _aux_input: Optional[Dict]
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    acc = (receiver_output.argmax(dim=1) == labels).detach().float()
    loss = F.cross_entropy(receiver_output, labels, reduction="none")
    return loss, {"acc": acc}


def main(params: List[str]) -> None:
    """
    Main training function using EGG Trainer framework.

    Unified population training for N≥2 agents using random pair sampling.
    Role alternation occurs naturally through random sampling - no explicit
    alternation mechanism needed.
    """
    opts = get_params(params)
    device = torch.device("cuda" if opts.cuda else "cpu")

    if opts.deterministic:
        import os
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if hasattr(torch, 'use_deterministic_algorithms'):
            torch.use_deterministic_algorithms(True)
        print("| Deterministic mode enabled")

    data_loader = VectorsLoader(
        perceptual_dimensions=opts.perceptual_dimensions,
        n_distractors=opts.n_distractors,
        batch_size=opts.batch_size,
        train_samples=opts.train_samples,
        validation_samples=opts.validation_samples,
        test_samples=opts.test_samples,
        shuffle_train_data=opts.shuffle_train_data,
        seed=opts.data_seed,
    )
    train_data, validation_data, test_data = data_loader.get_iterators(
        is_distributed=opts.distributed_context.is_distributed
    )

    print(f"\n{'='*70}")
    print(f"Training {opts.n_agents} agents in population")
    print(f"{'='*70}")
    print(f"| Pairs per batch: K=N={opts.n_agents}")
    if opts.warmup_threshold > 0:
        print(f"| Warmup: until {opts.warmup_threshold*100:.0f}% accuracy")
    else:
        print(f"| Warmup: disabled (aging starts immediately)")
    print(f"| Total epochs: {opts.n_epochs}")
    print(f"{'='*70}\n")

    # Create all agent slots (total_agents total, first n_agents active)
    all_agents = []
    print(f"Creating {opts.total_agents} agent slots ({opts.n_agents} active, {opts.total_agents - opts.n_agents} dormant)...")
    for i in range(opts.total_agents):
        agent = FullAgent(
            n_features=data_loader.n_features,
            sender_hidden=opts.sender_hidden,
            receiver_hidden=opts.receiver_hidden
        )
        all_agents.append(agent)
        if i < 3 or i == opts.total_agents - 1:
            status = "active" if i < opts.n_agents else "dormant"
            print(f"  Created Agent {i} ({status})")
        elif i == 3:
            print(f"  ...")

    # First n_agents are active (used for training)
    active_agents = all_agents[:opts.n_agents]

    total_params = sum(sum(p.numel() for p in agent.parameters()) for agent in all_agents)
    avg_params = total_params // opts.total_agents
    print(f"\nTotal population parameters: {total_params:,}")
    print(f"Average per agent: {avg_params:,}")
    print(f"Active agents: {opts.n_agents}, Dormant agents: {opts.total_agents - opts.n_agents}\n")

    print(f"Creating population game...")

    game = MultiPairPopulationGame(
        agents=active_agents,
        loss_fn=loss,
        vocab_size=opts.vocab_size,
        sender_embedding=opts.sender_embedding,
        sender_hidden=opts.sender_hidden,
        receiver_embedding=opts.receiver_embedding,
        receiver_hidden=opts.receiver_hidden,
        sender_cell=opts.sender_cell,
        receiver_cell=opts.receiver_cell,
        max_len=opts.max_len,
        total_agents=opts.total_agents,
        all_agents=all_agents,
    )

    # Create LanguageAnalysisCallback first so we can pass it to AgeTrackerCallback
    language_callback = LanguageAnalysisCallback(
        vocab_size=opts.vocab_size,
        n_agents=opts.n_agents,
        save_dir=f"{opts.checkpoint_dir}/snapshots" if opts.checkpoint_dir else None,
        compute_cross_gen_online=opts.compute_cross_gen_online,
        max_samples=int(opts.validation_samples),
        snapshot_freq=opts.snapshot_freq,
    )

    callbacks = [
        OptsSaver(),
        core.ConsoleLogger(print_train_loss=True, as_json=False),
        AgeTrackerCallback(opts, language_callback=language_callback),
        language_callback,
    ]

    if opts.checkpoint_dir:
        callbacks.append(TrainingMetricsLogger(f"{opts.checkpoint_dir}/training_metrics.csv"))

    if opts.distributed_context.is_distributed:
        callbacks.append(DistributedSamplerEpochSetter())

    if opts.tensorboard:
        callbacks.append(core.TensorboardLogger(writer=core.util.get_summary_writer()))

    print(f"Using learning rate: {opts.lr:.4f}")
    print(f"Using L2 regularization: 1e-5")

    optimizer = PerAgentOptimizer(game, lr=opts.lr, weight_decay=1e-5)
    game.agent_optimizer = optimizer

    print("Setting up EGG Trainer...")
    print(f"{'='*70}\n")

    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        train_data=train_data,
        validation_data=validation_data,
        callbacks=callbacks,
    )

    trainer.opts = opts

    # Reset epoch counter for fresh start (useful when loading warmup checkpoint for turnover)
    if opts.fresh_start and trainer.start_epoch > 0:
        print(f"| Fresh start: resetting epoch from {trainer.start_epoch} to 0")
        trainer.start_epoch = 0

    trainer.train(n_epochs=opts.n_epochs)

    print(f"\n{'='*70}")
    print("Training complete!")
    print(f"{'='*70}")
    print(f"Random baseline accuracy: {1 / (opts.n_distractors + 1):.3f}\n")


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
