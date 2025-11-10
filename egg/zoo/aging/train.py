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
from egg.zoo.aging.archs import FullAgent, AlternatingGame
from egg.zoo.aging.features import VectorsLoader
from egg.zoo.aging.game_callbacks import ResetAlternationCounter, DirectionalConsoleLogger


def get_params(params: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--perceptual_dimensions",
        type=str,
        default="[4, 4, 4, 4, 5]",
        help="Number of features for every perceptual dimension",
    )

    parser.add_argument(
        "--n_distractors",
        type=int,
        default=3,
        help="Number of distractor objects for the receiver (default: 3)",
    )
    parser.add_argument(
        "--train_samples",
        type=float,
        default=1e5,
        help="Number of tuples in training data (default: 1e5)",
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
        "--feature_hidden",
        type=int,
        default=50,
        help="Size of the feature encoder output (default: 50)",
    )
    parser.add_argument(
        "--sender_hidden",
        type=int,
        default=50,
        help="Size of the hidden layer of Sender (default: 50)",
    )
    parser.add_argument(
        "--receiver_hidden",
        type=int,
        default=50,
        help="Size of the hidden layer of Receiver (default: 50)",
    )

    parser.add_argument(
        "--sender_embedding",
        type=int,
        default=10,
        help="Dimensionality of the embedding hidden layer for Sender (default: 10)",
    )
    parser.add_argument(
        "--receiver_embedding",
        type=int,
        default=10,
        help="Dimensionality of the embedding hidden layer for Receiver (default: 10)",
    )

    parser.add_argument(
        "--sender_cell",
        type=str,
        default="rnn",
        help="Type of the cell used for Sender {rnn, gru, lstm} (default: rnn)",
    )
    parser.add_argument(
        "--receiver_cell",
        type=str,
        default="rnn",
        help="Type of the cell used for Receiver {rnn, gru, lstm} (default: rnn)",
    )

    parser.add_argument(
        "--sender_lr",
        type=float,
        default=1e-1,
        help="Learning rate for Sender's parameters (default: 1e-1)",
    )
    parser.add_argument(
        "--receiver_lr",
        type=float,
        default=1e-1,
        help="Learning rate for Receiver's parameters (default: 1e-1)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="GS temperature for the sender (default: 1.0)",
    )

    args = core.init(parser, params)

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
    Simpler and more maintainable than manual training loop.
    """
    opts = get_params(params)
    device = torch.device("cuda" if opts.cuda else "cpu")

    # Load data
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
    train_data, validation_data, test_data = data_loader.get_iterators()

    # Print training setup
    print(f"\n{'='*70}")
    print(f"Training 2 agents with role alternation")
    print(f"{'='*70}")
    print(f"| Alternation: Batch-level (A→B, then B→A, alternating)")
    print(f"| Total epochs: {opts.n_epochs}")
    print(f"| Feature encoders: SEPARATE (heterogeneous agents)")
    print(f"{'='*70}\n")

    # Create heterogeneous agents
    print("Creating Agent A...")
    agent_A = FullAgent(
        n_features=data_loader.n_features,
        feature_hidden=opts.feature_hidden,
        sender_hidden=opts.sender_hidden,
        receiver_hidden=opts.receiver_hidden
    )

    print("Creating Agent B...")
    agent_B = FullAgent(
        n_features=data_loader.n_features,
        feature_hidden=opts.feature_hidden,
        sender_hidden=opts.sender_hidden,
        receiver_hidden=opts.receiver_hidden
    )

    # Display parameter counts
    agent_A_params = sum(p.numel() for p in agent_A.parameters())
    agent_B_params = sum(p.numel() for p in agent_B.parameters())
    encoder_A_params = sum(p.numel() for p in agent_A.feature_encoder.parameters())
    encoder_B_params = sum(p.numel() for p in agent_B.feature_encoder.parameters())

    print(f"\nAgent A parameters: {agent_A_params:,}")
    print(f"Agent B parameters: {agent_B_params:,}")
    print(f"Total parameters: {agent_A_params + agent_B_params:,}")
    print(f"  (Separate encoders: A={encoder_A_params:,}, B={encoder_B_params:,})\n")

    # Create alternating game (wraps both agents with communication modules)
    print("Creating alternating game wrapper...")
    game = AlternatingGame(
        agent_A=agent_A,
        agent_B=agent_B,
        loss_fn=loss,
        vocab_size=opts.vocab_size,
        sender_embedding=opts.sender_embedding,
        sender_hidden=opts.sender_hidden,
        receiver_embedding=opts.receiver_embedding,
        receiver_hidden=opts.receiver_hidden,
        sender_cell=opts.sender_cell,
        receiver_cell=opts.receiver_cell,
        max_len=opts.max_len,
        temperature=opts.temperature,
    )

    # Single optimizer for all parameters
    lr = (opts.sender_lr + opts.receiver_lr) / 2
    print(f"Using learning rate: {lr:.4f}")
    optimizer = torch.optim.Adam(game.parameters(), lr=lr)

    # Set up callbacks
    callbacks = [
        ResetAlternationCounter(),  # Reset batch counter at start of each epoch
        DirectionalConsoleLogger(n_epochs=opts.n_epochs, print_train_loss=True),
        core.TemperatureUpdater(agent=game.sender_A, decay=0.9, minimum=0.1),
        core.TemperatureUpdater(agent=game.sender_B, decay=0.9, minimum=0.1),
    ]

    print("Setting up EGG Trainer...")
    print(f"{'='*70}\n")

    # Create and run trainer
    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        train_data=train_data,
        validation_data=validation_data,
        callbacks=callbacks,
    )

    trainer.train(n_epochs=opts.n_epochs)

    print(f"\n{'='*70}")
    print("Training complete!")
    print(f"{'='*70}")
    print(f"Random baseline accuracy: {1 / (opts.n_distractors + 1):.3f}\n")


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
