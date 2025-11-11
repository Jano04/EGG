# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple, Callable, Dict, Any

import torch
import torch.nn as nn
import egg.core as core
from egg.core.interaction import Interaction


class FeatureEncoder(nn.Module):
    """
    Encoder for feature vectors. Each agent has its own instance,
    creating heterogeneous agents with different perceptual representations.
    """
    def __init__(self, n_features: int, hidden_size: int) -> None:
        super(FeatureEncoder, self).__init__()
        self.fc = nn.Linear(n_features, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class Sender(nn.Module):
    """
    Sender agent that encodes target features.
    Output will be fed to RnnSenderGS to generate messages.
    """
    def __init__(self, feature_encoder: FeatureEncoder, n_hidden: int) -> None:
        super(Sender, self).__init__()
        self.feature_encoder = feature_encoder
        encoder_output_size: int = feature_encoder.fc.out_features
        self.fc1 = nn.Linear(encoder_output_size, n_hidden)

    def forward(self, x: torch.Tensor, _aux_input: Optional[Dict] = None) -> torch.Tensor:
        embedded = self.feature_encoder(x)
        return self.fc1(embedded).tanh()


class Receiver(nn.Module):
    """
    Receiver agent that encodes candidate features.
    Computes similarity between message embedding and encoded candidates.
    """
    def __init__(self, feature_encoder: FeatureEncoder, linear_units: int) -> None:
        super(Receiver, self).__init__()
        self.feature_encoder = feature_encoder
        encoder_output_size: int = feature_encoder.fc.out_features
        self.fc1 = nn.Linear(encoder_output_size, linear_units)

    def forward(
        self,
        x: torch.Tensor,
        _input: torch.Tensor,
        _aux_input: Optional[Dict] = None
    ) -> torch.Tensor:
        embedded_input = self.feature_encoder(_input)
        embedded_input = self.fc1(embedded_input).tanh()
        energies = torch.matmul(embedded_input, torch.unsqueeze(x, dim=-1))
        return energies.squeeze()


class FullAgent(nn.Module):
    """
    Full-fledged agent that can act as both sender and receiver.
    Enables true role alternation where the same agent can send and receive messages.

    Internal structure:
    - feature_encoder: Encodes input features (separate for each agent)
    - sender_module: Encodes targets for message generation
    - receiver_module: Encodes candidates for discrimination

    Each agent has its own feature encoder, creating heterogeneous agents
    with different perceptual representations. This forces genuine language
    emergence to bridge their different "world views".

    Args:
        n_features: Number of input features
        feature_hidden: Hidden size for feature encoder
        sender_hidden: Hidden size for sender network
        receiver_hidden: Hidden size for receiver network
    """
    def __init__(
        self,
        n_features: int,
        feature_hidden: int,
        sender_hidden: int,
        receiver_hidden: int
    ) -> None:
        super(FullAgent, self).__init__()

        # Each agent creates its own encoder (heterogeneous agents)
        self.feature_encoder = FeatureEncoder(n_features, feature_hidden)
        self.sender_module = Sender(self.feature_encoder, sender_hidden)
        self.receiver_module = Receiver(self.feature_encoder, receiver_hidden)

    def as_sender(self) -> Sender:
        """Returns the sender module for wrapping with RnnSenderGS"""
        return self.sender_module

    def as_receiver(self) -> Receiver:
        """Returns the receiver module for wrapping with RnnReceiverGS"""
        return self.receiver_module

    def send(
        self,
        target_features: torch.Tensor,
        aux_input: Optional[Dict] = None
    ) -> torch.Tensor:
        """Send mode: encode target for message generation"""
        return self.sender_module(target_features, aux_input)

    def receive(
        self,
        message_embedding: torch.Tensor,
        candidate_features: torch.Tensor,
        aux_input: Optional[Dict] = None
    ) -> torch.Tensor:
        """Receive mode: discriminate target from candidates given message"""
        return self.receiver_module(message_embedding, candidate_features, aux_input)


class AlternatingGame(nn.Module):
    """
    Game wrapper that alternates roles between two agents at the batch level.
    Compatible with EGG Trainer for streamlined training.

    This implements the core aging game mechanic: two heterogeneous agents
    that take turns being sender and receiver. The alternation happens at
    the batch level:
    - Batch 0: Agent A sends → Agent B receives
    - Batch 1: Agent B sends → Agent A receives
    - Batch 2: Agent A sends → Agent B receives
    - ...

    The game tracks which direction is active and tags each interaction
    accordingly, enabling separate metric tracking for A→B vs B→A.

    Args:
        agent_A: First FullAgent instance
        agent_B: Second FullAgent instance
        loss_fn: Loss function (same signature as objects_game)
        vocab_size: Vocabulary size for discrete messages
        sender_embedding: Embedding dimension for sender RNN
        sender_hidden: Hidden size for sender RNN
        receiver_embedding: Embedding dimension for receiver RNN
        receiver_hidden: Hidden size for receiver RNN
        sender_cell: RNN cell type ('rnn', 'gru', 'lstm')
        receiver_cell: RNN cell type ('rnn', 'gru', 'lstm')
        max_len: Maximum message length
        temperature: Gumbel-Softmax temperature
    """
    def __init__(
        self,
        agent_A: FullAgent,
        agent_B: FullAgent,
        loss_fn: Callable,
        vocab_size: int,
        sender_embedding: int,
        sender_hidden: int,
        receiver_embedding: int,
        receiver_hidden: int,
        sender_cell: str,
        receiver_cell: str,
        max_len: int,
        temperature: float,
    ) -> None:
        super(AlternatingGame, self).__init__()

        self.agent_A: FullAgent = agent_A
        self.agent_B: FullAgent = agent_B
        self.loss_fn: Callable = loss_fn

        # Wrap agent A's modules with RNN communication wrappers
        self.sender_A = core.RnnSenderGS(
            agent_A.as_sender(),
            vocab_size,
            sender_embedding,
            sender_hidden,
            cell=sender_cell,
            max_len=max_len,
            temperature=temperature,
        )
        self.receiver_A = core.RnnReceiverGS(
            agent_A.as_receiver(),
            vocab_size,
            receiver_embedding,
            receiver_hidden,
            cell=receiver_cell,
        )

        # Wrap agent B's modules with RNN communication wrappers
        self.sender_B = core.RnnSenderGS(
            agent_B.as_sender(),
            vocab_size,
            sender_embedding,
            sender_hidden,
            cell=sender_cell,
            max_len=max_len,
            temperature=temperature,
        )
        self.receiver_B = core.RnnReceiverGS(
            agent_B.as_receiver(),
            vocab_size,
            receiver_embedding,
            receiver_hidden,
            cell=receiver_cell,
        )

        # Create two directional games
        self.game_A_to_B = core.SenderReceiverRnnGS(self.sender_A, self.receiver_B, loss_fn)
        self.game_B_to_A = core.SenderReceiverRnnGS(self.sender_B, self.receiver_A, loss_fn)

        # Track batch number for alternation (using buffer so it's saved/loaded with model)
        self.register_buffer('batch_counter', torch.tensor(0, dtype=torch.long))

    def forward(
        self,
        sender_input: torch.Tensor,
        labels: torch.Tensor,
        receiver_input: Optional[torch.Tensor] = None,
        aux_input: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, Interaction]:
        """
        Forward pass that alternates between A→B and B→A based on batch counter.
        During training, alternates by batch. During validation/test, processes
        both directions to get complete metrics.

        Returns:
            loss: Scalar loss (average of both directions if in eval mode)
            interaction: Interaction object with 'direction' field in aux
                        (0.0 for A→B, 1.0 for B→A, or batch with both during eval)
        """
        # In eval mode (validation/test), process BOTH directions to get complete metrics
        if not self.training:
            return self._forward_both_directions(sender_input, labels, receiver_input, aux_input)

        # In training mode, alternate as usual
        direction = self.batch_counter % 2

        if direction == 0:
            # A sends, B receives
            loss, interaction = self.game_A_to_B(sender_input, labels, receiver_input, aux_input)
        else:
            # B sends, A receives
            loss, interaction = self.game_B_to_A(sender_input, labels, receiver_input, aux_input)

        # Tag the interaction with direction (make it batch-sized for consistency)
        batch_size: int = sender_input.size(0)
        interaction.aux['direction'] = torch.full(
            (batch_size,), direction.item(), dtype=torch.float, device=sender_input.device
        )

        # Increment counter for next batch
        self.batch_counter += 1

        return loss, interaction

    def _forward_both_directions(
        self,
        sender_input: torch.Tensor,
        labels: torch.Tensor,
        receiver_input: Optional[torch.Tensor] = None,
        aux_input: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, Interaction]:
        """
        Process the same batch in both directions (for validation/test).
        Returns combined loss and interaction with both directions tagged.
        """
        batch_size = sender_input.size(0)

        # Process A→B direction
        loss_AB, interaction_AB = self.game_A_to_B(sender_input, labels, receiver_input, aux_input)

        # Process B→A direction
        loss_BA, interaction_BA = self.game_B_to_A(sender_input, labels, receiver_input, aux_input)

        # Average the losses
        combined_loss = (loss_AB.mean() + loss_BA.mean()) / 2

        # Combine interactions by concatenating both directions
        # Start with interaction_AB and modify its fields
        combined_interaction = interaction_AB

        # Create direction tags: first half is A→B (0), second half is B→A (1)
        combined_interaction.aux['direction'] = torch.cat([
            torch.zeros(batch_size, dtype=torch.float, device=sender_input.device),
            torch.ones(batch_size, dtype=torch.float, device=sender_input.device)
        ])

        # Concatenate all other aux fields
        for key in interaction_AB.aux.keys():
            if key != 'direction':
                combined_interaction.aux[key] = torch.cat([
                    interaction_AB.aux[key],
                    interaction_BA.aux[key]
                ])

        # Concatenate main fields
        combined_interaction.sender_input = torch.cat([interaction_AB.sender_input, interaction_BA.sender_input])

        if interaction_AB.receiver_input is not None:
            combined_interaction.receiver_input = torch.cat([interaction_AB.receiver_input, interaction_BA.receiver_input])

        if interaction_AB.labels is not None:
            combined_interaction.labels = torch.cat([interaction_AB.labels, interaction_BA.labels])

        combined_interaction.message = torch.cat([interaction_AB.message, interaction_BA.message])

        if interaction_AB.receiver_output is not None:
            combined_interaction.receiver_output = torch.cat([interaction_AB.receiver_output, interaction_BA.receiver_output])

        if interaction_AB.message_length is not None:
            combined_interaction.message_length = torch.cat([interaction_AB.message_length, interaction_BA.message_length])

        return combined_loss, combined_interaction

    def reset_batch_counter(self) -> None:
        """Reset the batch counter (called at the start of each epoch)."""
        self.batch_counter.zero_()

    # Agent accessor methods for post-training analysis and aging mechanics

    def get_agent_A(self) -> FullAgent:
        """
        Return the unwrapped Agent A.

        Useful for:
        - Post-training analysis and evaluation
        - Extracting agents for individual testing
        - Implementing aging mechanics (agent death/birth/replacement)

        Returns:
            The unwrapped FullAgent A (contains feature_encoder, sender_module, receiver_module)
        """
        return self.agent_A

    def get_agent_B(self) -> FullAgent:
        """
        Return the unwrapped Agent B.

        Useful for:
        - Post-training analysis and evaluation
        - Extracting agents for individual testing
        - Implementing aging mechanics (agent death/birth/replacement)

        Returns:
            The unwrapped FullAgent B (contains feature_encoder, sender_module, receiver_module)
        """
        return self.agent_B

    def get_agents(self) -> Tuple[FullAgent, FullAgent]:
        """
        Return both unwrapped agents as a tuple.

        Useful for:
        - Batch operations on both agents
        - Comparing agent architectures or parameters
        - Implementing population-level operations

        Returns:
            Tuple of (agent_A, agent_B)
        """
        return self.agent_A, self.agent_B

    def get_wrapped_components(self) -> Dict[str, nn.Module]:
        """
        Return all wrapped communication components.

        Returns a dictionary containing all wrapped sender/receiver modules
        for fine-grained access to the RNN communication wrappers.

        Returns:
            Dictionary with keys:
                'sender_A': RnnSenderGS wrapping agent A's sender
                'receiver_A': RnnReceiverGS wrapping agent A's receiver
                'sender_B': RnnSenderGS wrapping agent B's sender
                'receiver_B': RnnReceiverGS wrapping agent B's receiver
        """
        return {
            'sender_A': self.sender_A,
            'receiver_A': self.receiver_A,
            'sender_B': self.sender_B,
            'receiver_B': self.receiver_B,
        }
