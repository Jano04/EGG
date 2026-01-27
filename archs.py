# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple, Callable, Dict, Any, List

import torch
import torch.nn as nn
import egg.core as core
from egg.core.interaction import Interaction


class Sender(nn.Module):
    """
    Sender agent that encodes target features.
    Output will be fed to RnnSenderGS to generate messages.
    """
    def __init__(self, n_features: int, n_hidden: int) -> None:
        super(Sender, self).__init__()
        self.fc1 = nn.Linear(n_features, n_hidden)

    def forward(self, x: torch.Tensor, _aux_input: Optional[Dict] = None) -> torch.Tensor:
        return self.fc1(x).tanh()


class Receiver(nn.Module):
    """
    Receiver agent that encodes candidate features.
    Computes similarity between message embedding and encoded candidates.
    """
    def __init__(self, n_features: int, linear_units: int) -> None:
        super(Receiver, self).__init__()
        self.fc1 = nn.Linear(n_features, linear_units)

    def forward(
        self,
        x: torch.Tensor,
        _input: torch.Tensor,
        _aux_input: Optional[Dict] = None
    ) -> torch.Tensor:
        embedded_input = self.fc1(_input).tanh()
        energies = torch.matmul(embedded_input, torch.unsqueeze(x, dim=-1))
        return energies.squeeze()


class FullAgent(nn.Module):
    """
    Full-fledged agent that can act as both sender and receiver.
    Enables true role alternation where the same agent can send and receive messages.

    Internal structure:
    - sender_module: Encodes targets for message generation
    - receiver_module: Encodes candidates for discrimination

    All agents work in the same perceptual space, operating directly on raw features.
    They learn different communication strategies but perceive the world identically.

    Args:
        n_features: Number of input features
        sender_hidden: Hidden size for sender network
        receiver_hidden: Hidden size for receiver network
    """
    def __init__(
        self,
        n_features: int,
        sender_hidden: int,
        receiver_hidden: int
    ) -> None:
        super(FullAgent, self).__init__()

        self.sender_module = Sender(n_features, sender_hidden)
        self.receiver_module = Receiver(n_features, receiver_hidden)

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


class MultiPairPopulationGame(nn.Module):
    """
    Population game for N symmetric agents (N>=2).

    Samples exactly 1 pair per batch during training (academic standard from
    Baroni/Rita papers). Role alternation occurs naturally through random
    sampling from alive agents.

    Architecture:
        - N symmetric FullAgents, each with sender and receiver capabilities
        - During training: sample 1 random pair per batch from alive agents
        - During evaluation: systematically iterate through all alive agent pairs
        - Supports self-communication (agent_i -> agent_i)

    Args:
        agents: List of FullAgent instances (length N>=2)
        loss_fn: Loss function signature
        vocab_size: Vocabulary size for discrete messages
        sender_embedding: Embedding dimension for sender RNN
        sender_hidden: Hidden size for sender RNN
        receiver_embedding: Embedding dimension for receiver RNN
        receiver_hidden: Hidden size for receiver RNN
        sender_cell: RNN cell type ('rnn', 'gru', 'lstm')
        receiver_cell: RNN cell type ('rnn', 'gru', 'lstm')
        max_len: Maximum message length
    """

    def __init__(
        self,
        agents: List[FullAgent],
        loss_fn: Callable,
        vocab_size: int,
        sender_embedding: int,
        sender_hidden: int,
        receiver_embedding: int,
        receiver_hidden: int,
        sender_cell: str,
        receiver_cell: str,
        max_len: int,
        total_agents: int = None,
        all_agents: List[FullAgent] = None,
    ) -> None:
        super(MultiPairPopulationGame, self).__init__()

        self.n_agents = len(agents)
        self.loss_fn = loss_fn

        self.total_agents = total_agents if total_agents else len(agents)
        self.all_agents = nn.ModuleList(all_agents if all_agents else agents)

        # Agent lifecycle states:
        #   - Alive: alive_mask[i] == True (currently active in training)
        #   - Dead: alive_mask[i] == False AND birth_epochs[i] != -1 (was alive, now dead)
        #   - Dormant: alive_mask[i] == False AND birth_epochs[i] == -1 (never born)
        self.alive_mask = [True] * len(agents) + [False] * (self.total_agents - len(agents))

        # Lifecycle tracking: when each agent was born/died
        # birth_epochs: 0 for original agents, -1 for never-born, >0 for later births
        self.birth_epochs = [0] * len(agents) + [-1] * (self.total_agents - len(agents))
        self.death_epochs = [-1] * self.total_agents

        # Ages for all slots (dormant agents stay at 0)
        self.ages = [0] * self.total_agents

        self.senders = nn.ModuleList([
            core.RnnSenderGS(
                self.all_agents[i].as_sender(),
                vocab_size,
                sender_embedding,
                sender_hidden,
                cell=sender_cell,
                max_len=max_len,
                temperature=1.0,
            )
            for i in range(self.total_agents)
        ])

        self.receivers = nn.ModuleList([
            core.RnnReceiverGS(
                self.all_agents[i].as_receiver(),
                vocab_size,
                receiver_embedding,
                receiver_hidden,
                cell=receiver_cell,
            )
            for i in range(self.total_agents)
        ])

        self.games = nn.ModuleDict()
        for i in range(self.total_agents):
            for j in range(self.total_agents):
                key = f"game_{i}_{j}"
                self.games[key] = core.SenderReceiverRnnGS(
                    self.senders[i],
                    self.receivers[j],
                    loss_fn,
                )

        # Track sampled pairs for optimizer (lists for K pairs per batch)
        self.last_sender_indices = []
        self.last_receiver_indices = []

    def forward(
        self,
        sender_input: torch.Tensor,
        labels: torch.Tensor,
        receiver_input: Optional[torch.Tensor] = None,
        aux_input: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, Interaction]:
        if self.training:
            return self._forward_training(sender_input, labels, receiver_input, aux_input)
        else:
            return self._forward_evaluation(sender_input, labels, receiver_input, aux_input)

    def _forward_training(
        self,
        sender_input: torch.Tensor,
        labels: torch.Tensor,
        receiver_input: Optional[torch.Tensor],
        aux_input: Optional[Dict]
    ) -> Tuple[torch.Tensor, Interaction]:
        alive_indices = self.get_alive_indices()
        n_alive = len(alive_indices)

        if n_alive < 1:
            raise RuntimeError(f"Need at least 1 alive agent, got {n_alive}")

        K = n_alive

        pairs = []
        for _ in range(K):
            s_idx = alive_indices[torch.randint(n_alive, (1,)).item()]
            r_idx = alive_indices[torch.randint(n_alive, (1,)).item()]
            pairs.append((s_idx, r_idx))

        self.last_sender_indices = [p[0] for p in pairs]
        self.last_receiver_indices = [p[1] for p in pairs]

        total_loss = 0
        all_accs = []
        sender_lengths = {}

        for s_idx, r_idx in pairs:
            game_key = f"game_{s_idx}_{r_idx}"
            loss, interaction = self.games[game_key](
                sender_input, labels, receiver_input, aux_input
            )
            total_loss = total_loss + loss
            all_accs.append(interaction.aux['acc'])
            sender_lengths.setdefault(s_idx, []).append(interaction.aux['length'])

        avg_loss = total_loss / K

        merged = interaction
        merged.aux['acc'] = torch.stack(all_accs).mean(dim=0)
        per_sender_means = [torch.stack(lengths).mean(dim=0) for lengths in sender_lengths.values()]
        merged.aux['length'] = torch.stack(per_sender_means).mean(dim=0)

        return avg_loss, merged

    def _forward_evaluation(
        self,
        sender_input: torch.Tensor,
        labels: torch.Tensor,
        receiver_input: Optional[torch.Tensor],
        aux_input: Optional[Dict]
    ) -> Tuple[torch.Tensor, Interaction]:
        alive_indices = self.get_alive_indices()
        all_losses = []
        all_accs = []
        sender_lengths = {}

        for s_idx in alive_indices:
            for r_idx in alive_indices:
                game_key = f"game_{s_idx}_{r_idx}"
                loss, interaction = self.games[game_key](
                    sender_input, labels, receiver_input, aux_input
                )
                all_losses.append(loss)
                all_accs.append(interaction.aux['acc'])
                if s_idx not in sender_lengths:
                    sender_lengths[s_idx] = interaction.aux['length']

        avg_loss = torch.stack(all_losses).mean()
        merged = interaction
        merged.aux['acc'] = torch.stack(all_accs).mean(dim=0)
        merged.aux['length'] = torch.stack(list(sender_lengths.values())).mean(dim=0)
        return avg_loss, merged

    @property
    def temperature(self) -> float:
        """Get current temperature (from first sender)."""
        return self.senders[0].temperature

    @temperature.setter
    def temperature(self, value: float) -> None:
        """Set temperature for all senders in the population."""
        for sender in self.senders:
            sender.temperature = value

    def increment_ages(self) -> None:
        """Increment age of all alive agents.

        All alive agents age by 1 each epoch.
        Dead agents freeze at their death age.
        Dormant agents (never born) stay at 0.
        """
        for i in range(len(self.ages)):
            if self.alive_mask[i]:
                self.ages[i] += 1

    def get_alive_indices(self) -> List[int]:
        """Return indices of currently alive agents."""
        return [i for i, alive in enumerate(self.alive_mask) if alive]

    def get_dead_indices(self) -> List[int]:
        """Return indices of dead agents (were alive, have since died)."""
        return [i for i in range(self.total_agents)
                if not self.alive_mask[i] and self.birth_epochs[i] != -1]

    def get_inactive_indices(self) -> List[int]:
        """Return indices of all inactive agents (both dead and dormant)."""
        return [i for i, alive in enumerate(self.alive_mask) if not alive]

    def get_dormant_indices(self) -> List[int]:
        """Return indices of agents never activated (birth_epoch == -1)."""
        return [i for i in range(self.total_agents)
                if not self.alive_mask[i] and self.birth_epochs[i] == -1]


    @property
    def n_alive(self) -> int:
        """Number of currently alive agents."""
        return sum(self.alive_mask)

    def reinitialize_agent(self, agent_idx: int) -> None:
        """Reset agent weights to fresh random values."""
        for component in [self.all_agents[agent_idx], self.senders[agent_idx], self.receivers[agent_idx]]:
            for module in component.modules():
                if hasattr(module, 'reset_parameters'):
                    module.reset_parameters()

    def kill_agent(self, agent_idx: int, epoch: int) -> None:
        """Mark agent as dead. Dead agents are excluded from pair sampling."""
        self.alive_mask[agent_idx] = False
        self.death_epochs[agent_idx] = epoch

    def birth_agent(self, agent_idx: int, epoch: int, reinitialize: bool = True) -> None:
        """Activate a dormant agent, optionally resetting its weights."""
        self.alive_mask[agent_idx] = True
        self.birth_epochs[agent_idx] = epoch
        self.ages[agent_idx] = 0

        if reinitialize:
            self.reinitialize_agent(agent_idx)

        if hasattr(self, 'agent_optimizer') and self.agent_optimizer is not None:
            self.agent_optimizer.reset_agent_optimizer(agent_idx)

    def state_dict(self, *args, **kwargs) -> Dict[str, Any]:
        state = super().state_dict(*args, **kwargs)
        # Exclude games - they're wrappers around senders/receivers with no unique parameters
        # This reduces state dict from ~230k keys to ~2.5k keys
        state = {k: v for k, v in state.items() if not k.startswith('games.')}
        state['_population.total_agents'] = self.total_agents
        state['_population.n_agents'] = self.n_agents
        state['_population.alive_mask'] = self.alive_mask.copy()
        state['_population.ages'] = self.ages.copy()
        state['_population.birth_epochs'] = self.birth_epochs.copy()
        state['_population.death_epochs'] = self.death_epochs.copy()
        state['_population.temperatures'] = [self.senders[i].temperature for i in range(self.total_agents)]
        state['_population.last_sender_indices'] = self.last_sender_indices.copy()
        state['_population.last_receiver_indices'] = self.last_receiver_indices.copy()
        if hasattr(self, 'agent_optimizer') and self.agent_optimizer is not None:
            state['_population.learning_rates'] = [
                self.agent_optimizer.optimizers[i].param_groups[0]['lr']
                for i in range(self.total_agents)
            ]
        return state

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True) -> None:
        has_population_state = any(k.startswith('_population.') for k in state_dict)

        if has_population_state:
            saved_max = state_dict.get('_population.total_agents')
            if saved_max != self.total_agents:
                raise ValueError(f"Checkpoint total_agents ({saved_max}) != current ({self.total_agents})")

            saved_n = state_dict.get('_population.n_agents')
            if saved_n != self.n_agents:
                print(f"| Note: Checkpoint n_agents ({saved_n}) differs from current ({self.n_agents})")

            self.alive_mask = state_dict['_population.alive_mask'].copy()
            self.ages = state_dict['_population.ages'].copy()
            self.birth_epochs = state_dict['_population.birth_epochs'].copy()
            self.death_epochs = state_dict['_population.death_epochs'].copy()

            if '_population.temperatures' in state_dict:
                for i, temp in enumerate(state_dict['_population.temperatures']):
                    self.senders[i].temperature = temp

            if '_population.last_sender_indices' in state_dict:
                self.last_sender_indices = state_dict['_population.last_sender_indices'].copy()
                self.last_receiver_indices = state_dict['_population.last_receiver_indices'].copy()

            if '_population.learning_rates' in state_dict:
                saved_lrs = state_dict['_population.learning_rates']
                print(f"| Checkpoint LRs: {saved_lrs}")

            nn_state = {k: v for k, v in state_dict.items() if not k.startswith('_population.')}
        else:
            nn_state = state_dict

        # Filter out games keys (they're wrappers with no unique parameters)
        # and use strict=False since games won't be in the state dict
        nn_state = {k: v for k, v in nn_state.items() if not k.startswith('games.')}
        super().load_state_dict(nn_state, strict=False)


class PerAgentOptimizer:
    """
    Separate optimizer per agent for population-based training.

    Wraps N individual Adam optimizers, one per agent. Only agents that
    participated in the current batch (tracked via game.last_sender_indices
    and game.last_receiver_indices) have their optimizers stepped.

    The `state` property returns references to underlying optimizer states,
    allowing EGG Trainer's move_to() to modify them in-place for device movement.
    """

    def __init__(self, game: MultiPairPopulationGame, lr: float, weight_decay: float = 1e-5):
        self.game = game
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizers = {i: self._create_optimizer(i) for i in range(game.total_agents)}
        print(f"| Created {game.total_agents} per-agent optimizers (lr={lr}, weight_decay={weight_decay})")

    def _create_optimizer(self, agent_idx: int) -> torch.optim.Adam:
        params = list(self.game.senders[agent_idx].parameters()) + \
                 list(self.game.receivers[agent_idx].parameters())
        return torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)

    def zero_grad(self) -> None:
        if not self.game.last_sender_indices:
            for i in self.game.get_alive_indices():
                self.optimizers[i].zero_grad()
            return

        participating = set(self.game.last_sender_indices + self.game.last_receiver_indices)
        for idx in participating:
            self.optimizers[idx].zero_grad()

    def step(self) -> None:
        if not self.game.last_sender_indices:
            return

        participating = set(self.game.last_sender_indices + self.game.last_receiver_indices)
        for idx in participating:
            self.optimizers[idx].step()

    def state_dict(self) -> dict:
        return {str(i): opt.state_dict() for i, opt in self.optimizers.items()}

    def load_state_dict(self, state_dict: dict) -> None:
        for i, state in state_dict.items():
            self.optimizers[int(i)].load_state_dict(state)

    @property
    def state(self):
        """
        Combined state from all underlying optimizers.

        Returns a dict mapping parameters to their optimizer state dicts.
        Values are references to the actual state dicts, so in-place
        modifications (e.g., by move_to) propagate automatically.
        """
        combined = {}
        for opt in self.optimizers.values():
            for param, state in opt.state.items():
                combined[param] = state
        return combined

    @state.setter
    def state(self, new_state):
        """Setter for EGG Trainer compatibility. No-op since move_to modifies in-place."""
        pass

    @property
    def param_groups(self):
        """Combined param_groups from all underlying optimizers."""
        groups = []
        for opt in self.optimizers.values():
            groups.extend(opt.param_groups)
        return groups

    def reset_agent_optimizer(self, agent_idx: int) -> None:
        self.optimizers[agent_idx] = self._create_optimizer(agent_idx)
        print(f"| Reset optimizer for agent {agent_idx}")

    def set_agent_lr(self, agent_idx: int, lr: float) -> None:
        """Update learning rate for a specific agent's optimizer."""
        for param_group in self.optimizers[agent_idx].param_groups:
            param_group['lr'] = lr


