# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional
import json
import pickle

import numpy as np
import torch

import egg.core as core
from egg.core.interaction import Interaction

from egg.zoo.aging.metrics import (
    Snapshot,
    compute_language_state,
    compute_cross_gen_accuracy,
    compute_cross_gen_similarity,
)

LanguageAnalysisCallback = 'LanguageAnalysisCallback'


class DistributedSamplerEpochSetter(core.Callback):
    def __init__(self):
        super().__init__()

    def on_epoch_begin(self, epoch):
        if self.trainer.distributed_context.is_distributed:
            self.trainer.train_data.sampler.set_epoch(epoch)


class AgeTrackerCallback(core.Callback):
    def __init__(self, opts, language_callback: Optional['LanguageAnalysisCallback'] = None):
        super().__init__()
        self.opts = opts
        self.language_callback = language_callback
        self.writer = None
        self.plasticity_fn = self._create_plasticity_fn() if getattr(opts, 'enable_plasticity', False) else None
        self.last_replacement_epoch = -1
        self.death_policy = self._create_death_policy()
        self.warmup_threshold = getattr(opts, 'warmup_threshold', 0.0)
        self.warmup_complete = (self.warmup_threshold <= 0)
        self.warmup_epoch = 0 if self.warmup_complete else -1
        self.death_count = 0
        self._needs_gen1_snapshot = False  # For resumed runs with warmup disabled

    def _create_death_policy(self):
        from egg.zoo.aging.death_policies import create_death_policy
        policy_name = getattr(self.opts, 'death_policy', 'oldest')
        kwargs = {}
        if policy_name == 'age_weighted':
            kwargs['exponent'] = getattr(self.opts, 'death_age_exponent', 1.0)
        return create_death_policy(policy_name, **kwargs)

    def _create_plasticity_fn(self):
        from egg.zoo.aging.plasticity import SigmoidPlasticity, LinearPlasticity
        fn_type = getattr(self.opts, 'plasticity_function', 'sigmoid')
        if fn_type == 'sigmoid':
            return SigmoidPlasticity(
                steepness=self.opts.plasticity_steepness,
                critical_point=self.opts.plasticity_critical_point
            )
        return LinearPlasticity()

    def _update_plasticity(self, game, epoch: int) -> None:
        if self.plasticity_fn is None:
            return

        optimizer = game.agent_optimizer
        plasticity_max_age = getattr(self.opts, 'plasticity_max_age', 100)

        for i in game.get_alive_indices():
            plasticity = self.plasticity_fn(game.ages[i], plasticity_max_age)

            temp = self.opts.temp_min + (self.opts.temp_max - self.opts.temp_min) * plasticity
            game.senders[i].temperature = temp

            lr = self.opts.lr_min + (self.opts.lr_max - self.opts.lr_min) * plasticity
            optimizer.set_agent_lr(i, lr)

            if self.writer:
                self.writer.add_scalar(f"plasticity/temp_agent_{i}", temp, epoch)
                self.writer.add_scalar(f"plasticity/lr_agent_{i}", lr, epoch)
                self.writer.add_scalar(f"plasticity/value_agent_{i}", plasticity, epoch)

    def _apply_staggered_ages(self, game) -> None:
        """Set ages based on stagger_ages parameter.

        If stagger_ages is an empty list (--stagger_ages with no args), use auto mode:
            ages = [0, k, 2k, ..., (n-1)*k] where k=kill_epoch
        If stagger_ages is a list of integers, use those specific ages.
        """
        stagger = getattr(self.opts, 'stagger_ages', None)
        if stagger is None:
            return

        alive_indices = game.get_alive_indices()

        if len(stagger) == 0:
            # Auto mode: [0, k, 2k, ...]
            k = getattr(self.opts, 'kill_epoch', 10)
            ages = [idx * k for idx in range(len(alive_indices))]
        else:
            # Custom ages provided
            ages = [int(a) for a in stagger]
            if len(ages) < len(alive_indices):
                # Extend with the last age if not enough provided
                ages.extend([ages[-1]] * (len(alive_indices) - len(ages)))

        for idx, agent_idx in enumerate(alive_indices):
            game.ages[agent_idx] = ages[idx]

        print(f"| Staggered ages applied: {[game.ages[i] for i in alive_indices]}")

    def on_train_begin(self, trainer_instance):
        super().on_train_begin(trainer_instance)
        self.writer = core.util.get_summary_writer()

        from egg.zoo.aging.archs import MultiPairPopulationGame
        game = trainer_instance.game
        if not isinstance(game, MultiPairPopulationGame):
            return

        # Check if we loaded from a checkpoint (not just if start_epoch > 0)
        # This is needed because fresh_start resets start_epoch to 0
        loaded_from_checkpoint = getattr(self.opts, 'load_from_checkpoint', None) is not None
        resumed = trainer_instance.start_epoch > 0

        if resumed:
            print(f"| Resumed from epoch {trainer_instance.start_epoch}")
            print(f"| Ages: {[game.ages[i] for i in game.get_alive_indices()]}")
            print(f"| Alive agents: {game.get_alive_indices()}")

        # Apply staggered ages if:
        # 1. We loaded from a checkpoint (fresh_start scenario), OR
        # 2. We're resuming from a checkpoint
        # AND stagger_ages is set (not None)
        stagger_ages = getattr(self.opts, 'stagger_ages', None)
        if stagger_ages is not None and (loaded_from_checkpoint or resumed):
            if getattr(self.opts, 'fresh_start', False) or resumed:
                self._apply_staggered_ages(game)
                # Update plasticity after staggering since ages changed
                if self.plasticity_fn:
                    self._update_plasticity(game, 0)
        elif resumed and self.plasticity_fn:
            self._update_plasticity(game, trainer_instance.start_epoch)

        # Capture gen1_snapshot when warmup is disabled (regardless of fresh_start)
        # This handles: 1) resumed runs with warmup disabled, 2) fresh_start from checkpoint
        if self.warmup_threshold <= 0 and self.language_callback is not None:
            self._needs_gen1_snapshot = True

        if not resumed:
            kill_epoch = getattr(self.opts, 'kill_epoch', 0)
            if self.warmup_threshold > 0:
                print(f"| Warmup: waiting for {self.warmup_threshold*100:.0f}% accuracy")
            if kill_epoch > 0:
                print(f"| Turnover: {self.death_policy.name} policy, every {kill_epoch} epochs after warmup")
            else:
                print(f"| Turnover: disabled (kill_epoch=0)")

    def on_validation_end(self, loss: float, logs: Interaction, epoch: int):
        if self.warmup_complete:
            return

        if not hasattr(logs, 'aux') or 'acc' not in logs.aux:
            return

        acc = logs.aux['acc'].mean().item()
        if acc >= self.warmup_threshold:
            self.warmup_complete = True
            # Offset warmup_epoch so first death triggers on the next epoch_begin
            # This makes integrated warmup behave like separate warmup (immediate first death)
            kill_epoch = getattr(self.opts, 'kill_epoch', 0)
            if kill_epoch > 0:
                self.warmup_epoch = epoch - kill_epoch + 1
            else:
                self.warmup_epoch = epoch
            self.death_count = 0
            print(f"| Warmup complete: accuracy {acc:.3f} >= {self.warmup_threshold} (epoch {epoch})")

            if getattr(self.opts, 'stagger_ages', None) is not None:
                from egg.zoo.aging.archs import MultiPairPopulationGame
                game = self.trainer.game
                if isinstance(game, MultiPairPopulationGame):
                    self._apply_staggered_ages(game)
                    # Update plasticity after staggering since ages changed
                    if self.plasticity_fn:
                        self._update_plasticity(game, epoch)

            if self.language_callback is not None:
                self.language_callback.on_warmup_complete(epoch)

            # Always save a checkpoint when warmup completes (for warmup experiments)
            # This ensures a checkpoint exists even if checkpoint_freq=0
            self._save_warmup_checkpoint(epoch)

            if getattr(self.opts, 'stop_after_warmup', False):
                print(f"| Stopping training (--stop_after_warmup)")
                self.trainer.should_stop = True

    def _save_warmup_checkpoint(self, epoch: int) -> None:
        """Save a checkpoint when warmup completes.

        This ensures a warmup checkpoint exists for loading in later experiments,
        regardless of checkpoint_freq settings.
        """
        checkpoint_dir = getattr(self.opts, 'checkpoint_dir', None)
        if checkpoint_dir is None:
            return

        from egg.core.callbacks import CheckpointSaver

        # Find existing CheckpointSaver or create a temporary one
        checkpointer = None
        for callback in self.trainer.callbacks:
            if isinstance(callback, CheckpointSaver):
                checkpointer = callback
                break

        if checkpointer is not None:
            # Use existing checkpointer
            checkpointer.save_checkpoint(filename="warmup")
            print(f"| Saved warmup checkpoint to {checkpoint_dir}/warmup.tar")
        else:
            # No checkpointer exists (checkpoint_freq might be 0 with no checkpoint_dir properly set)
            # Create a temporary one to save
            import pathlib
            temp_saver = CheckpointSaver(
                checkpoint_path=pathlib.Path(checkpoint_dir),
                checkpoint_freq=0,
            )
            temp_saver.trainer = self.trainer
            temp_saver.save_checkpoint(filename="warmup")
            print(f"| Saved warmup checkpoint to {checkpoint_dir}/warmup.tar")

    def on_epoch_begin(self, epoch: int):
        from egg.zoo.aging.archs import MultiPairPopulationGame
        game = self.trainer.game
        if not isinstance(game, MultiPairPopulationGame):
            return

        # Capture gen1 snapshot for resumed runs (deferred from on_train_begin)
        if self._needs_gen1_snapshot and self.language_callback is not None:
            print(f"| Capturing Gen 1 snapshot from resumed state (epoch {epoch})")
            self.language_callback.on_warmup_complete(epoch)
            self._needs_gen1_snapshot = False

        if not self.warmup_complete:
            return

        kill_epoch = getattr(self.opts, 'kill_epoch', 0)
        if kill_epoch > 0:
            epochs_since_warmup = epoch - self.warmup_epoch
            if epochs_since_warmup >= 0 and epochs_since_warmup % kill_epoch == 0:
                if epoch != self.last_replacement_epoch:
                    # Call on_pre_death BEFORE the death happens
                    # death_count is 0-indexed, so death_count + 1 is the upcoming death number
                    if self.language_callback is not None:
                        self.language_callback.on_pre_death(epoch, self.death_count + 1)

                    # Now perform the death
                    self._replace_agent(game, epoch)
                    self.last_replacement_epoch = epoch
                    self.death_count += 1

                    # Log generation boundary AFTER the death
                    population_size = self.opts.n_agents
                    if self.death_count > 0 and self.death_count % population_size == 0:
                        generation = self.death_count // population_size + 1
                        print(f"| Generation {generation - 1} complete (deaths: {self.death_count})")

        self._update_plasticity(game, epoch)

    def on_epoch_end(self, loss: float, logs, epoch: int):
        from egg.zoo.aging.archs import MultiPairPopulationGame
        game = self.trainer.game
        if not isinstance(game, MultiPairPopulationGame):
            return

        if self.warmup_complete:
            game.increment_ages()
            self._update_plasticity(game, epoch + 1)

        self._log_ages(game, epoch)

    def _compute_agent_accuracies(self, game) -> Dict[int, float]:
        if self.trainer.validation_data is None:
            return {}

        device = next(game.parameters()).device
        alive = game.get_alive_indices()
        acc_sums = {i: 0.0 for i in alive}
        acc_counts = {i: 0 for i in alive}

        was_training = game.training
        game.eval()
        with torch.no_grad():
            for batch in self.trainer.validation_data:
                sender_input, labels = batch[0].to(device), batch[1].to(device)
                receiver_input = batch[2].to(device) if len(batch) > 2 else None

                for sender_idx in alive:
                    for receiver_idx in alive:
                        game_key = f"game_{sender_idx}_{receiver_idx}"
                        _, interaction = game.games[game_key](
                            sender_input, labels, receiver_input, None
                        )
                        acc = interaction.aux['acc'].mean().item()
                        acc_sums[sender_idx] += acc
                        acc_counts[sender_idx] += 1

        if was_training:
            game.train()

        return {i: acc_sums[i] / acc_counts[i] if acc_counts[i] > 0 else 0.0 for i in alive}

    def _replace_agent(self, game, epoch: int) -> None:
        alive_indices = game.get_alive_indices()
        if len(alive_indices) <= 1:
            return

        accuracies = None
        if self.death_policy.name == 'performance_based':
            accuracies = self._compute_agent_accuracies(game)

        victim_idx = self.death_policy.select_victim(alive_indices, game.ages, accuracies)
        if victim_idx is None:
            return

        # Check for dormant agents BEFORE killing to avoid off-by-one error
        dormant = game.get_dormant_indices()
        if not dormant:
            print(f"| Epoch {epoch}: No dormant agents available - stopping turnover")
            self.trainer.should_stop = True
            return

        # Now safe to kill and replace
        game.kill_agent(victim_idx, epoch)
        print(f"| Epoch {epoch}: Agent {victim_idx} died (age={game.ages[victim_idx]}, policy={self.death_policy.name})")

        new_idx = dormant[0]
        game.birth_agent(new_idx, epoch)

        if self.plasticity_fn:
            game.senders[new_idx].temperature = self.opts.temp_max
            game.agent_optimizer.set_agent_lr(new_idx, self.opts.lr_max)

        print(f"| Epoch {epoch}: Agent {new_idx} born")

        self._log_population_status(game, epoch)

    def _log_ages(self, game, epoch: int) -> None:
        if self.writer is None:
            return

        for i, age in enumerate(game.ages):
            self.writer.add_scalar(f"population/age_agent_{i}", age, epoch)
        for i, alive in enumerate(game.alive_mask):
            self.writer.add_scalar(f"population/alive_agent_{i}", int(alive), epoch)

        alive_indices = game.get_alive_indices()
        if alive_indices:
            avg_age_alive = sum(game.ages[i] for i in alive_indices) / len(alive_indices)
            self.writer.add_scalar("population/avg_age_alive", avg_age_alive, epoch)
        self.writer.add_scalar("population/n_alive", game.n_alive, epoch)

    def _log_population_status(self, game, epoch: int) -> None:
        alive = game.get_alive_indices()
        print(f"| Population: {len(alive)} alive {alive}")
        if self.writer:
            self.writer.add_scalar("population/n_alive", len(alive), epoch)


class OptsSaver(core.Callback):
    def __init__(self):
        super().__init__()
        self.opts_saved = False

    def on_train_begin(self, trainer_instance):
        if not self.opts_saved and hasattr(trainer_instance, 'opts'):
            if trainer_instance.opts.checkpoint_dir is None:
                return

            checkpoint_dir = Path(trainer_instance.opts.checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            opts_file = checkpoint_dir / "opts.json"

            opts_dict = vars(trainer_instance.opts)
            json_safe_opts = {}
            for key, value in opts_dict.items():
                try:
                    json.dumps(value)
                    json_safe_opts[key] = value
                except (TypeError, ValueError):
                    json_safe_opts[key] = str(value)

            with open(opts_file, 'w') as f:
                json.dump(json_safe_opts, f, indent=2)

            self.opts_saved = True
            print(f"| Saved training options to {opts_file}")


class LanguageAnalysisCallback(core.Callback):
    """Callback for capturing language snapshots and computing language metrics.

    Captures snapshots at warmup completion (gen1) and before each agent death.
    Always computes language state metrics (vocab, topsim, posdis, bosdis).
    Optionally computes cross-generation accuracy/similarity during training.
    """

    def __init__(
        self,
        vocab_size: int,
        n_agents: int,
        save_dir: Optional[str] = None,
        compute_cross_gen_online: bool = False,
        max_samples: int = 1000,
        max_topo_samples: int = 200,
        writer=None,
        snapshot_freq: int = 0,
    ):
        """Initialize LanguageAnalysisCallback.

        Args:
            vocab_size: Size of the message vocabulary.
            n_agents: Number of active agents in the population.
            save_dir: Directory to save snapshots. None to skip saving.
            compute_cross_gen_online: If True, compute cross-generation accuracy
                and similarity during training (expensive). Default False.
            max_samples: Maximum samples to use for message collection.
            max_topo_samples: Maximum samples for topographic similarity.
            writer: Tensorboard writer. If None, uses core.util.get_summary_writer().
            snapshot_freq: Take periodic snapshots every N epochs. 0 disables.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.n_agents = n_agents
        self.save_dir = Path(save_dir) if save_dir else None
        self.compute_cross_gen_online = compute_cross_gen_online
        self.max_samples = max_samples
        self.max_topo_samples = max_topo_samples
        self.snapshot_freq = snapshot_freq

        self.writer = writer
        self.snapshots: List[Snapshot] = []
        self.gen1_snapshot: Optional[Snapshot] = None
        self.gen1_epoch: Optional[int] = None  # Epoch when gen1 snapshot was taken

        # CSV metrics file for snapshot metrics
        self.metrics_filepath = Path(save_dir).parent / 'snapshot_metrics.csv' if save_dir else None
        self._metrics_file_initialized = False

        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def on_train_begin(self, trainer_instance):
        super().on_train_begin(trainer_instance)
        if self.writer is None:
            self.writer = core.util.get_summary_writer()

        print(f"| LanguageAnalysisCallback: snapshots per agent death")
        if self.snapshot_freq > 0:
            print(f"|   Periodic snapshots: every {self.snapshot_freq} epochs")
        if self.compute_cross_gen_online:
            print(f"|   Cross-gen metrics: computed online (expensive)")
        else:
            print(f"|   Cross-gen metrics: compute post-hoc from snapshots")
        if self.save_dir:
            print(f"|   Snapshots saved to: {self.save_dir}")

    def on_warmup_complete(self, epoch: int) -> None:
        """Called when warmup phase completes. Takes gen1 snapshot (death_number=0)."""
        self.gen1_epoch = epoch  # Track when gen1 was taken for edge case handling
        snapshot, state = self._take_snapshot(epoch, death_number=0)
        if snapshot is None:
            return

        self.gen1_snapshot = snapshot
        print(f"| Gen 1 snapshot captured (epoch {epoch})")

        crossgen = None
        if self.compute_cross_gen_online:
            device = next(self.trainer.game.parameters()).device
            metrics = compute_cross_gen_accuracy(
                snapshot, snapshot,
                self.trainer.game, self.trainer.validation_data, device
            )

            print(f"| Cross-gen accuracy (Gen 1): {metrics['mean']:.4f}")
            print(f"| Cross-gen forward (Gen 1): {metrics['forward']:.4f}")
            print(f"| Cross-gen backward (Gen 1): {metrics['backward']:.4f}")
            print(f"| Cross-gen acc_youngest (Gen 1): {metrics['youngest']:.4f}")

            if self.writer:
                self.writer.add_scalar("crossgen/accuracy", metrics['mean'], epoch)
                self.writer.add_scalar("crossgen/forward", metrics['forward'], epoch)
                self.writer.add_scalar("crossgen/backward", metrics['backward'], epoch)
                self.writer.add_scalar("crossgen/acc_youngest", metrics['youngest'], epoch)
                self.writer.add_scalar("crossgen/generation", 1, epoch)

            sim_metrics = compute_cross_gen_similarity(snapshot, snapshot)
            print(f"| Cross-gen similarity (Gen 1): {sim_metrics['similarity']:.4f}")
            print(f"| Cross-gen sim_youngest (Gen 1): {sim_metrics['youngest']:.4f}")

            if self.writer:
                self.writer.add_scalar("crossgen/similarity", sim_metrics['similarity'], epoch)
                self.writer.add_scalar("crossgen/sim_youngest", sim_metrics['youngest'], epoch)

            crossgen = {
                'mean': metrics['mean'],
                'forward': metrics['forward'],
                'backward': metrics['backward'],
                'youngest': metrics['youngest'],
                'similarity': sim_metrics['similarity'],
                'sim_youngest': sim_metrics['youngest'],
            }

        self._write_snapshot_metrics(epoch, 0, state, crossgen)

    def on_pre_death(self, epoch: int, death_number: int) -> None:
        """Called before each agent death. Takes snapshot and computes metrics.

        Args:
            epoch: Current training epoch.
            death_number: Which death this is (1-indexed). death_number=1 is the first death.
        """
        # Skip if gen1 snapshot was already taken at this epoch (edge case: warmup_threshold=0, kill_epoch=1)
        if epoch == self.gen1_epoch:
            print(f"| Skipping pre-death snapshot: gen1 already captured at epoch {epoch}")
            return

        snapshot, state = self._take_snapshot(epoch, death_number=death_number)
        if snapshot is None:
            return

        print(f"| Pre-death snapshot #{death_number} captured (epoch {epoch})")

        crossgen = None
        if self.compute_cross_gen_online and self.gen1_snapshot is not None:
            device = next(self.trainer.game.parameters()).device
            metrics = compute_cross_gen_accuracy(
                self.gen1_snapshot, snapshot,
                self.trainer.game, self.trainer.validation_data, device
            )

            print(f"| Cross-gen accuracy (death #{death_number}): {metrics['mean']:.4f}")
            print(f"| Cross-gen forward: {metrics['forward']:.4f}")
            print(f"| Cross-gen backward: {metrics['backward']:.4f}")
            print(f"| Cross-gen acc_youngest: {metrics['youngest']:.4f}")

            if self.writer:
                self.writer.add_scalar("crossgen/accuracy", metrics['mean'], epoch)
                self.writer.add_scalar("crossgen/forward", metrics['forward'], epoch)
                self.writer.add_scalar("crossgen/backward", metrics['backward'], epoch)
                self.writer.add_scalar("crossgen/acc_youngest", metrics['youngest'], epoch)
                self.writer.add_scalar("crossgen/death_number", death_number, epoch)

            sim_metrics = compute_cross_gen_similarity(self.gen1_snapshot, snapshot)
            print(f"| Cross-gen similarity: {sim_metrics['similarity']:.4f}")
            print(f"| Cross-gen sim_youngest: {sim_metrics['youngest']:.4f}")

            if self.writer:
                self.writer.add_scalar("crossgen/similarity", sim_metrics['similarity'], epoch)
                self.writer.add_scalar("crossgen/sim_youngest", sim_metrics['youngest'], epoch)

            crossgen = {
                'mean': metrics['mean'],
                'forward': metrics['forward'],
                'backward': metrics['backward'],
                'youngest': metrics['youngest'],
                'similarity': sim_metrics['similarity'],
                'sim_youngest': sim_metrics['youngest'],
            }

        self._write_snapshot_metrics(epoch, death_number, state, crossgen)

    def _take_snapshot(self, epoch: int, death_number: int):
        """Take a snapshot and compute language state metrics.

        Args:
            epoch: Current training epoch.
            death_number: Which death this snapshot precedes. 0 for gen1/warmup.

        Returns:
            Tuple of (snapshot, state) or (None, None) if taking snapshot failed.
        """
        if self.trainer is None:
            return None, None

        from egg.zoo.aging.archs import MultiPairPopulationGame
        game = self.trainer.game
        if not isinstance(game, MultiPairPopulationGame):
            return None, None

        device = next(game.parameters()).device
        validation_data = self.trainer.validation_data

        was_training = game.training
        inputs, labels, alive_indices, messages_by_sender = self._collect_messages(
            game, validation_data, device
        )

        state = compute_language_state(
            messages_by_sender=messages_by_sender,
            inputs=inputs,
            vocab_size=self.vocab_size,
            alive_indices=alive_indices,
            max_topo_samples=self.max_topo_samples,
        )
        self._log_state(state, epoch, death_number)

        snapshot = self._create_snapshot(epoch, death_number, game, inputs, labels, messages_by_sender)
        self.snapshots.append(snapshot)

        if self.save_dir:
            self._save_snapshot(snapshot, epoch, death_number)

        if was_training:
            game.train()

        return snapshot, state

    def _collect_messages(
        self,
        game,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
    ):
        game.eval()
        alive_indices = game.get_alive_indices()

        all_inputs = []
        all_labels = []
        messages_by_agent = {i: [] for i in alive_indices}
        samples_collected = 0

        with torch.no_grad():
            for batch in dataloader:
                if samples_collected >= self.max_samples:
                    break

                sender_input = batch[0].to(device)
                labels_batch = batch[1]

                all_inputs.append(sender_input.cpu().numpy())
                all_labels.append(labels_batch.numpy())

                for agent_idx in alive_indices:
                    sender = game.senders[agent_idx]
                    output = sender(sender_input)

                    if isinstance(output, tuple):
                        message = output[0]
                    else:
                        message = output

                    if len(message.shape) == 3:
                        message_tokens = message.argmax(dim=-1)
                    else:
                        message_tokens = message

                    messages_by_agent[agent_idx].append(
                        message_tokens.long().cpu().numpy()
                    )

                samples_collected += sender_input.size(0)

        inputs = np.concatenate(all_inputs, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        messages_by_sender = {
            i: np.concatenate(msgs, axis=0)
            for i, msgs in messages_by_agent.items()
        }

        return inputs, labels, alive_indices, messages_by_sender

    def _log_state(self, state, epoch: int, death_number: int):
        """Log language state metrics to console and tensorboard."""
        death_info = f" (death #{death_number})" if death_number > 0 else " (gen1)"
        print(f"\n{'='*60}")
        print(f"LANGUAGE STATE - Epoch {epoch}{death_info}")
        print(f"{'='*60}")
        print(f"Vocab usage: {state.vocab_usage:.4f}")
        print(f"Message length: {state.message_length_mean:.2f} +/- {state.message_length_std:.2f}")
        print(f"Language similarity: {state.language_similarity:.4f}")
        print(f"Topographic similarity: {state.topographic_similarity:.4f}" if state.topographic_similarity else "Topographic similarity: N/A")
        print(f"Posdis: {state.posdis:.4f}" if state.posdis else "Posdis: N/A")
        print(f"Bosdis: {state.bosdis:.4f}" if state.bosdis else "Bosdis: N/A")
        print(f"{'='*60}\n")

        if self.writer:
            self.writer.add_scalar("state/vocab_usage", state.vocab_usage, epoch)
            self.writer.add_scalar("state/message_length_mean", state.message_length_mean, epoch)
            self.writer.add_scalar("state/message_length_std", state.message_length_std, epoch)
            self.writer.add_scalar("state/language_similarity", state.language_similarity, epoch)
            if state.topographic_similarity is not None:
                self.writer.add_scalar("state/topographic_similarity", state.topographic_similarity, epoch)
            if state.posdis is not None:
                self.writer.add_scalar("state/posdis", state.posdis, epoch)
            if state.bosdis is not None:
                self.writer.add_scalar("state/bosdis", state.bosdis, epoch)

    def _write_snapshot_metrics(self, epoch: int, death_number: int, state, crossgen: Optional[dict] = None):
        if self.metrics_filepath is None:
            return

        if not self._metrics_file_initialized:
            with open(self.metrics_filepath, 'w') as f:
                f.write('epoch,death_number,vocab_usage,message_length_mean,message_length_std,'
                        'language_similarity,topographic_similarity,posdis,bosdis,'
                        'crossgen_accuracy,crossgen_forward,crossgen_backward,'
                        'crossgen_similarity,crossgen_acc_youngest,crossgen_sim_youngest\n')
            self._metrics_file_initialized = True

        row = [
            epoch,
            death_number,
            f'{state.vocab_usage:.6f}',
            f'{state.message_length_mean:.4f}',
            f'{state.message_length_std:.4f}',
            f'{state.language_similarity:.6f}',
            f'{state.topographic_similarity:.6f}' if state.topographic_similarity is not None else '',
            f'{state.posdis:.6f}' if state.posdis is not None else '',
            f'{state.bosdis:.6f}' if state.bosdis is not None else '',
        ]

        if crossgen:
            row.extend([
                f"{crossgen.get('mean', ''):.6f}" if crossgen.get('mean') is not None else '',
                f"{crossgen.get('forward', ''):.6f}" if crossgen.get('forward') is not None else '',
                f"{crossgen.get('backward', ''):.6f}" if crossgen.get('backward') is not None else '',
                f"{crossgen.get('similarity', ''):.6f}" if crossgen.get('similarity') is not None else '',
                f"{crossgen.get('youngest', ''):.6f}" if crossgen.get('youngest') is not None else '',
                f"{crossgen.get('sim_youngest', ''):.6f}" if crossgen.get('sim_youngest') is not None else '',
            ])
        else:
            row.extend([''] * 6)

        with open(self.metrics_filepath, 'a') as f:
            f.write(','.join(str(x) for x in row) + '\n')

    def _create_snapshot(
        self,
        epoch: int,
        death_number: int,
        game,
        inputs: np.ndarray,
        labels: np.ndarray,
        messages_by_sender: Dict[int, np.ndarray],
    ) -> Snapshot:
        """Create a Snapshot with current population state.

        Args:
            epoch: Current training epoch.
            death_number: Which death this snapshot precedes. 0 for gen1/warmup.
            game: The population game instance.
            inputs: Validation inputs.
            labels: Validation labels.
            messages_by_sender: Messages collected from each agent.
        """
        alive_mask = list(game.alive_mask)
        alive_indices = game.get_alive_indices()
        agent_ages = {i: game.ages[i] for i in range(game.total_agents)}
        birth_epochs = {i: game.birth_epochs[i] for i in range(game.total_agents)}
        death_epochs = {i: game.death_epochs[i] for i in range(game.total_agents)}

        full_messages_by_sender = {i: np.array([]) for i in range(game.total_agents)}
        for agent_idx, msgs in messages_by_sender.items():
            full_messages_by_sender[agent_idx] = msgs

        sender_weights = {}
        receiver_weights = {}
        for i in alive_indices:
            sender_weights[i] = {
                k: v.cpu().clone() for k, v in game.senders[i].state_dict().items()
            }
            receiver_weights[i] = {
                k: v.cpu().clone() for k, v in game.receivers[i].state_dict().items()
            }

        return Snapshot(
            epoch=epoch,
            death_number=death_number,
            messages_by_sender=full_messages_by_sender,
            inputs=inputs,
            labels=labels,
            alive_mask=alive_mask,
            agent_ages=agent_ages,
            birth_epochs=birth_epochs,
            death_epochs=death_epochs,
            sender_weights=sender_weights,
            receiver_weights=receiver_weights,
            generation=0,
            is_pre_death=False,
        )

    def _save_snapshot(self, snapshot: Snapshot, epoch: int, death_number: int) -> None:
        """Save snapshot to disk.

        Filename format:
        - snapshot_gen1_epoch_E.pkl for gen1 (death_number=0)
        - snapshot_death_N_epoch_E.pkl for pre-death snapshots
        - snapshot_periodic_epoch_E.pkl for periodic (death_number=-1)
        """
        if death_number == 0:
            filepath = self.save_dir / f"snapshot_gen1_epoch_{epoch}.pkl"
        elif death_number == -1:
            filepath = self.save_dir / f"snapshot_periodic_epoch_{epoch}.pkl"
        else:
            filepath = self.save_dir / f"snapshot_death_{death_number}_epoch_{epoch}.pkl"
        with open(filepath, "wb") as f:
            pickle.dump(snapshot, f)

    def on_epoch_end(self, loss: float, logs, epoch: int):
        """Take periodic snapshot if snapshot_freq is enabled."""
        if self.snapshot_freq <= 0:
            return
        if epoch > 0 and epoch % self.snapshot_freq == 0:
            snapshot, state = self._take_snapshot(epoch, death_number=-1)
            if snapshot is None:
                return
            print(f"| Periodic snapshot captured (epoch {epoch})")
            self._write_snapshot_metrics(epoch, -1, state, crossgen=None)

    def on_train_end(self):
        if self.writer:
            self.writer.close()


class TrainingMetricsLogger(core.Callback):
    def __init__(self, filepath: str):
        super().__init__()
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

    def on_train_begin(self, trainer_instance):
        super().on_train_begin(trainer_instance)
        with open(self.filepath, 'w') as f:
            f.write('epoch,mode,loss,acc\n')

    def _write_row(self, epoch: int, mode: str, loss: float, acc: float):
        with open(self.filepath, 'a') as f:
            f.write(f'{epoch},{mode},{loss:.6f},{acc:.6f}\n')

    def on_epoch_end(self, loss: float, logs: Interaction, epoch: int):
        acc = logs.aux['acc'].mean().item() if hasattr(logs, 'aux') and 'acc' in logs.aux else 0.0
        self._write_row(epoch, 'train', loss, acc)

    def on_validation_end(self, loss: float, logs: Interaction, epoch: int):
        acc = logs.aux['acc'].mean().item() if hasattr(logs, 'aux') and 'acc' in logs.aux else 0.0
        self._write_row(epoch, 'test', loss, acc)
