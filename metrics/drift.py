from typing import TYPE_CHECKING, List

import copy

import numpy as np
import torch

if TYPE_CHECKING:
    from egg.zoo.aging.metrics.types import Snapshot


def _compute_accuracy(
    sender_snapshot: 'Snapshot',
    receiver_snapshot: 'Snapshot',
    sender_indices: List[int],
    receiver_indices: List[int],
    game_template: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    from egg.zoo.aging.archs import MultiPairPopulationGame
    if not isinstance(game_template, MultiPairPopulationGame):
        return 0.0

    if not sender_indices or not receiver_indices:
        return 0.0

    original_states = {}
    for i in range(game_template.total_agents):
        original_states[i] = {
            'sender': copy.deepcopy(game_template.senders[i].state_dict()),
            'receiver': copy.deepcopy(game_template.receivers[i].state_dict()),
        }

    try:
        for i in sender_indices:
            if i in sender_snapshot.sender_weights:
                state = {k: v.to(device) for k, v in sender_snapshot.sender_weights[i].items()}
                game_template.senders[i].load_state_dict(state)

        for i in receiver_indices:
            if i in receiver_snapshot.receiver_weights:
                state = {k: v.to(device) for k, v in receiver_snapshot.receiver_weights[i].items()}
                game_template.receivers[i].load_state_dict(state)

        game_template.eval()

        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in dataloader:
                sender_input = batch[0].to(device)
                labels = batch[1].long().to(device)
                receiver_input = batch[2].to(device) if len(batch) > 2 else None

                for s_idx in sender_indices:
                    if s_idx not in sender_snapshot.sender_weights:
                        continue
                    for r_idx in receiver_indices:
                        if r_idx not in receiver_snapshot.receiver_weights:
                            continue
                        if s_idx == r_idx:
                            continue

                        game_key = f"game_{s_idx}_{r_idx}"
                        _, interaction = game_template.games[game_key](
                            sender_input, labels, receiver_input, None
                        )

                        acc = interaction.aux['acc']
                        total_correct += acc.sum().item()
                        total_samples += len(acc)

        return total_correct / total_samples if total_samples > 0 else 0.0

    finally:
        for i in range(game_template.total_agents):
            game_template.senders[i].load_state_dict(original_states[i]['sender'])
            game_template.receivers[i].load_state_dict(original_states[i]['receiver'])


def compute_cross_gen_accuracy(
    snap_old: 'Snapshot',
    snap_new: 'Snapshot',
    game_template: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict:
    """Compute bidirectional accuracy between two snapshots.

    Returns dict with keys: 'mean', 'forward', 'backward', 'youngest'
    """
    old_alive = [i for i, alive in enumerate(snap_old.alive_mask) if alive]
    new_alive = [i for i, alive in enumerate(snap_new.alive_mask) if alive]

    if not old_alive or not new_alive:
        return {'mean': 0.0, 'forward': 0.0, 'backward': 0.0, 'youngest': 0.0}

    forward = _compute_accuracy(
        snap_old, snap_new, old_alive, new_alive,
        game_template, dataloader, device
    )
    backward = _compute_accuracy(
        snap_new, snap_old, new_alive, old_alive,
        game_template, dataloader, device
    )

    # Find youngest agent in new snapshot (highest birth epoch or lowest age)
    youngest_idx = None
    min_age = float('inf')
    for i in new_alive:
        age = snap_new.agent_ages.get(i, 0)
        if age < min_age:
            min_age = age
            youngest_idx = i

    # Compute accuracy for youngest agent specifically
    youngest_acc = 0.0
    if youngest_idx is not None:
        youngest_acc = _compute_accuracy(
            snap_old, snap_new, old_alive, [youngest_idx],
            game_template, dataloader, device
        )

    return {
        'mean': (forward + backward) / 2,
        'forward': forward,
        'backward': backward,
        'youngest': youngest_acc,
    }


def compute_cross_gen_similarity(
    snap_old: 'Snapshot',
    snap_new: 'Snapshot',
) -> dict:
    """Compute message similarity between two snapshots.

    Returns dict with keys: 'similarity', 'youngest'
    """
    from egg.zoo.aging.metrics.state import _normalized_edit_distance

    old_alive = [i for i, alive in enumerate(snap_old.alive_mask) if alive]
    new_alive = [i for i, alive in enumerate(snap_new.alive_mask) if alive]

    if not old_alive or not new_alive:
        return {'similarity': 0.0, 'youngest': 0.0}

    n_samples = min(
        min(len(snap_old.messages_by_sender[i]) for i in old_alive),
        min(len(snap_new.messages_by_sender[i]) for i in new_alive),
    )

    # Find youngest agent in new snapshot
    youngest_idx = None
    min_age = float('inf')
    for i in new_alive:
        age = snap_new.agent_ages.get(i, 0)
        if age < min_age:
            min_age = age
            youngest_idx = i

    all_similarities = []
    youngest_similarities = []

    for sample_idx in range(n_samples):
        for old_idx in old_alive:
            msg_old = snap_old.messages_by_sender[old_idx][sample_idx]
            for new_idx in new_alive:
                msg_new = snap_new.messages_by_sender[new_idx][sample_idx]
                sim = 1.0 - _normalized_edit_distance(msg_old, msg_new)
                all_similarities.append(sim)
                if new_idx == youngest_idx:
                    youngest_similarities.append(sim)

    return {
        'similarity': float(np.mean(all_similarities)) if all_similarities else 0.0,
        'youngest': float(np.mean(youngest_similarities)) if youngest_similarities else 0.0,
    }
