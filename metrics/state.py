from typing import Dict, List, Optional, Tuple

import editdistance
import numpy as np
import torch
from egg.core.language_analysis import TopographicSimilarity, Disent

from egg.zoo.aging.metrics.types import LanguageState


def _edit_distance(seq1: np.ndarray, seq2: np.ndarray) -> int:
    seq1 = np.asarray(seq1, dtype=np.int64).flatten()
    seq2 = np.asarray(seq2, dtype=np.int64).flatten()
    return editdistance.eval(seq1.tolist(), seq2.tolist())


def _normalized_edit_distance(seq1: np.ndarray, seq2: np.ndarray) -> float:
    seq1 = np.asarray(seq1, dtype=np.int64).flatten()
    seq2 = np.asarray(seq2, dtype=np.int64).flatten()

    def strip_padding(seq):
        zero_indices = np.where(seq == 0)[0]
        if len(zero_indices) == 0:
            return seq
        first_zero = zero_indices[0]
        if first_zero == 0:
            return seq[:1]
        return seq[:first_zero]

    seq1 = strip_padding(seq1)
    seq2 = strip_padding(seq2)
    distance = _edit_distance(seq1, seq2)
    max_len = max(len(seq1), len(seq2))
    return distance / max_len if max_len > 0 else 0.0


def compute_vocab_usage(
    messages_by_sender: Dict[int, np.ndarray],
    alive_indices: List[int],
    vocab_size: int,
) -> float:
    if not alive_indices or vocab_size <= 1:
        return 0.0
    all_messages = np.concatenate([messages_by_sender[i] for i in alive_indices], axis=0)
    unique = set(int(s) for s in all_messages.flatten() if 0 < s < vocab_size)
    return len(unique) / (vocab_size - 1)


def compute_message_length_stats(
    messages_by_sender: Dict[int, np.ndarray],
    alive_indices: List[int],
) -> Tuple[float, float]:
    if not alive_indices:
        return 0.0, 0.0
    all_messages = np.concatenate([messages_by_sender[i] for i in alive_indices], axis=0)
    lengths = (all_messages != 0).sum(axis=1)
    return float(np.mean(lengths)), float(np.std(lengths))


def compute_topographic_similarity(
    messages_by_sender: Dict[int, np.ndarray],
    inputs: np.ndarray,
    alive_indices: List[int],
    input_distance_fn: str = 'hamming',
    max_samples: int = 200,
) -> Optional[float]:
    if len(inputs) < 2 or not alive_indices:
        return None
    n_inputs = len(inputs)
    if n_inputs <= max_samples:
        sample_indices = np.arange(n_inputs)
    else:
        step = n_inputs / max_samples
        sample_indices = np.array([int(i * step) for i in range(max_samples)])
    sampled_inputs = inputs[sample_indices]
    inputs_tensor = torch.from_numpy(sampled_inputs).float()

    scores = []
    for agent_idx in alive_indices:
        messages_list = [messages_by_sender[agent_idx][i].tolist() for i in sample_indices]
        try:
            result = TopographicSimilarity.compute_topsim(
                meanings=inputs_tensor,
                messages=messages_list,
                meaning_distance_fn=input_distance_fn,
                message_distance_fn="edit",
            )
            if not np.isnan(result):
                scores.append(result)
        except Exception:
            pass
    return float(np.mean(scores)) if scores else None


def compute_language_similarity(
    messages_by_sender: Dict[int, np.ndarray],
    alive_indices: List[int],
    n_samples: int,
) -> float:
    if len(alive_indices) < 2:
        return 0.0
    similarities = []
    for sample_idx in range(n_samples):
        sample_messages = [messages_by_sender[agent_idx][sample_idx] for agent_idx in alive_indices]
        for i in range(len(sample_messages)):
            for j in range(i + 1, len(sample_messages)):
                dist = _normalized_edit_distance(sample_messages[i], sample_messages[j])
                similarities.append(1.0 - dist)
    return float(np.mean(similarities)) if similarities else 0.0


def compute_posdis(
    messages_by_sender: Dict[int, np.ndarray],
    inputs: np.ndarray,
    alive_indices: List[int],
) -> Optional[float]:
    if len(inputs) < 2 or not alive_indices:
        return None
    scores = []
    for agent_idx in alive_indices:
        try:
            result = Disent.posdis(
                torch.from_numpy(inputs).long() - 1,
                torch.from_numpy(messages_by_sender[agent_idx]).long()
            )
            if np.isfinite(result):
                scores.append(result)
        except Exception:
            pass
    return float(np.mean(scores)) if scores else None


def compute_bosdis(
    messages_by_sender: Dict[int, np.ndarray],
    inputs: np.ndarray,
    alive_indices: List[int],
    vocab_size: int,
) -> Optional[float]:
    if len(inputs) < 2 or not alive_indices:
        return None
    scores = []
    for agent_idx in alive_indices:
        try:
            result = Disent.bosdis(
                torch.from_numpy(inputs).long() - 1,
                torch.from_numpy(messages_by_sender[agent_idx]).long(),
                vocab_size
            )
            if np.isfinite(result):
                scores.append(result)
        except Exception:
            pass
    return float(np.mean(scores)) if scores else None


def compute_language_state(
    messages_by_sender: Dict[int, np.ndarray],
    inputs: np.ndarray,
    vocab_size: int,
    alive_indices: List[int],
    max_topo_samples: int = 200,
) -> LanguageState:
    """Compute all language state metrics.

    Always computes all metrics including compositionality (topsim, posdis, bosdis).
    """
    length_mean, length_std = compute_message_length_stats(messages_by_sender, alive_indices)
    return LanguageState(
        vocab_usage=compute_vocab_usage(messages_by_sender, alive_indices, vocab_size),
        message_length_mean=length_mean,
        message_length_std=length_std,
        topographic_similarity=compute_topographic_similarity(
            messages_by_sender, inputs, alive_indices, max_samples=max_topo_samples
        ),
        language_similarity=compute_language_similarity(messages_by_sender, alive_indices, len(inputs)),
        posdis=compute_posdis(messages_by_sender, inputs, alive_indices),
        bosdis=compute_bosdis(messages_by_sender, inputs, alive_indices, vocab_size),
    )
