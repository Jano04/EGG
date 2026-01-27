from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class LanguageState:
    """Computed metrics for language state at a point in time."""
    vocab_usage: float
    message_length_mean: float
    message_length_std: float
    topographic_similarity: Optional[float] = None
    language_similarity: Optional[float] = None
    posdis: Optional[float] = None
    bosdis: Optional[float] = None


@dataclass
class Snapshot:
    """Raw data captured at a specific epoch, typically just before an agent death.

    Attributes:
        epoch: Training epoch when snapshot was taken.
        death_number: Which death this snapshot precedes. 0 = gen1/warmup (founding population),
                      N > 0 = taken just before death #N.
    """
    epoch: int
    death_number: int
    messages_by_sender: Dict[int, np.ndarray]
    inputs: np.ndarray
    labels: np.ndarray
    alive_mask: List[bool]
    agent_ages: Dict[int, int]
    birth_epochs: Dict[int, int] = field(default_factory=dict)
    death_epochs: Dict[int, int] = field(default_factory=dict)
    sender_weights: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    receiver_weights: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    generation: int = 0
    is_pre_death: bool = False
