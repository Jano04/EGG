# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Optional, Dict, List

import numpy as np


class DeathPolicy(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def select_victim(
        self,
        alive_indices: List[int],
        ages: List[int],
        accuracies: Optional[Dict[int, float]] = None
    ) -> Optional[int]:
        pass


class OldestFirstPolicy(DeathPolicy):
    @property
    def name(self) -> str:
        return "oldest"

    def select_victim(self, alive_indices, ages, accuracies=None) -> Optional[int]:
        if len(alive_indices) <= 1:
            return None
        max_age = max(ages[i] for i in alive_indices)
        oldest = [i for i in alive_indices if ages[i] == max_age]
        return int(np.random.choice(oldest))


class RandomPolicy(DeathPolicy):
    @property
    def name(self) -> str:
        return "random"

    def select_victim(self, alive_indices, ages, accuracies=None) -> Optional[int]:
        if len(alive_indices) <= 1:
            return None
        return int(np.random.choice(alive_indices))


class PerformanceBasedPolicy(DeathPolicy):
    @property
    def name(self) -> str:
        return "performance_based"

    def select_victim(self, alive_indices, ages, accuracies=None) -> Optional[int]:
        if len(alive_indices) <= 1:
            return None
        if accuracies is None:
            return int(np.random.choice(alive_indices))
        alive_accs = {i: accuracies.get(i, 0.0) for i in alive_indices}
        min_acc = min(alive_accs.values())
        worst = [i for i, acc in alive_accs.items() if acc == min_acc]
        return int(np.random.choice(worst))


class AgeWeightedPolicy(DeathPolicy):
    def __init__(self, exponent: float = 1.0):
        self.exponent = exponent

    @property
    def name(self) -> str:
        return "age_weighted"

    def select_victim(self, alive_indices, ages, accuracies=None) -> Optional[int]:
        if len(alive_indices) <= 1:
            return None
        weights = np.array([(ages[i] + 1) ** self.exponent for i in alive_indices])
        probs = weights / weights.sum()
        return int(np.random.choice(alive_indices, p=probs))


def create_death_policy(name: str, **kwargs) -> DeathPolicy:
    policies = {
        'oldest': OldestFirstPolicy,
        'random': RandomPolicy,
        'performance_based': PerformanceBasedPolicy,
        'age_weighted': AgeWeightedPolicy,
    }
    if name not in policies:
        raise ValueError(f"Unknown policy: {name}. Available: {list(policies.keys())}")
    return policies[name](**kwargs)
