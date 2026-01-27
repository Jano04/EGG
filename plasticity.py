# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from abc import ABC, abstractmethod


class PlasticityMapping(ABC):
    """Base class for age-to-plasticity mapping functions."""

    @abstractmethod
    def __call__(self, age: int, max_age: int) -> float:
        """Map age to plasticity value in [0, 1]. High for young, low for old."""
        pass


class SigmoidPlasticity(PlasticityMapping):
    """
    Sigmoid plasticity with critical period modeling.

    plasticity = 1 / (1 + exp(steepness * (normalized_age - critical_point)))

    Age semantics:
    - age = number of training epochs completed (0-based)
    - max_age = total lifespan (agent dies when age >= max_age)
    - Agent trains at ages 0, 1, ..., max_age-1 (exactly max_age epochs)

    Args:
        steepness: Controls sharpness of transition (higher = sharper)
        critical_point: Normalized age (0-1) where transition midpoint occurs
    """

    def __init__(self, steepness: float = 10.0, critical_point: float = 0.5):
        self.steepness = steepness
        self.critical_point = critical_point

    def __call__(self, age: int, max_age: int) -> float:
        if max_age <= 1:
            return 1.0
        # Normalize to 0-1 range:
        # - age=0 → normalized=0 → high plasticity
        # - age=max_age-1 (last training) → normalized=1 → low plasticity
        normalized_age = age / (max_age - 1)
        return 1.0 / (1.0 + math.exp(self.steepness * (normalized_age - self.critical_point)))


class LinearPlasticity(PlasticityMapping):
    """
    Linear plasticity decay from 1.0 (young) to 0.0 (old).

    Age semantics:
    - age = number of training epochs completed (0-based)
    - max_age = total lifespan (agent dies when age >= max_age)
    - age=0 → plasticity=1.0, age=max_age-1 → plasticity=0.0
    """

    def __call__(self, age: int, max_age: int) -> float:
        if max_age <= 1:
            return 1.0
        # age=0 → plasticity=1.0, age=max_age-1 → plasticity=0.0
        return max(0.0, 1.0 - age / (max_age - 1))
