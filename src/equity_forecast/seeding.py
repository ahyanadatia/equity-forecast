"""Seeding helpers for reproducible runs.

Reproducibility is a stated requirement (same config + seed -> identical metrics
to 6 dp on CPU), so every source of randomness the package touches is seeded
from one place.
"""

from __future__ import annotations

import os
import random

import numpy as np

__all__ = ["set_global_seeds", "set_tensorflow_determinism"]


def set_global_seeds(seed: int) -> None:
    """Seed Python, NumPy and (if importable) TensorFlow.

    Args:
        seed: The seed to apply everywhere.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    set_tensorflow_determinism(seed)


def set_tensorflow_determinism(seed: int) -> None:
    """Seed TensorFlow and enable deterministic op execution.

    TensorFlow is an optional dependency; this is a no-op when it is not
    installed. On CPU, ``enable_op_determinism`` plus a fixed seed is sufficient
    for bitwise-identical training runs. On GPU some cuDNN kernels remain
    nondeterministic - see the reproducibility note in the README.

    Args:
        seed: The seed to apply to TensorFlow's global RNG.
    """
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    try:
        import tensorflow as tf
    except ImportError:  # pragma: no cover - exercised only without tensorflow
        return
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
