"""Factory for the source UMNN-M normalizing flow at joint 120D scale."""
from __future__ import annotations

def build_joint_umnn(target_size: int = 120, condition_size: int = 480):
    """Build the wind UMNN_M_1 architecture with joint input dimensions."""
    # Lazy import keeps the rest of stde_cdm usable when the legacy UMNN
    # package at the workspace root is not on sys.path.
    from models import (
        AutoregressiveConditioner,
        MonotonicNormalizer,
        buildFCNormalizingFlow,
    )
    conditioner_args = {
        "in_size": target_size,
        "hidden": [300, 300, 300, 300],
        "out_size": 20,
        "cond_in": condition_size,
    }
    normalizer_args = {
        "integrand_net": [40, 40, 40],
        "cond_size": 20,
        "nb_steps": 50,
        "solver": "CCParallel",
        "hot_encoding": True,
    }
    return buildFCNormalizingFlow(
        nb_steps=1,
        conditioner_type=AutoregressiveConditioner,
        conditioner_args=conditioner_args,
        normalizer_type=MonotonicNormalizer,
        normalizer_args=normalizer_args,
    )
