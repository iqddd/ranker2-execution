"""Training-only level-supervision experiment helpers."""

from .targets import LCCTarget, build_lcc_target
from .training import (
    GradientCapTrajectoryResult,
    batch_local_theta_loss,
    combine_capped_gradients,
    gradient_cap_result_from_payload,
    gradient_cap_result_payload,
    run_gradient_cap_trajectory,
)

__all__ = [
    "GradientCapTrajectoryResult",
    "LCCTarget",
    "batch_local_theta_loss",
    "build_lcc_target",
    "combine_capped_gradients",
    "gradient_cap_result_from_payload",
    "gradient_cap_result_payload",
    "run_gradient_cap_trajectory",
]
