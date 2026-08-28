"""Shared implementation for frozen-prefix encoder-adaptation experiments."""

from ranker.experiments.encoder_adaptation.producer_block import (
    ProducerBlockInputCache,
    ProducerBlockJointReadout,
    materialize_producer_input_cache,
    parameter_rows,
)
from ranker.experiments.encoder_adaptation.trajectory import (
    ProducerTrajectoryResult,
    result_from_payload,
    result_payload,
    run_producer_trajectory,
)

__all__ = [
    "ProducerBlockInputCache",
    "ProducerBlockJointReadout",
    "ProducerTrajectoryResult",
    "materialize_producer_input_cache",
    "parameter_rows",
    "result_from_payload",
    "result_payload",
    "run_producer_trajectory",
]
