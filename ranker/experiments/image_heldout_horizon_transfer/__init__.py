"""Pure helpers for matched image-heldout horizon-transfer experiments."""

from .analysis import (
    build_image_heldout_fold,
    classify_image_transfer,
    fold_percentile_maps,
    level_metric_vectors,
    mapping_digest,
    strict_endpoint_vectors,
    strict_heldout_exposures,
)
from .certification import (
    bootstrap_weight_matrix,
    endpoint_coverage,
    equal_identity_mean,
    fit_constrained_reml,
    partition_anchor_exposures,
    per_target_direct,
    uq_from_image_contrasts,
    variance_routing_and_projections,
)

__all__ = [
    "build_image_heldout_fold",
    "classify_image_transfer",
    "fold_percentile_maps",
    "level_metric_vectors",
    "mapping_digest",
    "strict_endpoint_vectors",
    "strict_heldout_exposures",
    "bootstrap_weight_matrix",
    "endpoint_coverage",
    "equal_identity_mean",
    "fit_constrained_reml",
    "partition_anchor_exposures",
    "per_target_direct",
    "uq_from_image_contrasts",
    "variance_routing_and_projections",
]
