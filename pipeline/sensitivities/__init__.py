"""Stage 1 sensitivity estimation: per-window FFNN Jacobians (ffnn) and
distance/diagnostic forms over the resulting S matrix (sensitivity_matrix).
"""

from __future__ import annotations

from pipeline.sensitivities.ffnn import (
    SensitivityWindow,
    best_device,
    fit_sensitivities_window,
    make_lagged_inputs,
)
from pipeline.sensitivities.sensitivity_matrix import (
    correlation_from_S,
    distance_concentration,
    distance_from_S,
    effective_dimensionality,
    lopez_de_prado_distance,
)

__all__ = [
    "SensitivityWindow",
    "best_device",
    "fit_sensitivities_window",
    "make_lagged_inputs",
    "distance_from_S",
    "correlation_from_S",
    "lopez_de_prado_distance",
    "distance_concentration",
    "effective_dimensionality",
]
