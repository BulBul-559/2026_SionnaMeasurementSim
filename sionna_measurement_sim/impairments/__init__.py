"""Re-export impairment models from the PHY layer."""

from sionna_measurement_sim.phy.impairments import (
    ImpairmentConfig,
    ImpairmentSample,
    apply_base_impairments,
)

__all__ = [
    "ImpairmentConfig",
    "ImpairmentSample",
    "apply_base_impairments",
]
