"""Configuration for HDF5 visualization reports."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_VISUALIZATION_PLOTS: tuple[str, ...] = (
    "topology",
    "link_overview",
    "cfr_lines",
    "cfr_heatmap",
    "cfr_error",
    "waveform_grid",
    "aoa",
    "nlos_paths",
    "spatial_spectrum",
    "nmse_snr",
    "path_samples",
)

ALLOWED_VISUALIZATION_PLOTS = DEFAULT_VISUALIZATION_PLOTS + (
    "dataset_preview",
    "full_summary",
    "radio_map",
)


@dataclass(frozen=True)
class VisualizationRunConfig:
    """Runtime options for sampled visualization reports."""

    enabled: bool = False
    output_dir: str = "figures"
    sample_policy: str = "valid_links_first"
    random_seed: int = 42
    max_bs: int = 5
    sample_ue_count: int = 3
    max_ue: int = 5
    dpi: int = 140
    format: str = "png"
    plots: tuple[str, ...] = DEFAULT_VISUALIZATION_PLOTS
    radio_map_mode: str = "interpolated"
    radio_map_grid_resolution_m: float | None = None
    radio_map_show_samples: bool = False

    def __post_init__(self) -> None:
        allowed_policies = ("valid_links_first", "spatially_spread_valid_links", "random", "first")
        if self.sample_policy not in allowed_policies:
            raise ValueError(f"Unsupported visualization sample_policy: {self.sample_policy!r}")
        if self.max_bs < 1 or self.sample_ue_count < 1 or self.max_ue < 1:
            raise ValueError("max_bs, sample_ue_count, and max_ue must be >= 1")
        if self.dpi < 50:
            raise ValueError("visualization dpi must be >= 50")
        if self.format != "png":
            raise ValueError("Only visualization format='png' is supported")
        if self.radio_map_mode not in ("interpolated", "samples", "both"):
            raise ValueError(
                "visualization.radio_map_mode must be interpolated/samples/both"
            )
        if (
            self.radio_map_grid_resolution_m is not None
            and self.radio_map_grid_resolution_m <= 0.0
        ):
            raise ValueError("visualization.radio_map_grid_resolution_m must be positive")
        plots = tuple(str(plot) for plot in self.plots)
        unknown = set(plots) - set(ALLOWED_VISUALIZATION_PLOTS)
        if unknown:
            raise ValueError(f"Unsupported visualization plots: {sorted(unknown)}")
        object.__setattr__(self, "plots", plots)
