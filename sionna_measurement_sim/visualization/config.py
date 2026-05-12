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

    def __post_init__(self) -> None:
        if self.sample_policy not in ("valid_links_first", "random", "first"):
            raise ValueError(f"Unsupported visualization sample_policy: {self.sample_policy!r}")
        if self.max_bs < 1 or self.sample_ue_count < 1 or self.max_ue < 1:
            raise ValueError("max_bs, sample_ue_count, and max_ue must be >= 1")
        if self.dpi < 50:
            raise ValueError("visualization dpi must be >= 50")
        if self.format != "png":
            raise ValueError("Only visualization format='png' is supported")
        plots = tuple(str(plot) for plot in self.plots)
        unknown = set(plots) - set(ALLOWED_VISUALIZATION_PLOTS)
        if unknown:
            raise ValueError(f"Unsupported visualization plots: {sorted(unknown)}")
        object.__setattr__(self, "plots", plots)
