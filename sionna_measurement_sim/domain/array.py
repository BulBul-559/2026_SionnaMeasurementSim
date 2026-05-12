"""Array observation configuration models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ArraySpectrumConfig:
    """Configuration for angle-grid spatial spectrum generation."""

    enabled: bool = False
    sources: tuple[str, ...] = ("truth_cfr", "cfr_est", "rx_grid")
    method: str = "bartlett"
    zenith_bins: int = 91
    azimuth_bins: int = 181
    zenith_min_rad: float = 0.0
    zenith_max_rad: float = 3.141592653589793
    azimuth_min_rad: float = -3.141592653589793
    azimuth_max_rad: float = 3.141592653589793
    normalize: str = "per_link_max"
    aggregate_subcarriers: str = "mean"
    aggregate_symbols: str = "mean"

    def __post_init__(self) -> None:
        sources = tuple(str(source) for source in self.sources)
        allowed_sources = {"truth_cfr", "cfr_est", "rx_grid"}
        unknown_sources = set(sources) - allowed_sources
        if unknown_sources:
            raise ValueError(f"Unsupported spectrum sources: {sorted(unknown_sources)}")
        if self.method != "bartlett":
            raise ValueError("Only method='bartlett' is supported")
        if self.normalize != "per_link_max":
            raise ValueError("Only normalize='per_link_max' is supported")
        if self.aggregate_subcarriers != "mean":
            raise ValueError("Only aggregate_subcarriers='mean' is supported")
        if self.aggregate_symbols != "mean":
            raise ValueError("Only aggregate_symbols='mean' is supported")
        if self.zenith_bins < 2 or self.azimuth_bins < 2:
            raise ValueError("zenith_bins and azimuth_bins must be >= 2")
        if self.zenith_max_rad <= self.zenith_min_rad:
            raise ValueError("zenith_max_rad must be greater than zenith_min_rad")
        if self.azimuth_max_rad <= self.azimuth_min_rad:
            raise ValueError("azimuth_max_rad must be greater than azimuth_min_rad")
        object.__setattr__(self, "sources", sources)

    @property
    def policy(self) -> str:
        return (
            "method=bartlett;"
            f"sources={','.join(self.sources)};"
            f"grid={self.zenith_bins}x{self.azimuth_bins};"
            f"zenith=[{self.zenith_min_rad},{self.zenith_max_rad}];"
            f"azimuth=[{self.azimuth_min_rad},{self.azimuth_max_rad}];"
            f"normalize={self.normalize};"
            f"aggregate_subcarriers={self.aggregate_subcarriers};"
            f"aggregate_symbols={self.aggregate_symbols}"
        )
