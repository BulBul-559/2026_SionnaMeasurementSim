"""PHY module interfaces and built-in PHY observation adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from sionna_measurement_sim.domain.observation import (
    EvaluationResult,
    ImpairmentSpec,
    ObservationResult,
    ReceiverSpec,
    WaveformSpec,
)
from sionna_measurement_sim.phy.observation_pipeline import (
    AWGNObservationConfig,
    PHYObservationBundle,
    run_awgn_ls_observation,
)


@dataclass(frozen=True)
class PHYContext:
    """Runtime context passed into a PHY module."""

    config: Any
    adapter_result: Any
    tracer: Any | None = None


@dataclass(frozen=True)
class PHYModuleResult:
    """Domain outputs produced by a PHY module."""

    waveform: WaveformSpec | None = None
    observation: ObservationResult | None = None
    impairments: ImpairmentSpec | None = None
    receiver: ReceiverSpec | None = None
    evaluation: EvaluationResult | None = None
    waveform_extras: dict[str, Any] = field(default_factory=dict)
    array_outputs: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_bundle(self) -> PHYObservationBundle | None:
        """Return the legacy bundle object when all core outputs are present."""

        if (
            self.waveform is None
            or self.observation is None
            or self.impairments is None
            or self.receiver is None
            or self.evaluation is None
        ):
            return None
        return PHYObservationBundle(
            waveform=self.waveform,
            observation=self.observation,
            impairments=self.impairments,
            receiver=self.receiver,
            evaluation=self.evaluation,
        )


class PHYModule(Protocol):
    """Interface implemented by pluggable PHY observation modules."""

    standard: str

    def run(self, context: PHYContext) -> PHYModuleResult:
        """Run the PHY module and return domain outputs."""


class CustomOFDMModule:
    """Adapter for the existing custom OFDM AWGN/LS observation path."""

    standard = "custom_ofdm"

    def run(self, context: PHYContext) -> PHYModuleResult:
        config = context.config
        adapter_result = context.adapter_result
        bundle = run_awgn_ls_observation(
            adapter_result.truth.cfr,
            AWGNObservationConfig(
                snr_db=config.observation_snr_db,
                random_seed=config.observation_seed,
                sample_rate_hz=config.bandwidth_hz,
                fft_size=config.num_subcarriers,
                impairment=config.impairment_config,
            ),
            has_signal=adapter_result.truth.has_geometric_signal,
            cfr_snapshots=adapter_result.truth.cfr_snapshots,
        )
        return PHYModuleResult(
            waveform=bundle.waveform,
            observation=bundle.observation,
            impairments=bundle.impairments,
            receiver=bundle.receiver,
            evaluation=bundle.evaluation,
        )


class NRPUSCHModule:
    """Adapter for the existing NR PUSCH observation path."""

    standard = "nr_pusch"

    def run(self, context: PHYContext) -> PHYModuleResult:
        config = context.config
        adapter_result = context.adapter_result
        from sionna_measurement_sim.phy.nr_pusch_observation import (
            run_nr_pusch_observation,
        )

        nr_result = run_nr_pusch_observation(
            cir_coefficients=adapter_result.cir_truth.coefficients,
            cir_delays=adapter_result.cir_truth.delays_s,
            link_config=config.link_config,
            phy_config=config,
            carrier_config=config,
            path_power_db=adapter_result.truth.path_power_db,
        )
        waveform_extras = {
            "num_prb": config.num_prb,
            "subcarrier_spacing_khz": config.subcarrier_spacing_khz,
            "subcarrier_spacing_hz": config.subcarrier_spacing_khz * 1000.0,
            "slot_number": 0,
            "cyclic_prefix": "normal",
            "target_coderate": 0.54,
            "modulation": "16QAM",
            "num_layers": config.num_layers,
            "num_antenna_ports": config.num_antenna_ports,
            "mcs_index": config.mcs_index,
            "mcs_table": config.mcs_table,
            "dmrs_config_type": config.pusch_dmrs_config_type,
            "dmrs_length": config.pusch_dmrs_length,
            "dmrs_additional_position": config.pusch_dmrs_additional_position,
            "num_cdm_groups_without_data": config.pusch_num_cdm_groups_without_data,
            **nr_result["waveform_grids"],
        }
        return PHYModuleResult(
            waveform=nr_result["nr_waveform_spec"],
            observation=nr_result["observation"],
            impairments=nr_result["impairments"],
            receiver=nr_result["receiver_spec"],
            evaluation=nr_result["evaluation"],
            waveform_extras=waveform_extras,
            array_outputs=nr_result["array_outputs"],
            diagnostics={
                "batching_stats": nr_result.get("batching_stats", {}),
            },
            metadata={
                "pusch_config": nr_result["pusch_config"],
            },
        )


class NRSRSModule:
    """Adapter for the NR SRS standards-shaped v2 observation path."""

    standard = "nr_srs"

    def run(self, context: PHYContext) -> PHYModuleResult:
        config = context.config
        adapter_result = context.adapter_result
        from sionna_measurement_sim.phy.nr_srs_observation import (
            run_nr_srs_observation,
        )

        srs_result = run_nr_srs_observation(
            truth_cfr=adapter_result.truth.cfr,
            cfr_snapshots=adapter_result.truth.cfr_snapshots,
            has_signal=adapter_result.truth.has_geometric_signal,
            path_power_db=adapter_result.truth.path_power_db,
            link_config=config.link_config,
            phy_config=config,
            carrier_config=config,
            tracer=context.tracer,
        )
        waveform_extras = {
            "subcarrier_spacing_khz": config.subcarrier_spacing_khz,
            "subcarrier_spacing_hz": config.subcarrier_spacing_khz * 1000.0,
            "slot_number": getattr(getattr(config, "srs_config", None), "slot_number", 0),
            "cyclic_prefix": "normal",
            "modulation": "srs_zc_like_subset",
            **srs_result["waveform_grids"],
        }
        return PHYModuleResult(
            waveform=srs_result["nr_waveform_spec"],
            observation=srs_result["observation"],
            impairments=srs_result["impairments"],
            receiver=srs_result["receiver_spec"],
            evaluation=srs_result["evaluation"],
            waveform_extras=waveform_extras,
            metadata=srs_result.get("metadata", {}),
        )


PHY_REGISTRY: dict[str, PHYModule] = {
    CustomOFDMModule.standard: CustomOFDMModule(),
    NRPUSCHModule.standard: NRPUSCHModule(),
    NRSRSModule.standard: NRSRSModule(),
}


def get_phy_module(standard: str) -> PHYModule:
    """Return the registered PHY module for a config standard."""

    key = str(standard)
    try:
        return PHY_REGISTRY[key]
    except KeyError as exc:
        supported = ", ".join(sorted(PHY_REGISTRY))
        raise ValueError(f"Unsupported PHY standard {key!r}. Supported: {supported}") from exc
