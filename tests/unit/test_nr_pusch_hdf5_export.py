from __future__ import annotations

from dataclasses import replace

import h5py
import numpy as np

from sionna_measurement_sim.domain.observation import (
    EvaluationResult,
    ImpairmentSpec,
    ObservationResult,
    ReceiverSpec,
    WaveformSpec,
)
from sionna_measurement_sim.domain.results import create_phase1_minimal_result
from sionna_measurement_sim.io.hdf5_writer import write_measurement_result
from sionna_measurement_sim.phy.nr_pusch_observation import build_array_outputs_from_waveform


def test_nr_pusch_hdf5_writes_waveform_grids_and_array_outputs(tmp_path):
    base = create_phase1_minimal_result()
    link_shape = (1, 1, 1)
    cfr_est = base.truth.cfr[np.newaxis, ...]
    tx_grid = np.ones((1, 1, 1, 1, 14, 8), dtype=np.complex64)
    rx_grid = (2.0 * tx_grid).astype(np.complex64)
    noise_variance = np.full(link_shape, 0.25, dtype=np.float32)

    result = replace(
        base,
        waveform=WaveformSpec(
            standard="nr_pusch",
            sample_rate_hz=240_000.0,
            fft_size=8,
            cp_length=0,
            num_ofdm_symbols=14,
            pilot_indices=np.array([], dtype=np.int32),
            data_subcarrier_indices=np.arange(8, dtype=np.int32),
            pilot_symbols=np.array([], dtype=np.complex64),
            tx_power_dbm=0.0,
        ),
        observation=ObservationResult(
            cfr_est=cfr_est,
            valid_mask=np.ones(link_shape, dtype=np.bool_),
            detection_success=np.ones(link_shape, dtype=np.bool_),
            estimation_success=np.ones(link_shape, dtype=np.bool_),
            snr_db=np.full(link_shape, 30.0, dtype=np.float32),
            rssi_dbm=np.zeros(link_shape, dtype=np.float32),
            noise_power_dbm=np.zeros(link_shape, dtype=np.float32),
            cfo_hz=np.zeros(link_shape, dtype=np.float32),
            sfo_ppm=np.zeros(link_shape, dtype=np.float32),
            timing_offset_samples=np.zeros(link_shape, dtype=np.float32),
            phase_offset_rad=np.zeros(link_shape, dtype=np.float32),
            agc_gain_db=np.zeros((1, 1), dtype=np.float32),
            clipping_flag=np.zeros(link_shape, dtype=np.bool_),
        ),
        receiver=ReceiverSpec(receiver_type="pusch_receiver", mimo_detector="lmmse"),
        evaluation=EvaluationResult(
            nmse_db=np.zeros(link_shape, dtype=np.float32),
            nmse_db_total=np.zeros(link_shape, dtype=np.float32),
            amplitude_error_db=np.zeros(link_shape, dtype=np.float32),
            phase_error_rad=np.zeros(link_shape, dtype=np.float32),
            correlation=np.ones(link_shape, dtype=np.float32),
            detection_rate=1.0,
            estimation_failure_rate=0.0,
            num_blocks=1,
        ),
        impairments=ImpairmentSpec(
            model_version="nr_pusch_mimo_v1",
            random_seed=1,
            awgn_config='{"snr_db": 30.0}',
        ),
        waveform_extras={
            "num_prb": 1,
            "subcarrier_spacing_khz": 30,
            "num_layers": 1,
            "num_antenna_ports": 1,
            "mcs_index": 14,
            "mcs_table": 1,
            "dmrs_config_type": 1,
            "dmrs_length": 1,
            "dmrs_additional_position": 1,
            "num_cdm_groups_without_data": 2,
            "tx_grid": tx_grid,
            "rx_grid": rx_grid,
            "noise_variance": noise_variance,
        },
        array_outputs=build_array_outputs_from_waveform(rx_grid),
    )

    output_path = tmp_path / "nr_pusch_export.h5"
    write_measurement_result(output_path, result)

    with h5py.File(output_path, "r") as h5:
        assert "waveform/tx_time" not in h5
        assert "waveform/rx_time" not in h5
        assert h5["waveform/tx_grid"].shape == tx_grid.shape
        assert h5["waveform/rx_grid"].shape == rx_grid.shape
        assert h5["waveform/noise_variance"].shape == noise_variance.shape
        assert h5["waveform/tx_grid"].attrs["unit"] == "linear_complex"
        assert h5["waveform/rx_grid"].attrs["index_order"] == (
            "snapshot,ul_tx,ul_rx,ul_rx_ant,ofdm_symbol,subcarrier"
        )
        assert h5["array/rx_snapshot_matrix"].shape == (1, 1, 1, 1, 1)
        assert h5["array/aoa_heatmap_label"].shape == (1, 1, 1, 91, 181)
        assert "array/spatial_spectrum_label" not in h5
        assert h5["array/angle_grid_rad"].shape == (91, 181, 2)
        assert h5["array/aoa_label_rad"].attrs["coordinate_frame"] == "scene"
        assert h5["array/aoa_heatmap_label"].attrs["coordinate_frame"] == "scene"
        assert h5["array/angle_grid_rad"].attrs["coordinate_frame"] == "scene"
        assert h5["array/spectrum_policy"][()].decode("utf-8").startswith("method=bartlett")
