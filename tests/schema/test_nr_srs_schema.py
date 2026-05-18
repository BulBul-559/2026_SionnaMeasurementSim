"""Schema validation tests for NR SRS-like HDF5 output."""

from pathlib import Path

import h5py
import numpy as np

from sionna_measurement_sim.domain.array import ArraySpectrumConfig
from sionna_measurement_sim.io.schema_validator import validate_hdf5_contract
from sionna_measurement_sim.rt.truth_pipeline import RTTruthRunConfig, run_rt_truth_pipeline


def test_nr_srs_pipeline_writes_srs_fields_and_schema(tmp_path: Path):
    path = run_rt_truth_pipeline(
        RTTruthRunConfig(
            label_file=Path("tests/fixtures/scenes/test/test5.json"),
            scene_file=Path("tests/fixtures/scenes/test/scene.xml"),
            output_dir=tmp_path / "nr_srs_schema",
            num_subcarriers=8,
            seed=7,
            max_bs=1,
            max_ue=1,
            bs_num_rows=2,
            bs_num_cols=2,
            ue_num_rows=1,
            ue_num_cols=2,
            max_depth=1,
            observation_snr_db=80.0,
            phy_standard="nr_srs",
            num_ofdm_symbols=2,
            spectrum_config=ArraySpectrumConfig(
                enabled=True,
                sources=("truth_cfr", "srs_cfr_est"),
                zenith_bins=5,
                azimuth_bins=7,
            ),
        )
    )

    validate_hdf5_contract(path)
    with h5py.File(path, "r") as h5:
        assert h5["waveform/standard"][()].decode("utf-8") == "nr_srs"
        assert h5["waveform/srs_tx_grid"].shape == (1, 1, 1, 2, 2, 8)
        assert h5["waveform/srs_rx_grid"].shape == (1, 1, 1, 4, 2, 8)
        assert h5["waveform/srs_noise_variance"].shape == (1, 1, 1)
        assert h5["observation/srs_cfr_est"].shape == h5["observation/cfr_est"].shape
        assert h5["array/spatial_spectrum_srs"].shape == (1, 1, 1, 5, 7)
        assert "waveform/tx_time" not in h5
        assert "waveform/rx_time" not in h5
        assert int(h5["evaluation/num_blocks"][()]) == 0
        assert np.all(np.isfinite(h5["evaluation/nmse_db"][()]))
