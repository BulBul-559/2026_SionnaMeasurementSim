"""Schema validation tests for NR SRS standards-shaped v2 HDF5 output."""

from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from sionna_measurement_sim.domain.array import ArraySpectrumConfig
from sionna_measurement_sim.io.schema_validator import validate_hdf5_contract
from sionna_measurement_sim.ranging.config import RangingConfig
from sionna_measurement_sim.rt.truth_pipeline import RTTruthRunConfig, run_rt_truth_pipeline


def test_nr_srs_pipeline_writes_common_waveform_fields_and_schema(tmp_path: Path):
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
            num_ofdm_symbols=14,
            srs_config=SimpleNamespace(
                slot_length_symbols=14,
                start_symbol=12,
                num_srs_symbols=2,
                comb_size=2,
                comb_offset=0,
                bwp_start_prb=0,
                bwp_num_prb=None,
                trigger_mode="aperiodic",
                periodicity_slots=1,
                slot_offset=0,
                slot_number=0,
                sequence_type="zc_like",
                sequence_id=0,
                group_hopping="disabled",
                sequence_hopping="disabled",
                cyclic_shift_multiplexing="cyclic_shift",
                cyclic_shift_indices=None,
            ),
            spectrum_config=ArraySpectrumConfig(
                enabled=True,
                sources=("truth_cfr", "cfr_est"),
                zenith_bins=5,
                azimuth_bins=7,
            ),
            ranging_config=RangingConfig(enabled=True),
        )
    )

    validate_hdf5_contract(path)
    with h5py.File(path, "r") as h5:
        assert h5["waveform/standard"][()].decode("utf-8") == "nr_srs"
        assert h5["waveform/tx_grid"].shape == (1, 1, 1, 2, 14, 8)
        assert h5["waveform/rx_grid"].shape == (1, 1, 1, 4, 14, 8)
        assert h5["waveform/noise_variance"].shape == (1, 1, 1)
        assert h5["waveform/tx_power_dbm_per_port"].shape == (1, 1, 2)
        assert h5["waveform/tx_power_scale_linear"].shape == (1, 1, 2)
        assert h5["waveform/serving_rx_index"].shape == (1, 1)
        assert h5["waveform/path_loss_db"].shape == (1, 1)
        assert h5["waveform/power_clipped_flag"].shape == (1, 1, 2)
        assert h5["waveform/srs_resource_mask"].shape == (14, 8)
        assert h5["waveform/srs_pilot_symbols"].shape == (2, 14, 8)
        assert h5["waveform/srs_re_symbol_indices"].shape == h5[
            "waveform/srs_re_subcarrier_indices"
        ].shape
        assert h5["waveform/srs_port_tx_ant_map"].shape == (2, 2)
        assert h5["waveform/srs_prb_start_per_symbol"].shape == (2,)
        assert h5["waveform/srs_prb_count_per_symbol"].shape == (2,)
        assert h5["waveform/srs_tx_power_dbm"].shape == (1, 1, 2)
        assert h5["waveform/srs_power_scale_linear"].shape == (1, 1, 2)
        np.testing.assert_allclose(
            h5["waveform/srs_power_scale_linear"][()],
            h5["waveform/tx_power_scale_linear"][()],
        )
        assert h5["waveform/srs_re_subcarrier_indices"].ndim == 1
        assert h5["observation/cfr_est_resource"].shape[-1] == h5[
            "waveform/srs_re_subcarrier_indices"
        ].shape[0]
        assert h5["observation/cfr_est_resource"].shape[-2] == 2
        assert "waveform/pilot_code" not in h5
        assert "observation/srs_cfr_est" not in h5
        assert "waveform/srs_tx_grid" not in h5
        assert h5["array/spatial_spectrum_cfr_est"].shape == (1, 1, 1, 5, 7)
        assert "array/spatial_spectrum_srs" not in h5
        assert "array/spatial_spectrum_label" not in h5
        assert "waveform/tx_time" not in h5
        assert "waveform/rx_time" not in h5
        assert h5["derived/first_path_propagation_range_m"].shape == (1, 1)
        assert "derived/rtt_like_m" not in h5
        assert h5["ranging/default_estimator"][()].decode("utf-8") == "pdp_peak"
        assert h5["ranging/pdp_peak/toa_est_s"].shape == (1, 1, 1)
        assert h5["ranging/pdp_peak/one_way_range_est_m"].attrs["unit"] == "m"
        assert h5["ranging/phase_slope/range_error_m"].shape == (1, 1, 1)
        assert int(h5["evaluation/num_blocks"][()]) == 0
        assert np.all(np.isfinite(h5["evaluation/nmse_db"][()]))


def test_nr_srs_schema_rejects_legacy_pilot_code(tmp_path: Path):
    path = run_rt_truth_pipeline(
        RTTruthRunConfig(
            label_file=Path("tests/fixtures/scenes/test/test5.json"),
            scene_file=Path("tests/fixtures/scenes/test/scene.xml"),
            output_dir=tmp_path / "nr_srs_schema_legacy",
            num_subcarriers=8,
            seed=8,
            max_bs=1,
            max_ue=1,
            bs_num_rows=1,
            bs_num_cols=1,
            ue_num_rows=1,
            ue_num_cols=1,
            max_depth=1,
            observation_snr_db=80.0,
            phy_standard="nr_srs",
            srs_config=SimpleNamespace(
                slot_length_symbols=14,
                start_symbol=12,
                num_srs_symbols=1,
                comb_size=1,
                comb_offset=0,
                bwp_start_prb=0,
                bwp_num_prb=None,
                trigger_mode="aperiodic",
                periodicity_slots=1,
                slot_offset=0,
                slot_number=0,
                sequence_type="zc_like",
                sequence_id=0,
                group_hopping="disabled",
                sequence_hopping="disabled",
                cyclic_shift_multiplexing="cyclic_shift",
                cyclic_shift_indices=None,
            ),
        )
    )
    with h5py.File(path, "a") as h5:
        h5["waveform"].create_dataset("pilot_code", data=np.ones((1, 1), dtype=np.complex64))

    with np.testing.assert_raises_regex(Exception, "Forbidden dataset"):
        validate_hdf5_contract(path)


def test_nr_srs_pipeline_writes_multiuser_shared_frame(tmp_path: Path):
    path = run_rt_truth_pipeline(
        RTTruthRunConfig(
            label_file=Path("tests/fixtures/scenes/test/test5.json"),
            scene_file=Path("tests/fixtures/scenes/test/scene.xml"),
            output_dir=tmp_path / "nr_srs_multiuser_schema",
            num_subcarriers=24,
            seed=17,
            max_bs=1,
            max_ue=3,
            bs_num_rows=1,
            bs_num_cols=2,
            ue_num_rows=1,
            ue_num_cols=1,
            max_depth=1,
            observation_snr_db=120.0,
            phy_standard="nr_srs",
            num_ofdm_symbols=14,
            srs_config=SimpleNamespace(
                slot_length_symbols=14,
                start_symbol=12,
                num_srs_symbols=1,
                comb_size=4,
                comb_offset=0,
                bwp_start_prb=0,
                bwp_num_prb=None,
                trigger_mode="aperiodic",
                periodicity_slots=1,
                slot_offset=0,
                slot_number=0,
                sequence_type="zc_like",
                sequence_id=0,
                group_hopping="disabled",
                sequence_hopping="disabled",
                cyclic_shift_multiplexing="cyclic_shift",
                cyclic_shift_indices=None,
                multiuser=SimpleNamespace(
                    enabled=True,
                    active_ue_count=3,
                    resource_strategy="comb_offset",
                    frame_policy="sequential",
                ),
            ),
            spectrum_config=ArraySpectrumConfig(
                enabled=False,
                zenith_bins=5,
                azimuth_bins=7,
            ),
        )
    )

    validate_hdf5_contract(path)
    with h5py.File(path, "r") as h5:
        assert h5["meta/schema_version"][()].decode("utf-8") == "2.0.0"
        assert h5["multiuser/standard"][()].decode("utf-8") == "nr_srs"
        assert h5["multiuser/resource_strategy"][()].decode("utf-8") == "comb_offset"
        assert h5["multiuser/rx_grid_shared"].shape == (1, 1, 1, 2, 14, 24)
        assert h5["multiuser/active_tx_indices"][()].tolist() == [[0, 1, 2]]
        assert h5["multiuser/active_tx_global_indices"][()].tolist() == [[0, 1, 2]]
        assert h5["multiuser/comb_offset"][()].tolist() == [[0, 1, 2]]
        assert not np.any(h5["multiuser/resource_collision_mask"][()])
        assert h5["multiuser/cfr_est_resource"].shape[:5] == (1, 1, 3, 1, 2)
        assert h5["multiuser/cfr_est_allocated"].shape[:6] == (1, 1, 3, 1, 2, 1)
        assert h5["observation/cfr_est"].shape[:3] == (1, 3, 1)


def test_nr_srs_pipeline_writes_iq_link_and_noncooperative_time_iq(tmp_path: Path):
    path = run_rt_truth_pipeline(
        RTTruthRunConfig(
            label_file=Path("tests/fixtures/scenes/test/test5.json"),
            scene_file=Path("tests/fixtures/scenes/test/scene.xml"),
            output_dir=tmp_path / "nr_srs_iq_schema",
            num_subcarriers=16,
            seed=19,
            max_bs=1,
            max_ue=2,
            bs_num_rows=1,
            bs_num_cols=2,
            ue_num_rows=1,
            ue_num_cols=1,
            max_depth=1,
            observation_snr_db=90.0,
            phy_standard="nr_srs",
            num_prb=1,
            num_ofdm_symbols=14,
            srs_config=SimpleNamespace(
                slot_length_symbols=14,
                start_symbol=12,
                num_srs_symbols=1,
                comb_size=2,
                comb_offset=0,
                bwp_start_prb=0,
                bwp_num_prb=1,
                trigger_mode="aperiodic",
                periodicity_slots=1,
                slot_offset=0,
                slot_number=0,
                sequence_type="zc_like",
                sequence_id=0,
                group_hopping="disabled",
                sequence_hopping="disabled",
                cyclic_shift_multiplexing="cyclic_shift",
                cyclic_shift_indices=None,
                multiuser=SimpleNamespace(
                    enabled=False,
                    active_ue_count=2,
                    resource_strategy="comb_offset",
                    frame_policy="sequential",
                ),
            ),
            iq_config=SimpleNamespace(
                enabled=True,
                save_frequency_clean=True,
                save_frequency_observed=True,
                save_time_clean=True,
                save_time_observed=True,
                cp_length=2,
            ),
            noncooperative_config=SimpleNamespace(
                enabled=True,
                signal_standard="nr_srs",
                active_tx_count=2,
                frame_policy="sequential",
                resource_strategy="comb_offset",
                save_time_clean=True,
                save_time_observed=True,
                cp_length=2,
            ),
            spectrum_config=ArraySpectrumConfig(
                enabled=False,
                zenith_bins=5,
                azimuth_bins=7,
            ),
        )
    )

    validate_hdf5_contract(path)
    with h5py.File(path, "r") as h5:
        assert h5["meta/schema_version"][()].decode("utf-8") == "2.0.0"
        assert h5["iq/cp_length"][()] == 2
        assert h5["iq/link/frequency_clean"].shape == h5["waveform/rx_grid"].shape
        assert h5["iq/link/frequency_observed"].shape == h5["waveform/rx_grid"].shape
        assert h5["iq/link/time_clean"].shape == (1, 2, 1, 2, 14 * 18)
        assert h5["iq/link/time_observed"].shape == (1, 2, 1, 2, 14 * 18)
        assert h5["iq/noncooperative/rx_time_clean"].shape == (1, 1, 1, 2, 14 * 18)
        assert h5["iq/noncooperative/rx_time_observed"].shape == (1, 1, 1, 2, 14 * 18)
        assert h5["iq/noncooperative/active_tx_indices"][()].tolist() == [[0, 1]]
        assert h5["iq/noncooperative/active_tx_positions_m"].shape == (1, 2, 3)
        assert not np.any(h5["iq/noncooperative/resource_collision_mask"][()])


def test_nr_srs_schema_requires_flattened_resource_symbol_indices(tmp_path: Path):
    path = run_rt_truth_pipeline(
        RTTruthRunConfig(
            label_file=Path("tests/fixtures/scenes/test/test5.json"),
            scene_file=Path("tests/fixtures/scenes/test/scene.xml"),
            output_dir=tmp_path / "nr_srs_schema_missing_re_symbol",
            num_subcarriers=8,
            seed=9,
            max_bs=1,
            max_ue=1,
            bs_num_rows=1,
            bs_num_cols=1,
            ue_num_rows=1,
            ue_num_cols=1,
            max_depth=1,
            observation_snr_db=80.0,
            phy_standard="nr_srs",
            srs_config=SimpleNamespace(
                slot_length_symbols=14,
                start_symbol=12,
                num_srs_symbols=1,
                comb_size=1,
                comb_offset=0,
                bwp_start_prb=0,
                bwp_num_prb=None,
                trigger_mode="aperiodic",
                periodicity_slots=1,
                slot_offset=0,
                slot_number=0,
                sequence_type="zc_like",
                sequence_id=0,
                group_hopping="disabled",
                sequence_hopping="disabled",
                cyclic_shift_multiplexing="cyclic_shift",
                cyclic_shift_indices=None,
            ),
        )
    )
    with h5py.File(path, "a") as h5:
        del h5["waveform/srs_re_symbol_indices"]

    with np.testing.assert_raises_regex(Exception, "srs_re_symbol_indices"):
        validate_hdf5_contract(path)
