"""Schema validation tests for NR PUSCH HDF5 output.

Validates that NR PUSCH output files contain all required MIMO-specific
fields with correct shapes and types.
"""

from pathlib import Path

import h5py
import numpy as np
import pytest

from sionna_measurement_sim.domain.array import ArraySpectrumConfig
from sionna_measurement_sim.io.schema_validator import validate_hdf5_contract


def _generate_nr_pusch_hdf5(tmp_path: Path, **kw) -> Path:
    """Run a minimal NR PUSCH pipeline and return the HDF5 path."""
    from sionna_measurement_sim.rt.truth_pipeline import (
        RTTruthRunConfig,
        run_rt_truth_pipeline,
    )

    defaults = dict(
        label_file=Path("data/scenes/test/test5.json"),
        scene_file=Path("data/scenes/test/scene.xml"),
        output_dir=tmp_path / "output_schema",
        num_subcarriers=48,
        seed=42,
        max_tx=1,
        max_rx=1,
        tx_num_rows=2,
        tx_num_cols=2,
        rx_num_rows=2,
        rx_num_cols=2,
        max_depth=3,
        los=True,
        specular_reflection=True,
        observation_snr_db=40.0,
        phy_standard="nr_pusch",
        num_prb=4,
        num_layers=1,
        num_antenna_ports=4,
        mcs_index=14,
        mcs_table=1,
        perfect_csi=True,
        receiver_failure_policy="fail_fast",
        mimo_detector="lmmse",
        channel_estimator="perfect",
    )
    defaults.update(kw)
    config = RTTruthRunConfig(**defaults)
    return run_rt_truth_pipeline(config)


class TestNRPUSCHSchema:
    """NR PUSCH-specific schema validation."""

    def test_waveform_standard_is_nr_pusch(self, tmp_path):
        try:
            path = _generate_nr_pusch_hdf5(tmp_path)
        except ImportError:
            pytest.skip("NR PUSCH receiver not available")
        except Exception as exc:
            pytest.fail(f"Pipeline failed: {exc}")

        with h5py.File(path, "r") as h5:
            std = h5["waveform/standard"][()]
            if isinstance(std, bytes):
                std = std.decode()
            assert std == "nr_pusch"

    def test_nr_pusch_waveform_fields_present(self, tmp_path):
        """NR PUSCH output must contain all NR-specific waveform fields."""
        try:
            path = _generate_nr_pusch_hdf5(tmp_path)
        except ImportError:
            pytest.skip("NR PUSCH receiver not available")
        except Exception as exc:
            pytest.fail(f"Pipeline failed: {exc}")

        with h5py.File(path, "r") as h5:
            required = [
                "waveform/num_prb",
                "waveform/subcarrier_spacing_khz",
                "waveform/num_layers",
                "waveform/num_antenna_ports",
                "waveform/mcs_index",
                "waveform/mcs_table",
                "waveform/dmrs_config_type",
                "waveform/dmrs_length",
                "waveform/dmrs_additional_position",
                "waveform/num_cdm_groups_without_data",
                "waveform/tx_grid",
                "waveform/rx_grid",
                "waveform/noise_variance",
                "array/rx_snapshot_matrix",
                "array/aoa_label_rad",
                "array/aoa_heatmap_label",
                "array/spatial_spectrum_label",
                "array/angle_grid_rad",
                "array/spectrum_policy",
                "receiver/mimo_detector",
                "receiver/receiver_type",
            ]
            for field in required:
                assert field in h5, f"Missing required NR PUSCH field: /{field}"
            assert "waveform/tx_time" not in h5
            assert "waveform/rx_time" not in h5

    def test_waveform_grid_and_array_attrs(self, tmp_path):
        """NR PUSCH waveform grids and array labels document shapes via attrs."""
        try:
            path = _generate_nr_pusch_hdf5(tmp_path)
        except ImportError:
            pytest.skip("NR PUSCH receiver not available")
        except Exception as exc:
            pytest.fail(f"Pipeline failed: {exc}")

        with h5py.File(path, "r") as h5:
            tx_grid = h5["waveform/tx_grid"]
            rx_grid = h5["waveform/rx_grid"]
            noise_variance = h5["waveform/noise_variance"]
            assert tx_grid.shape[:3] == rx_grid.shape[:3]
            assert noise_variance.shape == tx_grid.shape[:3]
            assert tx_grid.attrs["unit"] == "linear_complex"
            assert rx_grid.attrs["unit"] == "linear_complex"
            assert noise_variance.attrs["unit"] == "linear"
            assert tx_grid.attrs["index_order"] == (
                "snapshot,ul_tx,ul_rx,ul_tx_ant,ofdm_symbol,subcarrier"
            )
            assert rx_grid.attrs["index_order"] == (
                "snapshot,ul_tx,ul_rx,ul_rx_ant,ofdm_symbol,subcarrier"
            )
            assert noise_variance.attrs["index_order"] == "snapshot,ul_tx,ul_rx"

            snapshot_matrix = h5["array/rx_snapshot_matrix"]
            aoa = h5["array/aoa_label_rad"]
            heatmap = h5["array/aoa_heatmap_label"]
            spectrum = h5["array/spatial_spectrum_label"]
            angle_grid = h5["array/angle_grid_rad"]
            assert snapshot_matrix.shape == (*rx_grid.shape[:3], rx_grid.shape[3], rx_grid.shape[3])
            assert aoa.shape == (*rx_grid.shape[:3], 2)
            assert angle_grid.shape == (91, 181, 2)
            assert heatmap.shape == (*rx_grid.shape[:3], *angle_grid.shape[:2])
            assert spectrum.shape == (*rx_grid.shape[:3], *angle_grid.shape[:2])
            assert snapshot_matrix.attrs["index_order"] == (
                "snapshot,ul_tx,ul_rx,ul_rx_ant,ul_rx_ant"
            )
            assert heatmap.attrs["index_order"] == "snapshot,ul_tx,ul_rx,zenith,azimuth"
            assert spectrum.attrs["index_order"] == "snapshot,ul_tx,ul_rx,zenith,azimuth"
            assert angle_grid.attrs["index_order"] == "zenith,azimuth,angle_component"
            np.testing.assert_allclose(angle_grid[0, 0], [0.0, -np.pi])
            np.testing.assert_allclose(angle_grid[-1, -1], [np.pi, np.pi])

    def test_spectrum_enabled_writes_truth_and_observation_spectra(self, tmp_path):
        try:
            path = _generate_nr_pusch_hdf5(
                tmp_path,
                spectrum_config=ArraySpectrumConfig(
                    enabled=True,
                    sources=("truth_cfr", "rx_grid"),
                    zenith_bins=5,
                    azimuth_bins=7,
                ),
            )
        except ImportError:
            pytest.skip("NR PUSCH receiver not available")
        except Exception as exc:
            pytest.fail(f"Pipeline failed: {exc}")

        validate_hdf5_contract(path)
        with h5py.File(path, "r") as h5:
            link_shape = h5["waveform/rx_grid"].shape[:3]
            assert h5["array/angle_grid_rad"].shape == (5, 7, 2)
            assert h5["array/aoa_heatmap_label"].shape == (*link_shape, 5, 7)
            assert h5["array/spatial_spectrum_truth"].shape == (*link_shape, 5, 7)
            assert h5["array/spatial_spectrum_observation"].shape == (
                *link_shape,
                5,
                7,
            )
            assert np.all(np.isfinite(h5["array/spatial_spectrum_observation"][()]))

    def test_mimo_detector_not_empty(self, tmp_path):
        try:
            path = _generate_nr_pusch_hdf5(tmp_path)
        except ImportError:
            pytest.skip("NR PUSCH receiver not available")
        except Exception as exc:
            pytest.fail(f"Pipeline failed: {exc}")

        with h5py.File(path, "r") as h5:
            det = h5["receiver/mimo_detector"][()]
            if isinstance(det, bytes):
                det = det.decode()
            assert det, "mimo_detector must not be empty"
            assert det in ("lmmse", "kbest"), f"Unknown mimo_detector: {det!r}"

    def test_num_layers_positive(self, tmp_path):
        try:
            path = _generate_nr_pusch_hdf5(tmp_path)
        except ImportError:
            pytest.skip("NR PUSCH receiver not available")
        except Exception as exc:
            pytest.fail(f"Pipeline failed: {exc}")

        with h5py.File(path, "r") as h5:
            nl = int(h5["waveform/num_layers"][()])
            assert nl >= 1, f"num_layers must be >= 1, got {nl}"

    def test_num_antenna_ports_gte_num_layers(self, tmp_path):
        try:
            path = _generate_nr_pusch_hdf5(tmp_path)
        except ImportError:
            pytest.skip("NR PUSCH receiver not available")
        except Exception as exc:
            pytest.fail(f"Pipeline failed: {exc}")

        with h5py.File(path, "r") as h5:
            nl = int(h5["waveform/num_layers"][()])
            nap = int(h5["waveform/num_antenna_ports"][()])
            assert nap >= nl, (
                f"num_antenna_ports ({nap}) must be >= num_layers ({nl})"
            )

    def test_4x4_cfr_est_shape_consistency(self, tmp_path):
        """cfr_est.shape[1:] must match truth_cfr.shape for 4x4."""
        try:
            path = _generate_nr_pusch_hdf5(tmp_path, num_antenna_ports=4)
        except ImportError:
            pytest.skip("NR PUSCH receiver not available")
        except Exception as exc:
            pytest.fail(f"Pipeline failed: {exc}")

        with h5py.File(path, "r") as h5:
            truth = h5["channel/truth/cfr"]
            cfr_est = h5["observation/cfr_est"]
            assert cfr_est.shape[1:] == truth.shape, (
                f"cfr_est.shape[1:]={cfr_est.shape[1:]} != truth.shape={truth.shape}"
            )
            # Verify 4x4 antenna dimensions
            assert truth.shape[2] == 4  # rx_ant
            assert truth.shape[3] == 4  # tx_ant

    def test_receiver_type_is_pusch_receiver(self, tmp_path):
        try:
            path = _generate_nr_pusch_hdf5(tmp_path)
        except ImportError:
            pytest.skip("NR PUSCH receiver not available")
        except Exception as exc:
            pytest.fail(f"Pipeline failed: {exc}")

        with h5py.File(path, "r") as h5:
            rt = h5["receiver/receiver_type"][()]
            if isinstance(rt, bytes):
                rt = rt.decode()
            assert rt == "pusch_receiver", f"receiver_type={rt!r}"

    def test_link_group_complete(self, tmp_path):
        try:
            path = _generate_nr_pusch_hdf5(tmp_path)
        except ImportError:
            pytest.skip("NR PUSCH receiver not available")
        except Exception as exc:
            pytest.fail(f"Pipeline failed: {exc}")

        with h5py.File(path, "r") as h5:
            link = h5["link"]
            for field in (
                "duplex_mode", "phy_link_direction", "rt_trace_direction",
                "reciprocity_mode", "reciprocity_applied",
            ):
                assert field in link, f"Missing /link/{field}"
