"""NR PUSCH observation integration tests per doc 13.

Replaces file-reading tests with a self-generating pipeline run
that validates end-to-end HDF5 output.
"""

from pathlib import Path

import h5py
import numpy as np


class TestNRPUSCHIntegration:
    def test_nr_pusch_pipeline_self_generated(self, tmp_path: Path):
        from sionna_measurement_sim.rt.truth_pipeline import (
            RTTruthRunConfig,
            run_rt_truth_pipeline,
        )

        config = RTTruthRunConfig(
            label_file=Path("tests/fixtures/scenes/test/test5.json"),
            scene_file=Path("tests/fixtures/scenes/test/scene.xml"),
            output_dir=tmp_path / "output",
            num_subcarriers=48,
            seed=1,
            max_bs=2,
            max_ue=2,
            bs_num_rows=1,
            bs_num_cols=1,
            ue_num_rows=1,
            ue_num_cols=1,
            max_depth=0,
            specular_reflection=False,
            observation_snr_db=30.0,
            phy_standard="nr_pusch",
            num_prb=4,
            num_layers=1,
            num_antenna_ports=1,
            mcs_index=14,
            mcs_table=1,
        )
        path = run_rt_truth_pipeline(config)

        with h5py.File(path, "r") as h5:
            assert "link" in h5, "link group missing"
            assert "evaluation" in h5, "evaluation group missing"
            ber = h5["evaluation/ber"][()]
            bler = h5["evaluation/bler"][()]
            assert np.isfinite(float(ber)), f"ber not finite: {ber}"
            assert np.isfinite(float(bler)), f"bler not finite: {bler}"
            assert h5["waveform/standard"][()].decode() == "nr_pusch"
            assert h5["receiver/receiver_type"][()].decode() == "pusch_receiver"
            nmse = h5["evaluation/nmse_db"][()]
            assert np.all(np.isfinite(nmse)), "nmse_db has non-finite values"
