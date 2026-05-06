"""Unit tests for shape_contracts.py TX-first conversions."""

import numpy as np
import pytest

from sionna_measurement_sim.adapters.sionna_rt.shape_contracts import (
    assert_tx_first_scalar,
    to_project_cfr,
    to_project_cir,
    to_project_path_interaction,
    to_project_path_scalar,
    to_project_path_vertices,
)


class TestCFRShape:
    def test_single_time_step_returns_5d(self):
        raw = np.ones((1, 1, 1, 1, 1, 8), dtype=np.complex64)
        cfr, snaps = to_project_cfr(raw, num_time_steps=1)
        assert cfr.shape == (1, 1, 1, 1, 8)
        assert snaps is None

    def test_multi_time_step_returns_snapshots(self):
        raw = np.ones((1, 1, 1, 1, 3, 8), dtype=np.complex64)
        cfr, snaps = to_project_cfr(raw, num_time_steps=3)
        assert cfr.shape == (1, 1, 1, 1, 8)
        assert snaps is not None
        assert snaps.shape == (3, 1, 1, 1, 1, 8)


class TestCIRShape:
    def test_single_snapshot_6d(self):
        a = np.ones((1, 1, 1, 1, 4, 1), dtype=np.complex64)  # 1 time step
        tau = np.ones((1, 1, 1, 1, 4), dtype=np.float32)
        valid = np.ones((1, 1, 1, 1, 4), dtype=np.bool_)
        coeff, delays, v = to_project_cir(a, tau, valid, num_time_steps=1)
        assert coeff.shape == (1, 1, 1, 1, 1, 4)
        assert delays.shape == (1, 1, 1, 1, 1, 4)
        assert v.shape == (1, 1, 1, 1, 1, 4)

    def test_multi_snapshot_6d(self):
        a = np.ones((1, 1, 1, 1, 4, 3), dtype=np.complex64)
        tau = np.ones((1, 1, 1, 1, 4), dtype=np.float32)
        valid = np.ones((1, 1, 1, 1, 4), dtype=np.bool_)
        coeff, delays, v = to_project_cir(a, tau, valid, num_time_steps=3)
        assert coeff.shape == (3, 1, 1, 1, 1, 4)
        assert delays.shape == (3, 1, 1, 1, 1, 4)


class TestPathScalarShape:
    def test_5d_scalar(self):
        val = np.ones((1, 1, 1, 1, 4), dtype=np.float32)
        result = to_project_path_scalar(val, "test")
        assert result.shape == (1, 1, 1, 1, 4)

    def test_4x4_mimo_scalar(self):
        val = np.ones((4, 4, 2, 4, 10), dtype=np.float32)
        result = to_project_path_scalar(val, "test")
        assert result.shape == (2, 4, 4, 4, 10)  # tx=2, rx=4, rx_ant=4, tx_ant=4


class TestPathInteractionShape:
    def test_6d_interaction(self):
        val = np.ones((2, 1, 1, 1, 1, 4), dtype=np.uint32)
        result = to_project_path_interaction(val, "test")
        assert result.shape == (1, 1, 1, 1, 4, 2)


class TestPathVerticesShape:
    def test_7d_vertices(self):
        val = np.ones((2, 1, 1, 1, 1, 4, 3), dtype=np.float32)
        result = to_project_path_vertices(val)
        assert result.shape == (1, 1, 1, 1, 4, 2, 3)


class TestAssertions:
    def test_valid_scalar_passes(self):
        arr = np.ones((2, 3, 4, 4, 10), dtype=np.float32)
        assert_tx_first_scalar(arr, "test")  # should not raise

    def test_wrong_rank_fails(self):
        arr = np.ones((2, 3, 10), dtype=np.float32)
        with pytest.raises(ValueError):
            assert_tx_first_scalar(arr, "test")
