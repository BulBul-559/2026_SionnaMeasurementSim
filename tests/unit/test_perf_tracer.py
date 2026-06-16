import json
from pathlib import Path
from types import SimpleNamespace

from sionna_measurement_sim.perf import PerfTracer


def _debug_config(**overrides):
    values = {
        "enabled": True,
        "hardware_interval_s": 10.0,
        "link_log_interval": 2,
        "torch_synchronize": False,
        "write_hardware_samples": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_perf_tracer_summarizes_dataset_writes(tmp_path: Path):
    tracer = PerfTracer(tmp_path, _debug_config())
    tracer.start()

    with tracer.span("write"):
        tracer.record_event(
            "hdf5.dataset_write",
            path="/channel/truth/cfr",
            shape=(1, 2, 3),
            dtype="complex64",
            raw_bytes=48,
            storage_bytes=24,
            compression="gzip",
            compression_opts=1,
            duration_s=0.25,
        )
    summary = tracer.finish()

    assert summary["status"] == "success"
    assert summary["stage_totals_s"]["write"] >= 0.0
    write_summary = summary["dataset_write_summary"]
    assert write_summary["dataset_count"] == 1
    assert write_summary["total_raw_bytes"] == 48
    assert write_summary["total_storage_bytes"] == 24
    assert write_summary["raw_to_storage_ratio"] == 2.0
    assert write_summary["top_by_duration"][0]["path"] == "/channel/truth/cfr"
    assert write_summary["top_by_duration"][0]["compression_opts"] == 1

    on_disk = json.loads((tmp_path / "logs" / "perf_summary.json").read_text())
    assert on_disk["dataset_write_summary"]["dataset_count"] == 1


def test_perf_tracer_failure_summary_keeps_completed_stages(tmp_path: Path):
    tracer = PerfTracer(tmp_path, _debug_config())
    tracer.start()

    err = RuntimeError("boom")
    with tracer.span("completed_stage"):
        pass
    summary = tracer.finish(status="failed", exception=err)

    assert summary["status"] == "failed"
    assert summary["exception"] == {"type": "RuntimeError", "message": "boom"}
    assert "completed_stage" in summary["stage_totals_s"]
