"""Basic system preflight helpers."""

from __future__ import annotations

import importlib.metadata as metadata
import platform
import sys


def collect_basic_environment() -> dict[str, str | bool]:
    """Return dependency-light environment details for CLI smoke checks."""

    sionna_version = _version_or_empty("sionna")
    sionna_rt_version = _version_or_empty("sionna-rt")
    torch_version = _version_or_empty("torch")
    mitsuba_version = _version_or_empty("mitsuba")
    drjit_version = _version_or_empty("drjit")

    cuda_available = False
    cuda_device_name = ""
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        if cuda_available:
            cuda_device_name = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "sionna_version": sionna_version,
        "sionna_rt_version": sionna_rt_version,
        "torch_version": torch_version,
        "mitsuba_version": mitsuba_version,
        "drjit_version": drjit_version,
        "cuda_available": cuda_available,
        "cuda_device_name": cuda_device_name,
    }


def _version_or_empty(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return ""
