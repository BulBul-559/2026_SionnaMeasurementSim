"""Basic system preflight helpers."""

from __future__ import annotations

import platform
import sys


def collect_basic_environment() -> dict[str, str]:
    """Return dependency-light environment details for CLI smoke checks."""

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
