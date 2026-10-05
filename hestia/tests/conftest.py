"""Test fixtures and isolation for Hestia tests."""

from __future__ import annotations

import os
from typing import Generator
import pytest

from hestia.device_lock import _ACTIVE_LEASES


@pytest.fixture(autouse=True)
def isolation_fixture() -> Generator[None, None, None]:
    """Ensure clean auth environment and GPU lease state for each test."""
    old_token = os.environ.get("HESTIA_SERVICE_TOKEN")
    old_dev = os.environ.get("METIS_INSECURE_DEV")
    
    # Enable insecure-dev by default for legacy tests, tests specifically testing auth can override
    os.environ["METIS_INSECURE_DEV"] = "1"
    
    # Release any held device leases from prior tests
    for lease in list(_ACTIVE_LEASES.values()):
        try:
            lease.release()
        except Exception:
            pass
    _ACTIVE_LEASES.clear()

    yield

    # Clean up leases after test
    for lease in list(_ACTIVE_LEASES.values()):
        try:
            lease.release()
        except Exception:
            pass
    _ACTIVE_LEASES.clear()

    if old_token is not None:
        os.environ["HESTIA_SERVICE_TOKEN"] = old_token
    else:
        os.environ.pop("HESTIA_SERVICE_TOKEN", None)

    if old_dev is not None:
        os.environ["METIS_INSECURE_DEV"] = old_dev
    else:
        os.environ.pop("METIS_INSECURE_DEV", None)
