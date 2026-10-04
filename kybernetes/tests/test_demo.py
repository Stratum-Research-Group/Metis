"""Integration tests for Kybernetes demo scenarios."""

import pytest
from kybernetes.demo import run_all_demos


@pytest.mark.asyncio
async def test_demo_runner_completes_cleanly():
    await run_all_demos()
