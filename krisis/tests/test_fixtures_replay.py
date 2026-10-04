"""Integration test replaying all 16 fixtures from krisis-fixtures-0.1.json."""

from krisis.audit.replay import FixtureReplayRunner


def test_all_fixtures_pass():
    runner = FixtureReplayRunner(
        policy_path="krisis-policy-0.1.json",
        fixtures_path="krisis-fixtures-0.1.json",
    )
    summary = runner.run_all()

    assert summary["all_passed"] is True, f"Failed fixtures: {[r['id'] for r in summary['results'] if not r['passed']]}"
    assert summary["passed_count"] == 16
    assert summary["failed_count"] == 0
