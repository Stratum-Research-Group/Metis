"""Tests for Prefix Executability Trace Metric (Item 114 / arXiv:2610.40444)."""

import pytest
from dokime.explainers.prefix_executability import (
    ActionStep,
    PrefixExecutabilityAuditor,
    PrefixExecutabilityReport,
)


def test_fully_executable_trace():
    """Verify metrics for an unbroken, valid trajectory."""
    trace = [
        ActionStep(step_index=0, action_name="list_files", arguments={"path": "/workspace"}),
        ActionStep(step_index=1, action_name="read_file", arguments={"path": "/workspace/main.py"}),
        ActionStep(step_index=2, action_name="run_tests", arguments={"suite": "unit"}),
        ActionStep(step_index=3, action_name="emit_verdict", arguments={"status": "PASS"}),
    ]

    report = PrefixExecutabilityAuditor.audit_trace(trace)
    assert report.total_steps == 4
    assert report.executable_steps == 4
    assert report.first_failure_index is None
    assert report.prefix_executability_ratio == 1.0
    assert report.strict_prefix_ratio == 1.0
    assert report.is_trace_fully_executable is True


def test_trace_with_intermediate_execution_failure():
    """Verify metrics when an intermediate step fails with an error code."""
    trace = [
        ActionStep(step_index=0, action_name="list_files"),
        ActionStep(step_index=1, action_name="read_file", return_code=1, error="FileNotFoundError"),
        ActionStep(step_index=2, action_name="read_file", return_code=0),
        ActionStep(step_index=3, action_name="emit_verdict"),
    ]

    report = PrefixExecutabilityAuditor.audit_trace(trace)
    assert report.total_steps == 4
    assert report.executable_steps == 3
    assert report.first_failure_index == 1
    assert report.prefix_executability_ratio == 0.75
    assert report.strict_prefix_ratio == 0.25  # Only step 0 before first failure
    assert report.is_trace_fully_executable is False

    failing_audit = report.step_audits[1]
    assert failing_audit.is_executable is False
    assert failing_audit.failure_category == "execution_error"
    assert "FileNotFoundError" in (failing_audit.diagnostic or "")


def test_precondition_violation():
    """Verify precondition failures are captured."""
    trace = [
        ActionStep(step_index=0, action_name="init_env"),
        ActionStep(
            step_index=1,
            action_name="modify_db",
            preconditions={"in_sandbox": False, "authenticated": True},
        ),
    ]

    report = PrefixExecutabilityAuditor.audit_trace(trace)
    assert report.total_steps == 2
    assert report.executable_steps == 1
    assert report.first_failure_index == 1
    assert report.prefix_executability_ratio == 0.5
    assert report.step_audits[1].failure_category == "precondition_violation"


def test_serialization():
    """Verify dictionary serialization."""
    trace = [ActionStep(step_index=0, action_name="ping")]
    report = PrefixExecutabilityAuditor.audit_trace(trace)
    d = report.to_dict()
    assert d["total_steps"] == 1
    assert d["is_trace_fully_executable"] is True
    assert len(d["step_audits"]) == 1
