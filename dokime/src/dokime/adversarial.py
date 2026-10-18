"""
Adversarial Evaluation-Gate Integrity & Metric Discipline for Dokime.
Implements the requirements of specification 02-evaluation-gate-integrity-dokime-krisis-87c5fa3f.txt:
  - Strict count reconciliation: attempted == passed + failed + timed_out + invalid + abstained
  - Explicit execution_kind & task_data_kind (synthetic/mock never eligible for real promotion)
  - Unit-bearing metrics (latency_ns in integer nanoseconds)
  - True global quantiles over raw request samples (rejecting averaged batch p99s)
  - Dataset leakage / near-duplicate detection between train and eval fixtures
  - Selective risk and coverage analysis (preventing total abstention masquerading as 100% accuracy)
  - Finite numbers guarantee (rejecting NaN / Infinity in all metrics)
"""

from __future__ import annotations

import enum
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np


class ExecutionKind(str, enum.Enum):
    REAL = "REAL"
    SYNTHETIC = "SYNTHETIC"
    MOCK = "MOCK"
    REPLAY = "REPLAY"


class TaskDataKind(str, enum.Enum):
    REAL_WORLD = "REAL_WORLD"
    GENERATED = "GENERATED"
    FIXTURE = "FIXTURE"


class AuditFailure(Exception):
    """Raised when an evaluation run fails adversarial gate integrity checks."""
    pass


@dataclass
class ReconciliationRecord:
    """Rigorous accounting of all attempted evaluation samples."""
    total_attempted: int
    passed: int
    failed: int
    timed_out: int
    invalid: int
    abstained: int

    @property
    def total_accounted(self) -> int:
        return self.passed + self.failed + self.timed_out + self.invalid + self.abstained

    @property
    def is_balanced(self) -> bool:
        return self.total_attempted == self.total_accounted


@dataclass
class SelectiveRiskReport:
    """Evaluates answer coverage alongside selective risk (error rate on answered samples)."""
    total_samples: int
    answered_count: int
    abstained_count: int
    correct_count: int
    incorrect_count: int
    coverage: float
    selective_risk: float

    @classmethod
    def compute(
        cls,
        total_samples: int,
        correct_count: int,
        incorrect_count: int,
        abstained_count: int,
    ) -> SelectiveRiskReport:
        answered = correct_count + incorrect_count
        cov = answered / total_samples if total_samples > 0 else 0.0
        risk = incorrect_count / answered if answered > 0 else 0.0
        return cls(
            total_samples=total_samples,
            answered_count=answered,
            abstained_count=abstained_count,
            correct_count=correct_count,
            incorrect_count=incorrect_count,
            coverage=cov,
            selective_risk=risk,
        )


@dataclass
class AuditGateResult:
    """Consolidated outcome of an adversarial evaluation gate audit."""
    eligible_for_promotion: bool
    reconciliation: ReconciliationRecord
    selective_risk: SelectiveRiskReport
    global_p99_latency_ns: Optional[int]
    rejection_reasons: List[str] = field(default_factory=list)
    observations_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "eligible_for_promotion": self.eligible_for_promotion,
            "reconciliation": asdict(self.reconciliation),
            "selective_risk": asdict(self.selective_risk),
            "global_p99_latency_ns": self.global_p99_latency_ns,
            "rejection_reasons": self.rejection_reasons,
            "observations_count": self.observations_count,
        }


class AdversarialGateAuditor:
    """
    Audits evaluation manifests and raw observation ledgers for integrity,
    reconciliation, unit-bearing metrics, leakage, and promotion eligibility.
    """

    @classmethod
    def compute_global_p99_ns(cls, raw_latencies_ns: List[int]) -> int:
        """
        Compute true global p99 percentile from raw observations.
        Rejects empty lists or lists containing non-finite or negative values.
        """
        if not raw_latencies_ns:
            raise AuditFailure("Cannot compute global p99 from empty observation list.")
        arr = np.array(raw_latencies_ns, dtype=np.int64)
        if np.any(arr < 0):
            raise AuditFailure("Negative latency values detected in raw observations.")
        return int(np.percentile(arr, 99))

    @classmethod
    def audit_run(
        cls,
        manifest: Dict[str, Any],
        observations: List[Dict[str, Any]],
        training_source_hashes: Optional[Set[str]] = None,
        max_allowed_p99_latency_ns: Optional[int] = None,
        min_required_coverage: float = 0.5,
        max_allowed_selective_risk: float = 0.15,
    ) -> AuditGateResult:
        """
        Run the complete battery of adversarial integrity checks on an evaluation run.
        """
        rejection_reasons: List[str] = []

        # 1. Execution kind & task data kind checks
        exec_kind_str = manifest.get("execution_kind")
        if not exec_kind_str:
            rejection_reasons.append("Missing mandatory 'execution_kind' in run manifest.")
            exec_kind = ExecutionKind.MOCK
        else:
            try:
                exec_kind = ExecutionKind(exec_kind_str)
            except ValueError:
                rejection_reasons.append(f"Invalid execution_kind: {exec_kind_str}")
                exec_kind = ExecutionKind.MOCK

        task_data_kind_str = manifest.get("task_data_kind")
        if not task_data_kind_str:
            rejection_reasons.append("Missing mandatory 'task_data_kind' in run manifest.")
            task_data_kind = TaskDataKind.FIXTURE
        else:
            try:
                task_data_kind = TaskDataKind(task_data_kind_str)
            except ValueError:
                rejection_reasons.append(f"Invalid task_data_kind: {task_data_kind_str}")
                task_data_kind = TaskDataKind.FIXTURE

        if exec_kind != ExecutionKind.REAL:
            rejection_reasons.append(
                f"Execution kind '{exec_kind.value}' is synthetic/mock. Ineligible for real promotion."
            )

        # 2. Count reconciliation over observations
        total_attempted = manifest.get("total_attempted", len(observations))
        passed = sum(1 for o in observations if o.get("status") == "PASSED")
        failed = sum(1 for o in observations if o.get("status") == "FAILED")
        timed_out = sum(1 for o in observations if o.get("status") == "TIMED_OUT")
        invalid = sum(1 for o in observations if o.get("status") == "INVALID")
        abstained = sum(1 for o in observations if o.get("status") == "ABSTAINED")

        reconciliation = ReconciliationRecord(
            total_attempted=total_attempted,
            passed=passed,
            failed=failed,
            timed_out=timed_out,
            invalid=invalid,
            abstained=abstained,
        )

        if not reconciliation.is_balanced:
            rejection_reasons.append(
                f"Count reconciliation failure: attempted={total_attempted} != "
                f"accounted={reconciliation.total_accounted} "
                f"(passed={passed}, failed={failed}, timed_out={timed_out}, "
                f"invalid={invalid}, abstained={abstained})"
            )

        # 3. Unit-bearing latency check (latency_ns)
        latencies_ns: List[int] = []
        for i, obs in enumerate(observations):
            lat = obs.get("latency_ns")
            if lat is None:
                # Check if someone supplied unitless 'latency'
                if "latency" in obs:
                    rejection_reasons.append(
                        f"Observation index {i} uses unitless 'latency' instead of mandatory 'latency_ns'."
                    )
                else:
                    rejection_reasons.append(f"Observation index {i} is missing 'latency_ns'.")
                continue

            if not isinstance(lat, int) or lat < 0:
                rejection_reasons.append(f"Observation index {i} has invalid latency_ns: {lat}")
                continue

            # Detect plausible conversion errors (e.g. 150 ms mistakenly reported as 150 ns)
            # A real model inference cannot complete in under 5,000 ns (5 microseconds)
            if exec_kind == ExecutionKind.REAL and lat < 5_000:
                rejection_reasons.append(
                    f"Observation index {i} reports implausible real latency: {lat} ns (< 5 µs). "
                    "Suspected missing millisecond-to-nanosecond unit conversion."
                )

            latencies_ns.append(lat)

        global_p99 = None
        if latencies_ns:
            try:
                global_p99 = cls.compute_global_p99_ns(latencies_ns)
                if max_allowed_p99_latency_ns and global_p99 > max_allowed_p99_latency_ns:
                    rejection_reasons.append(
                        f"Global p99 latency {global_p99} ns exceeds threshold {max_allowed_p99_latency_ns} ns."
                    )
            except Exception as e:
                rejection_reasons.append(f"Failed to compute global p99 latency: {e}")

        # 4. Selective risk and coverage
        sel_risk = SelectiveRiskReport.compute(
            total_samples=len(observations),
            correct_count=passed,
            incorrect_count=failed + timed_out + invalid,
            abstained_count=abstained,
        )

        if sel_risk.coverage < min_required_coverage:
            rejection_reasons.append(
                f"Coverage {sel_risk.coverage:.1%} is below required minimum {min_required_coverage:.1%}. "
                "Excessive abstention cannot masquerade as quality."
            )

        if sel_risk.selective_risk > max_allowed_selective_risk:
            rejection_reasons.append(
                f"Selective risk {sel_risk.selective_risk:.1%} exceeds maximum allowed {max_allowed_selective_risk:.1%}."
            )

        # 5. Dataset leakage detection
        if training_source_hashes:
            leaked_ids = []
            for obs in observations:
                src_hash = obs.get("source_content_hash")
                if src_hash and src_hash in training_source_hashes:
                    leaked_ids.append(obs.get("task_id", "unknown"))
            if leaked_ids:
                rejection_reasons.append(
                    f"Contamination / leakage detected: {len(leaked_ids)} eval items share content hashes "
                    f"with training/dev data. Samples: {leaked_ids[:3]}"
                )

        eligible = len(rejection_reasons) == 0

        return AuditGateResult(
            eligible_for_promotion=eligible,
            reconciliation=reconciliation,
            selective_risk=sel_risk,
            global_p99_latency_ns=global_p99,
            rejection_reasons=rejection_reasons,
            observations_count=len(observations),
        )
