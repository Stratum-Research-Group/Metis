"""Tests for policy schema validation."""

import json
from pathlib import Path
import pytest
import jsonschema


@pytest.fixture
def policy_and_schema():
    policy = json.loads(Path("krisis-policy-0.1.json").read_text(encoding="utf-8"))
    schema = json.loads(Path("krisis-policy.schema.json").read_text(encoding="utf-8"))
    return policy, schema


def test_draft_policy_validates(policy_and_schema):
    policy, schema = policy_and_schema
    assert policy["status"] == "draft"
    # Should validate cleanly without error
    jsonschema.validate(instance=policy, schema=schema)


def test_frozen_policy_requires_sealed_hash(policy_and_schema):
    policy, schema = policy_and_schema
    # Mutate to frozen while sha256 is null
    frozen_policy = dict(policy)
    frozen_policy["status"] = "frozen"
    frozen_policy["governance"]["sealed_test"]["sha256"] = None

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=frozen_policy, schema=schema)


def test_frozen_policy_with_valid_hash_validates(policy_and_schema):
    import copy
    policy, schema = policy_and_schema
    frozen_policy = copy.deepcopy(policy)
    frozen_policy["status"] = "frozen"
    
    # Governance sealed test fields
    sealed = frozen_policy["governance"]["sealed_test"]
    sealed["hash_recorded_before_tuning"] = True
    sealed["sha256"] = "a" * 64
    sealed["protocol_sha256"] = "b" * 64
    sealed["sealed_at_utc"] = "2026-09-30T12:00:00Z"
    sealed["custodian"] = "independent_custodian"

    # Calibration map fields for grounded_v1
    cal_map = frozen_policy["profiles"]["grounded_v1"]["score"]["cal_ll_map"]
    cal_map["status"] = "validated"
    cal_map["model_digest"] = "c" * 64
    cal_map["calibration_dataset_sha256"] = "d" * 64
    cal_map["calibration_report_sha256"] = "e" * 64

    jsonschema.validate(instance=frozen_policy, schema=schema)
