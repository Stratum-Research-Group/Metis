"""Tests for 6-decimal precision scoring and affine mapping."""

from decimal import Decimal
from krisis.contracts.candidate import Claim
from krisis.scoring.decimal_calc import (
    compute_cal_ll_from_raw,
    compute_candidate_score,
    render_6dp,
    to_decimal,
)


def test_6dp_half_even_rounding():
    assert render_6dp("0.1234564") == "0.123456"
    assert render_6dp("0.1234566") == "0.123457"
    # Half-even test: 0.1234565 rounds to even (6), 0.1234575 rounds to even (8)
    assert render_6dp("0.1234565") == "0.123456"
    assert render_6dp("0.1234575") == "0.123458"


def test_affine_logprob_map():
    # clamp((raw - (-10)) / 10, 0, 1)
    assert compute_cal_ll_from_raw("-0.500000") == "0.950000"
    assert compute_cal_ll_from_raw("-2.000000") == "0.800000"
    assert compute_cal_ll_from_raw("-8.000000") == "0.200000"
    # Clamping
    assert compute_cal_ll_from_raw("-15.000000") == "0.000000"
    assert compute_cal_ll_from_raw("2.000000") == "1.000000"


def test_weighted_grounded_v1_score():
    claims = [
        Claim(claim_id="cl1", verdict="supported", calibrated_confidence="0.900000"),
    ]
    # 0.7 * 0.900000 + 0.3 * 0.800000 = 0.630000 + 0.240000 = 0.870000
    score, err = compute_candidate_score(
        profile_id="grounded_v1",
        claims=claims,
        cal_ll_str="0.800000",
        raw_ll_str=None,
    )
    assert err is None
    assert score == "0.870000"


def test_missing_cal_ll_causes_feature_invalid():
    claims = [
        Claim(claim_id="cl1", verdict="supported", calibrated_confidence="0.900000"),
    ]
    score, err = compute_candidate_score(
        profile_id="grounded_v1",
        claims=claims,
        cal_ll_str=None,
        raw_ll_str=None,
    )
    assert score is None
    assert err == "FEATURE_INVALID"
