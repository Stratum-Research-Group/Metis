"""Arbitrary precision decimal arithmetic with 6dp half-even rounding."""

from decimal import Decimal, ROUND_HALF_EVEN
from typing import Any

SIX_PLACES = Decimal("0.000001")


def to_decimal(val: str | int | float | Decimal | None) -> Decimal | None:
    if val is None:
        return None
    if isinstance(val, Decimal):
        return val
    return Decimal(str(val))


def render_6dp(val: Decimal | str | float) -> str:
    """Render decimal to exactly 6 places using ROUND_HALF_EVEN."""
    d = to_decimal(val)
    assert d is not None
    return str(d.quantize(SIX_PLACES, rounding=ROUND_HALF_EVEN))


def compute_cal_ll_from_raw(
    raw_ll_val: str | float | Decimal,
    ll_min_val: str | float | Decimal = "-10.000000",
    ll_max_val: str | float | Decimal = "0.000000",
) -> str:
    """Compute calibrated log-likelihood via affine mapping: clamp((raw_ll - ll_min)/(ll_max - ll_min), 0, 1)."""
    raw_ll = to_decimal(raw_ll_val)
    ll_min = to_decimal(ll_min_val)
    ll_max = to_decimal(ll_max_val)

    assert raw_ll is not None and ll_min is not None and ll_max is not None
    denom = ll_max - ll_min
    if denom == Decimal(0):
        norm = Decimal(0)
    else:
        norm = (raw_ll - ll_min) / denom

    clamped = max(Decimal(0), min(Decimal(1), norm))
    return render_6dp(clamped)


def compute_mean_claim_confidence(claims: list[Any]) -> Decimal:
    """Compute arithmetic mean of calibrated claim confidences in arbitrary precision."""
    if not claims:
        return Decimal(0)
    
    total = Decimal(0)
    for c in claims:
        conf_str = getattr(c, "calibrated_confidence", None) or (c.get("calibrated_confidence") if isinstance(c, dict) else None)
        if conf_str is None:
            raise ValueError(f"Claim missing calibrated_confidence: {c}")
        total += to_decimal(conf_str) # type: ignore

    return total / Decimal(len(claims))


def compute_candidate_score(
    profile_id: str,
    claims: list[Any],
    cal_ll_str: str | None,
    raw_ll_str: str | None,
    weights: dict[str, str] | None = None,
    cal_ll_map: dict[str, Any] | None = None,
) -> tuple[str | None, str | None]:
    """Compute candidate score rendered at 6dp half-even.
    
    Returns (score_str, rejection_reason).
    """
    if profile_id == "eligibility_baseline":
        return None, None

    if profile_id == "grounded_support_only":
        if not claims:
            return None, "NO_VALID_CANDIDATE"
        mean_conf = compute_mean_claim_confidence(claims)
        return render_6dp(mean_conf), None

    if profile_id == "grounded_v1":
        if not claims:
            return None, "NO_VALID_CANDIDATE"

        # Determine calibrated log-probability
        cal_ll_dec: Decimal | None = None
        if cal_ll_str is not None:
            cal_ll_dec = to_decimal(cal_ll_str)
        elif raw_ll_str is not None and cal_ll_map:
            ll_min = cal_ll_map.get("ll_min", "-10.000000")
            ll_max = cal_ll_map.get("ll_max", "0.000000")
            cal_ll_dec = to_decimal(compute_cal_ll_from_raw(raw_ll_str, ll_min, ll_max))

        if cal_ll_dec is None:
            # cal_ll required by grounded_v1 is missing
            return None, "FEATURE_INVALID"

        w = weights or {"w_conf": "0.700000", "w_ll": "0.300000"}
        w_conf = to_decimal(w.get("w_conf", "0.700000"))
        w_ll = to_decimal(w.get("w_ll", "0.300000"))

        assert w_conf is not None and w_ll is not None
        mean_conf = compute_mean_claim_confidence(claims)
        score_val = (w_conf * mean_conf) + (w_ll * cal_ll_dec)
        return render_6dp(score_val), None

    return None, f"Unknown profile: {profile_id}"
