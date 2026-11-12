"""Regression coverage for G8, added in revision r13.

The submitted toolkit had no test naming G8, ``G8_BELOW_FLOOR`` or
``alpha_reason_code`` -- the one rule the paper advertises as its correction to
an earlier exception had no regression coverage at all. These tests exist so that
the two defects found in review cannot come back:

1. G8 must test the *same* sigma that alpha uses. alpha is K/(a^2+K) with
   K = b^2 + c^2 + d^2, and a is the raw channel: kappa*sigma_freeboard on the
   freeboard channel, sigma_thickness on the posterior channel. The submitted
   code divided the declared *thickness* sigma by the floor on both channels, so
   the freeboard channel was cleared at 0.884 -- the posterior channel's number --
   while its own ratio is 0.12-0.24.

2. A non-finite ratio must be ``not_assessable``, never ``pass``. ``NaN < 0.5``
   and ``NaN < 1.0`` are both False, so the submitted three-way expression fell
   through to ``pass``.
"""
import numpy as np
import pytest

from ice_identifiability import BudgetPolicy, Priors, compute_budget
from ice_identifiability.gates import evaluate_gates
from ice_identifiability.ncio import Field, Grid, Product


def _product(h=1.0, sigma=0.7, snow=0.2, snow_unc=0.05, conc=100.0,
             fb=0.05, fb_unc=0.02, sigma_nan=False):
    shape = (1, 2, 2)
    grid = Grid(lat=np.full((2, 2), 70.0), lon=np.zeros((2, 2)), shape=(2, 2))

    def arr(x):
        return np.broadcast_to(np.asarray(x, dtype="float64"), shape).copy()

    sig = arr(np.nan if sigma_nan else sigma)
    return Product(
        label="synthetic", path="memory", time_labels=["2019-04"], grid=grid,
        h=Field("h", "m", 1.0, arr(h)),
        sigma=Field("sea_ice_thickness_uncertainty", "m", 1.0, sig),
        snow=Field("snow_depth", "m", 1.0, arr(snow)),
        snow_unc=Field("snow_depth_uncertainty", "m", 1.0, arr(snow_unc)),
        concentration=Field("concentration", "%", 1.0, arr(conc)),
        freeboard=Field("freeboard", "m", 1.0, arr(fb)),
        freeboard_unc=Field("freeboard_uncertainty", "m", 1.0, arr(fb_unc)),
        quality=None,
        var_names={"h": "sea_ice_thickness"},
        sha256="0" * 64,
    )


def _stats(**kw):
    base = {"nu_lag1": 0.2, "nu_lag2": 0.2, "nu_lag4": 0.2, "nu_lag8": 0.2,
            "block_excess_4": 20.0, "block_excess_8": 60.0, "block_excess_16": 200.0,
            "block_excess_primary": 60.0,
            "block_excess_free_4": 12.0, "block_excess_free_8": 45.0,
            "block_excess_free_16": 170.0, "block_excess_free_primary": 45.0,
            "block_cells": 8, "q_sigma": 0.5, "snr_median": 2.0, "n_valid": 100,
            "cell_km_x": 25.0, "cell_km_y": 25.0,
            "block_extent_km_x": 200.0, "block_extent_km_y": 200.0,
            "block_km_target": 200.0}
    base.update(kw)
    return base


def _g8(product, policy):
    budget = compute_budget(product, Priors(), policy)
    res = evaluate_gates(product, budget, _stats(block_excess_free_primary=45.0),
                         None, Priors(), policy)
    return next(g for g in res["gates"] if g["id"] == "G8"), res


def test_g8_uses_the_channel_own_sigma_not_the_declared_thickness_sigma():
    """The freeboard channel is not cleared on the posterior channel's ratio."""
    g_fb, _ = _g8(_product(), BudgetPolicy(raw_channel="freeboard"))
    g_post, _ = _g8(_product(), BudgetPolicy(raw_channel="posterior"))
    # a = kappa * sigma_fb = 8.78 * 0.02 = 0.176 m, against an unchanged floor,
    # so the freeboard ratio must be well below the posterior one.
    assert g_fb["value"] is not None and g_post["value"] is not None
    assert g_fb["value"] < 0.5 * g_post["value"]
    assert g_fb["status"] == "fail"
    assert g_post["status"] in ("warn", "pass")


def test_g8_ratio_equals_a_over_sqrt_k():
    """The reported value is exactly median(a / sqrt(b^2+c^2+d^2))."""
    product = _product()
    policy = BudgetPolicy(raw_channel="posterior")
    budget = compute_budget(product, Priors(), policy)
    expected = float(np.nanmedian(budget.comps["a"] / budget.sigma_floor))
    g8, _ = _g8(product, policy)
    assert g8["value"] == pytest.approx(expected, rel=1e-9)


def test_g8_fails_when_the_channel_is_far_below_its_floor():
    g8, res = _g8(_product(fb_unc=0.001), BudgetPolicy(raw_channel="freeboard"))
    assert g8["status"] == "fail"
    assert "below half its own nuisance floor" in g8["reason"]
    assert res["verdict"] != "assessable"
    assert res["alpha_reason_code"] == "G8_BELOW_FLOOR"


def test_g8_non_finite_ratio_is_not_assessable_never_pass():
    """A missing operand must not read as a pass -- the defect the paper claims to fix."""
    g8, res = _g8(_product(fb_unc=np.nan), BudgetPolicy(raw_channel="freeboard"))
    assert g8["status"] == "not_assessable"
    assert g8["status"] != "pass"
    assert res["verdict"] == "not_assessable"


def test_g8_threshold_string_records_all_three_bands():
    """The record must be readable without the code."""
    g8, _ = _g8(_product(), BudgetPolicy(raw_channel="posterior"))
    for token in ("1.0", "0.5", "warn", "fail"):
        assert token in g8["threshold"]


# --------------------------------------------------------------------------
# Band-wise evaluation.  K = b^2+c^2+d^2 is not a constant of the priors: c
# grows with h and d with h_s, so the floor rises northwards and a single
# domain-median ratio can hide a band where the raw channel is far below it.
# G8 therefore decides band by band and takes the worst one.
# --------------------------------------------------------------------------
def _banded_product(h_south=1.0, h_north=3.0, fb_unc=0.045):
    """144-cell grid, half in 60-70N and half in 70-80N, with h growing north."""
    ny, nx = 12, 12
    lat = np.empty((ny, nx))
    lat[:6] = 65.0
    lat[6:] = 76.0
    grid = Grid(lat=lat, lon=np.zeros((ny, nx)), shape=(ny, nx))

    def arr(v):
        return np.broadcast_to(np.asarray(v, dtype="float64"), (1, ny, nx)).copy()

    h = np.empty((ny, nx))
    h[:6] = h_south
    h[6:] = h_north
    return Product(
        label="banded", path="memory", time_labels=["2019-04"], grid=grid,
        h=Field("h", "m", 1.0, arr(h)),
        sigma=Field("sea_ice_thickness_uncertainty", "m", 1.0, arr(0.7)),
        snow=Field("snow_depth", "m", 1.0, arr(0.2)),
        snow_unc=Field("snow_depth_uncertainty", "m", 1.0, arr(0.05)),
        concentration=Field("concentration", "%", 1.0, arr(100.0)),
        freeboard=Field("freeboard", "m", 1.0, arr(0.05)),
        freeboard_unc=Field("freeboard_uncertainty", "m", 1.0, arr(fb_unc)),
        quality=None,
        var_names={"h": "sea_ice_thickness"},
        sha256="0" * 64,
    )


def test_g8_reports_a_ratio_for_every_populated_band():
    g8, _ = _g8(_banded_product(), BudgetPolicy(raw_channel="freeboard"))
    assert set(g8["inputs"]["ratio_by_band"]) == {"60-70", "70-80"}
    assert g8["inputs"]["deciding_band"] in ("60-70", "70-80")


def test_g8_takes_the_worst_band_not_the_domain_median():
    """A domain median would read 'warn'; the northern band is 'fail'."""
    product = _banded_product()
    policy = BudgetPolicy(raw_channel="freeboard")
    budget = compute_budget(product, Priors(), policy)
    g8, res = _g8(product, policy)
    by_band = g8["inputs"]["ratio_by_band"]
    assert by_band["70-80"] < by_band["60-70"]
    assert g8["inputs"]["deciding_band"] == "70-80"
    assert g8["value"] == pytest.approx(by_band["70-80"], rel=1e-12)
    # the pooled median over all cells, i.e. what the single-number version
    # reported, sits in the warn band and would have cleared the gate
    pooled = float(np.nanmedian(budget.comps["a"] / budget.sigma_floor))
    assert 0.5 <= pooled < 1.0
    assert g8["status"] == "fail"
    assert res["alpha_reason_code"] == "G8_BELOW_FLOOR"


def test_g8_band_ratio_equals_the_alpha_variance_share():
    """median(a/sqrt(K)) == sqrt(share_a/(1-share_a)) -- the identity a re-derivation
    from archived shares relies on, since median commutes with monotone maps."""
    product = _banded_product()
    policy = BudgetPolicy(raw_channel="freeboard")
    budget = compute_budget(product, Priors(), policy)
    g8, _ = _g8(product, policy)
    for i, name in enumerate(("60-70", "70-80")):
        sel = budget.mask & (np.asarray(product.grid.lat) == (65.0 if i == 0 else 76.0))
        share = float(np.nanmedian(budget.shares["a"][sel]))
        implied = float(np.sqrt(share / (1.0 - share)))
        assert g8["inputs"]["ratio_by_band"][name] == pytest.approx(implied, rel=1e-9)


def test_missing_declaration_outranks_an_unrealisable_block():
    """PIOMAS: G1 and G3 are both unassessable; the reason the user reads is G1.

    The ordering is a decision, not an accident: a provider can publish a
    sigma, and no file can repair a grid, so the actionable objection is
    reported first and the geometric one stays in its own gate record.
    """
    from ice_identifiability.gates import compose_verdict

    statuses = {f"G{i}": "pass" for i in range(9)}
    statuses["G1"] = statuses["G2"] = statuses["G3"] = "not_assessable"
    verdict, reason, code = compose_verdict(statuses, "block not realisable",
                                            "n/a", "product publishes no sigma")
    assert verdict == "not_assessable"
    assert code == "INPUT_NOT_ASSESSABLE"
    assert "does not declare the quantity the audit tests" in reason
    assert "block" not in reason


def test_unevaluable_floor_gets_its_own_reason_code():
    """not_assessable must not borrow the failure code."""
    from ice_identifiability.gates import compose_verdict

    statuses = {f"G{i}": "pass" for i in range(9)}
    statuses["G8"] = "not_assessable"
    verdict, _, code = compose_verdict(statuses, "", "ratio non-finite", "")
    assert verdict == "not_assessable"
    assert code == "G8_NOT_ASSESSABLE"


def test_missing_declaration_outranks_an_unrealisable_block():
    """PIOMAS: G1 and G3 are both unassessable; the reason the user reads is G1.

    The ordering is a decision, not an accident: a provider can publish a
    sigma, and no file can repair a grid, so the actionable objection is
    reported first and the geometric one stays in its own gate record.
    """
    from ice_identifiability.gates import compose_verdict

    statuses = {f"G{i}": "pass" for i in range(9)}
    statuses["G1"] = statuses["G2"] = statuses["G3"] = "not_assessable"
    verdict, reason, code = compose_verdict(statuses, "block not realisable",
                                            "n/a", "product publishes no sigma")
    assert verdict == "not_assessable"
    assert code == "INPUT_NOT_ASSESSABLE"
    assert "does not declare the quantity the audit tests" in reason
    assert "block" not in reason


def test_unevaluable_floor_gets_its_own_reason_code():
    """not_assessable must not borrow the failure code."""
    from ice_identifiability.gates import compose_verdict

    statuses = {f"G{i}": "pass" for i in range(9)}
    statuses["G8"] = "not_assessable"
    verdict, _, code = compose_verdict(statuses, "", "ratio non-finite", "")
    assert verdict == "not_assessable"
    assert code == "G8_NOT_ASSESSABLE"
