import numpy as np
import pandas as pd
import pytest

from ice_identifiability import BudgetPolicy, Priors, compute_budget
from ice_identifiability.audit import (ALPHA_COLUMNS, BUDGET_COLUMNS, run_audit,
                                       write_empty_tables)
from ice_identifiability.budget import band_of
from ice_identifiability.cli import main
from ice_identifiability.ncio import Field, Grid, NotAssessable, Product
from ice_identifiability.gates import _sigma_as_error_scale, evaluate_gates
from ice_identifiability.structure import (nugget_ratio, snr_median, block_excess,
                                           block_excess_free)


def _product(h, sigma, snow, snow_unc, conc, fb=None, fb_unc=None):
    shape = (1, 2, 2)
    grid = Grid(lat=np.full((2, 2), 70.0), lon=np.zeros((2, 2)), shape=(2, 2))

    def arr(x):
        return np.broadcast_to(np.asarray(x, dtype="float64"), shape).copy()

    return Product(
        label="synthetic", path="memory", time_labels=["2019-04"], grid=grid,
        h=Field("h", "m", 1.0, arr(h)),
        sigma=(Field("u", "m", 1.0, arr(sigma)) if sigma is not None else None),
        snow=(Field("s", "m", 1.0, arr(snow)) if snow is not None else None),
        snow_unc=(Field("su", "m", 1.0, arr(snow_unc)) if snow_unc is not None else None),
        freeboard=(Field("fb", "m", 1.0, arr(fb)) if fb is not None else None),
        freeboard_unc=(Field("fbu", "m", 1.0, arr(fb_unc)) if fb_unc is not None else None),
        concentration=Field("c", "%", 1.0, arr(conc)),
        quality=None, var_names={}, sha256="",
    )


def test_budget_algebra_and_shares_sum_to_one():
    p = _product(h=1.0, sigma=0.2, snow=0.2, snow_unc=0.05, conc=100,
                 fb=0.3, fb_unc=0.01)
    pri = Priors()
    b = compute_budget(p, pri, BudgetPolicy())
    a = pri.kappa * 0.01
    post2 = a ** 2 + (pri.snow_sensitivity * 0.05) ** 2 \
        + (1.0 / pri.deff * pri.sig_rho_i) ** 2 + (0.2 / pri.deff * pri.sig_rho_s) ** 2 \
        + (pri.eps_sd * pri.kappa * (pri.refraction_n0 - 1.0) * 0.2) ** 2
    assert b.alpha_object == "nuisance_share_of_freeboard_channel"
    assert np.allclose(b.alpha[0, 0, 0], 1 - a ** 2 / post2)
    assert abs(sum(b.shares[k][0, 0, 0] for k in "abcd") - 1.0) < 1e-12


def test_posterior_object_label_and_alpha_low_for_large_declared_sigma():
    p = _product(h=1.0, sigma=0.7, snow=0.2, snow_unc=0.05, conc=100)
    b = compute_budget(p, Priors(), BudgetPolicy(raw_channel="posterior"))
    assert b.alpha_object == "nuisance_share_of_declared_posterior"
    assert 0.0 < b.alpha[0, 0, 0] < 1.0


def test_missing_snow_refuses_to_default():
    p = _product(h=1.0, sigma=0.2, snow=None, snow_unc=None, conc=100)
    with pytest.raises(NotAssessable):
        compute_budget(p, Priors(), BudgetPolicy())


def test_white_noise_is_rejected_by_structure():
    rng = np.random.default_rng(0)
    h = rng.normal(1.0, 0.3, size=(1, 200, 200))
    mask = np.ones_like(h, dtype=bool)
    sigma = np.full_like(h, 0.3)
    h[~mask] = np.nan
    nu = nugget_ratio(h, mask, 1)
    assert 0.9 < nu < 1.1
    assert block_excess(h, mask, sigma, 8) < 1.3
    assert abs(snr_median(h, mask, sigma) - 3.33) < 0.2


def test_smooth_field_has_small_nugget():
    x = np.arange(200)
    h = (np.sin(x / 20.0)[None, :, None] * np.ones((1, 200, 1))).repeat(200, axis=2)
    h = h + 1.5
    mask = np.ones_like(h, dtype=bool)
    sigma = np.full_like(h, 0.3)
    assert nugget_ratio(h, mask, 1) < 0.05


def test_band_of():
    lat = np.array([55.0, 65.0, 75.0, 85.0])
    out = band_of(lat)
    assert list(out) == ["", "60-70", "70-80", "80+"]


def test_sigma_semantics_detects_population_sd():
    """ATL20 式：字段自称 standard deviation，不是 standard error of the mean。"""
    from ice_identifiability.gates import _sigma_semantics

    p = _product(1.0, 0.02, 0.1, 0.02, 90.0)
    p.sigma.attrs["long_name"] = "Monthly standard deviation of freeboard height"
    sem, reason = _sigma_semantics(p)
    assert sem == "population_sd"
    assert "not a standard error" in reason


def test_sigma_semantics_keeps_standard_error_pass():
    from ice_identifiability.gates import _sigma_semantics

    p = _product(1.0, 0.02, 0.1, 0.02, 90.0)
    p.sigma.attrs["standard_name"] = "sea_ice_thickness standard_error"
    sem, _ = _sigma_semantics(p)
    assert sem == "declared_1sigma"


def test_sigma_free_block_excess_and_the_error_scale_subcheck():
    """The sigma-free excess has a null of 1 on any grid, and the G5 sub-check
    must answer *not assessable* - never *fail* - when sigma is absent or is
    not a per-cell noise scale."""
    rng = np.random.default_rng(1)
    for n_cells in (4, 8, 16):          # same statistic, three cell sizes
        h = rng.normal(0.0, 1.0, size=(1, 64 * n_cells, 64 * n_cells))
        mask = np.ones_like(h, dtype=bool)
        be_free = block_excess_free(h, mask, n_cells)
        assert 0.8 < be_free < 1.3, (n_cells, be_free)

    # a field that only varies between blocks saturates near n_cells
    blocky = np.repeat(np.repeat(rng.normal(0, 1, (1, 8, 8)), 8, axis=1), 8, axis=2)
    assert block_excess_free(blocky, np.ones_like(blocky, dtype=bool), 8) > 20

    # sigma absent -> the sub-check has nothing to divide by
    status, value, _ = _sigma_as_error_scale(
        {"block_excess_primary": None, "block_excess_free_primary": 30.0,
         "snr_median": 2.0})
    assert (status, value) == ("not_assessable", None)
    # sigma larger than the field's own variance -> not a noise scale
    status, value, reason = _sigma_as_error_scale(
        {"block_excess_primary": 0.5, "block_excess_free_primary": 30.0,
         "snr_median": 2.0})
    assert status == "not_assessable" and value == 60.0
    assert "cannot be a per-cell noise scale" in reason
    # a genuine per-cell error leaves most of the variance to structure
    status, value, _ = _sigma_as_error_scale(
        {"block_excess_primary": 40.0, "block_excess_free_primary": 30.0,
         "snr_median": 2.0})
    assert status == "pass" and value == pytest.approx(0.75)


def _stats(**over):
    """A well-behaved structure dict so gate tests can break one field at a time."""
    s = {"nu_lag1": 0.2, "nu_lag8": 0.2, "q_sigma": 1.2, "snr_median": 3.0,
         "block_cells": 8, "block_extent_km_x": 200.0, "block_extent_km_y": 200.0,
         "block_excess_primary": 50.0, "block_excess_free_primary": 50.0}
    s.update(over)
    return s


def test_unmeasurable_block_statistic_is_not_assessable_not_unidentifiable():
    """R07: a NaN block statistic means "cannot measure", not "white noise"."""
    p = _product(h=1.0, sigma=0.2, snow=0.2, snow_unc=0.05, conc=100)
    b = compute_budget(p, Priors(), BudgetPolicy())
    res = evaluate_gates(p, b, _stats(block_excess_free_primary=float("nan")),
                         None, Priors(), BudgetPolicy())
    g3 = next(g for g in res["gates"] if g["id"] == "G3")
    assert g3["status"] == "not_assessable"
    assert "cannot be measured" in g3["reason"]
    assert res["verdict"] == "not_assessable"
    assert res["verdict"] != "unidentifiable"


def test_unidentifiable_claim_names_only_the_clause_that_failed():
    """R18: the text must enumerate the failed clauses, not assume both failed."""
    p = _product(h=1.0, sigma=0.2, snow=0.2, snow_unc=0.05, conc=100)
    b = compute_budget(p, Priors(), BudgetPolicy())
    res = evaluate_gates(p, b, _stats(block_excess_free_primary=0.5),
                         None, Priors(), BudgetPolicy())
    assert res["verdict"] == "unidentifiable"
    claims = [c for c in res["claims_not_supported"]
              if c["claim"] == "the product resolves Arctic sea-ice thickness"]
    assert claims and claims[0]["status"] == "not supported"
    why = claims[0]["why"]
    assert "block_excess_free" in why and "nu_lag1" not in why


def test_g5_warn_still_records_the_standard_error_claim():
    """R11: a population sd / maximum is a warn, but the claim must be withheld."""
    p = _product(1.0, 0.02, 0.1, 0.02, 90.0)
    p.sigma.attrs["long_name"] = "Monthly standard deviation of thickness"
    b = compute_budget(p, Priors(), BudgetPolicy())
    res = evaluate_gates(p, b, _stats(), None, Priors(), BudgetPolicy())
    g5 = next(g for g in res["gates"] if g["id"] == "G5")
    assert g5["status"] == "warn"
    claims = [c for c in res["claims_not_supported"]
              if c["claim"] == "the declared uncertainty is a standard error"]
    assert claims and claims[0]["status"] == "not supported"


def test_priors_outside_the_envelope_are_refused_by_name():
    """R17: the documented envelope must actually refuse, and say which prior."""
    p = _product(h=1.0, sigma=0.2, snow=0.2, snow_unc=0.05, conc=100)
    with pytest.raises(NotAssessable) as exc:
        run_audit(p, Priors(rho_s=450.0), BudgetPolicy(), "unused")
    msg = str(exc.value)
    assert "rho_s=450" in msg
    assert "[200, 400]" in msg


def test_empty_tables_use_the_contract_header(tmp_path):
    """R33: one schema per filename, including when nothing was computed."""
    write_empty_tables(str(tmp_path))
    a = pd.read_csv(tmp_path / "alpha.csv")
    b = pd.read_csv(tmp_path / "budget.csv")
    assert list(a.columns) == ALPHA_COLUMNS and len(ALPHA_COLUMNS) == 18
    assert list(b.columns) == BUDGET_COLUMNS and len(BUDGET_COLUMNS) == 14


def test_cli_refuses_sigma_snow_without_source_and_writes_diagnostics(tmp_path):
    """R28: an input error is exit 5 with diagnostics, not a bare exit 2."""
    out = tmp_path / "out"
    code = main(["run", "--thickness", "does-not-exist.nc", "--out", str(out),
                 "--sigma-snow", "0.05"])
    assert code == 5
    assert (out / "gates.json").exists() and (out / "report.md").exists()
    assert "sigma-snow-source" in (out / "report.md").read_text(encoding="utf-8")
    assert list(pd.read_csv(out / "alpha.csv").columns) == ALPHA_COLUMNS
