"""K-gates: the part of the tool that refuses to answer.

Each gate returns a status in {pass, warn, fail, not_assessable}.  The
overall verdict is derived from them.  alpha is a monotone rescale of the
*declared* raw-channel value ``a`` against the fixed prior budget
``K = b**2 + c**2 + d**2`` (see ``budget.py``); it carries no information
beyond the ratio ``a/K`` and is never a headline result.  It is reported
only when every premise gate passes, and it is *withheld* (no alpha field,
a structured reason code) whenever a premise gate fails: when G3 fails the
verdict is ``unidentifiable``; when the nuisance-floor gate G8 fails the
verdict is ``not_assessable`` with reason code ``G8_BELOW_FLOOR``.  A check
that cannot support its number must not emit one.
"""

from __future__ import annotations

import math

import numpy as np

from .budget import BudgetResult, band_of
from .constants import G3_BLOCK_EXCESS_MIN, G3_NU_MAX
from .ncio import Product

# Latitude bands used by the alpha table and by G8.  Keeping the two on one
# definition is the point: a gate that decides on a different partition from
# the one the table reports cannot be checked against the table.
_BANDS = ("60-70", "70-80", "80+")
# A band must carry at least this many admissible cells to set the run status;
# below it the band median is not a stable summary and the run falls back to
# the domain median.
_G8_MIN_CELLS = 50


def g8_status(ratio: float) -> str:
    """Three bands, NaN-safe: a missing operand is never a pass."""
    if ratio is None or not math.isfinite(ratio):
        return "not_assessable"
    if ratio < 0.5:
        return "fail"
    if ratio < 1.0:
        return "warn"
    return "pass"


def g8_reason_text(value, where: str, status: str) -> str:
    """Prose for G8, one sentence per state; shared by the tool and the
    re-derivation of archived runs so the two cannot describe it differently."""
    if status == "not_assessable" or value is None:
        return ("the ratio is non-finite on every admissible cell; a non-finite "
                "operand must never be read as a pass")
    head = f"raw channel / own nuisance floor is {value:.3f} ({where})"
    if status == "pass":
        return head + "; it is consistent with its own nuisance floor"
    if status == "warn":
        return (head + "; it sits below its own nuisance floor but within a factor of two, "
                "so a 1-sigma posterior is not excluded and the margin is thin")
    return (head + "; it lies below half its own nuisance floor, so it cannot be a "
            "1-sigma posterior under the recorded priors")


def compose_verdict(statuses: dict, g3_reason: str, g8_reason: str,
                    g1_reason: str = ""):
    """Map gate statuses to (verdict, verdict_reason, alpha_reason_code).

    Kept as a separate function because the verdict is part of the output
    contract and is also re-composed by ``examples/rederive_g8.py`` when a gate
    is re-evaluated from an archived record; two copies of this precedence
    order would drift.

    alpha_reason_code is a machine-readable tag for *why* no alpha summary is
    emitted; when alpha is reported it is ``ASSESSED``.  A check that could not
    be evaluated is never a pass: both the ``fail`` and the ``not_assessable``
    states of G8 withhold alpha.

    Order of the ``not assessable`` reasons: a product that does not declare
    the quantity under audit is refused on that ground (G1/G2) before the
    statistic that could not be formed (G3) or the nuisance terms that are
    absent (G4).  The missing declaration is both the more fundamental and the
    only actionable objection -- a provider can publish a sigma, and no file
    can repair a grid -- so it is the one the user reads first.
    """
    if statuses["G3"] == "fail":
        return ("unidentifiable",
                "G3 failed: the field carries no resolvable coherent signal; "
                "no alpha is reported (reason code G3_NO_SIGNAL)",
                "G3_NO_SIGNAL")
    if "fail" in (statuses["G1"], statuses["G2"]):
        return ("invalid_input", "input contract/validity gate failed",
                "INPUT_CONTRACT_FAIL")
    if "not_assessable" in (statuses["G1"], statuses["G2"]):
        return ("not_assessable",
                "the product does not declare the quantity the audit tests: "
                + (g1_reason or "no usable uncertainty field")
                + " (alpha withheld: reason code INPUT_NOT_ASSESSABLE)",
                "INPUT_NOT_ASSESSABLE")
    if statuses["G3"] == "not_assessable":
        # A signal-existence verdict could not be issued (no declared sigma, or
        # the grid cannot realise the calibrated block).  The alpha arithmetic
        # may still be computable, but its premise is untested, so it is not
        # published as an assessable result.
        return ("not_assessable", "G3 could not be evaluated: " + g3_reason,
                "G3_NOT_ASSESSABLE")
    if statuses["G4"] == "not_assessable":
        return ("not_assessable", "nuisance inputs missing; alpha undefined",
                "G4_NOT_ASSESSABLE")
    if statuses["G8"] == "fail":
        # Nuisance-floor gate.  When it fails the raw channel is smaller than
        # the snow and density terms it must contain, so it cannot be a 1-sigma
        # posterior under the recorded priors.  alpha is a monotone rescale of
        # that same raw channel against the same prior budget, so it carries no
        # information that survives this failure; the protocol withholds it
        # rather than emitting a number pinned near one.
        return ("not_assessable",
                "G8 failed: " + g8_reason
                + " (alpha withheld: reason code G8_BELOW_FLOOR)",
                "G8_BELOW_FLOOR")
    if statuses["G8"] == "not_assessable":
        # The nuisance-floor guard could not be evaluated (no finite ratio on
        # any admissible cell).  Withholding alpha is the same rule applied one
        # step earlier: the number's premise was never tested, so the number is
        # not published.
        return ("not_assessable",
                "G8 could not be evaluated: " + g8_reason
                + " (alpha withheld: reason code G8_NOT_ASSESSABLE)",
                "G8_NOT_ASSESSABLE")
    return ("assessable",
            "gates passed; alpha reported (a monotone rescale of a) with its "
            "object label and caveats", "ASSESSED")


def _num(x):
    if x is None:
        return None
    if isinstance(x, (int, np.integer)):
        return int(x)
    try:
        v = float(x)
    except (TypeError, ValueError):
        return x
    return None if not math.isfinite(v) else v


def _gate(gid, name, status, value=None, threshold=None, reason="", inputs=None):
    return {
        "id": gid, "name": name, "status": status,
        "value": _num(value), "threshold": threshold,
        "reason": reason, "inputs": inputs or {},
    }


def _sigma_semantics(product: Product) -> tuple[str, str]:
    if product.sigma is None:
        return "absent", "no declared thickness uncertainty variable"
    sn = str(product.sigma.attrs.get("standard_name", "")).lower()
    cm = str(product.sigma.attrs.get("comment", "")).lower()
    ln = str(product.sigma.attrs.get("long_name", "")).lower()
    desc = str(product.sigma.attrs.get("description", "")).lower()
    if "standard_error" in sn:
        if "maximum" in cm:
            return "maximum_estimate", "standard_name=standard_error but comment says maximum"
        return ("declared_1sigma", "standard_name declares a standard error"
                + ("; comment mentions gridding reduction" if "gridding" in cm or "reduction" in cm else ""))
    if "standard deviation" in ln or "standard deviation" in desc or "standard deviation" in cm:
        return ("population_sd",
                "the field is declared as a standard deviation, not a standard error of "
                "the mean; converting it to an error of the mean needs an independent "
                "sample size N_eff, which the product does not declare")
    if "maximum" in cm:
        return "maximum_estimate", "comment declares a maximum uncertainty estimate"
    return "unknown", "no machine-readable standard_name/comment declaring 1-sigma semantics"


def _sigma_as_error_scale(stats: dict) -> tuple[str, float | None, str]:
    """Can the declared sigma act as a per-cell *noise* scale on this field?

    Reads only the two block-variance statistics and S/N.  Their ratio is the
    declared noise share of the field's own variance,

        block_excess_free / block_excess = mean(sigma^2) / Var(h),

    because the block-mean variance cancels.  A genuine per-cell error must
    leave most of the variance to spatial structure, i.e. the share must be
    < 1; at >= 1 the declared sigma accounts for all (or more than all) of the
    observed variability, so it is a regional or upper-bound quantity and the
    sigma-normalised statistics cannot be read as signal-to-noise.

    Returns pass / not_assessable.  Missing inputs are *not* a failure: a
    product without a declared sigma simply cannot answer the question.
    """
    be_sig = stats.get("block_excess_primary")
    be_free = stats.get("block_excess_free_primary")
    snr = stats.get("snr_median")
    if (be_sig is None or be_free is None
            or not np.isfinite(be_sig) or not np.isfinite(be_free) or be_sig <= 0):
        return ("not_assessable", None,
                "the declared sigma or the block statistics are unavailable: whether "
                "sigma can act as a local error scale is not decidable")
    phi = float(be_free / be_sig)
    if snr is None or not np.isfinite(snr):
        return ("not_assessable", phi,
                "median(|h|/sigma) is unavailable: whether sigma is smaller than the "
                "field it describes is not decidable")
    if snr < 1.0:
        return ("not_assessable", phi,
                f"median(|h|/sigma) = {snr:.2f} < 1: the declared sigma is not smaller "
                "than the field it describes")
    if phi >= 1.0:
        return ("not_assessable", phi,
                f"mean(sigma^2)/Var(h) = {phi:.2f} >= 1: the declared sigma accounts for "
                "all of the observed variance and cannot be a per-cell noise scale; the "
                "sigma-normalised block excess and S/N are not interpretable as S/N")
    return ("pass", phi,
            f"mean(sigma^2)/Var(h) = {phi:.2f} < 1 with median(|h|/sigma) = {snr:.2f} >= 1: "
            "sigma is consistent with a per-cell noise scale")


def evaluate_gates(product: Product, budget: BudgetResult | None,
                   stats: dict, ref: dict | None,
                   priors, policy, budget_error_reason: str | None = None) -> dict:
    gates = []

    # G0 -------------------------------------------------------------- contract
    gates.append(_gate("G0", "data contract", "pass",
                       inputs={"thickness_var": product.h.name,
                               "thickness_units": product.h.units,
                               "time_labels": product.time_labels[:6],
                               "grid_shape": list(product.grid.shape)}))

    # G1 -------------------------------------------------- uncertainty validity
    sig = (product.sigma.data * product.sigma.scale
           if product.sigma is not None else None)
    h = product.h.data * product.h.scale
    if sig is None:
        gates.append(_gate("G1", "uncertainty validity", "not_assessable",
                           reason="product publishes no thickness uncertainty"))
    else:
        finite_h = np.isfinite(h) & (h > 0)
        bad = finite_h & (~np.isfinite(sig) | (sig <= 0))
        frac_bad = float(bad.sum() / max(finite_h.sum(), 1))
        status = "pass" if frac_bad <= 0.02 else "fail"
        gates.append(_gate(
            "G1", "uncertainty validity", status,
            value=frac_bad, threshold="<= 0.02 of valid cells may have sigma<=0 or fill",
            reason=("declared sigma is positive and finite on valid cells" if status == "pass"
                    else "a nonzero share of valid cells carries sigma<=0 or a fill value; "
                         "sigma=0 is treated as invalid, never as zero error"),
            inputs={"sigma_var": product.sigma.name, "sigma_units": product.sigma.units}))

    # G2 ---------------------------------------------------------- physical domain
    # The upper bound is the documented physical domain (BudgetPolicy.max_thickness,
    # 6 m; see manuscript Table 1), not a free parameter: it must match the
    # budget mask and the manuscript, or the gate record contradicts both.
    ok = np.isfinite(h) & (h > 0) & (h < policy.max_thickness)
    if sig is not None:
        ok &= np.isfinite(sig) & (sig > 0) & (sig / h <= 1.0)
        frac_bad = float(1.0 - ok.sum() / max((np.isfinite(h) & (h > 0)).sum(), 1))
        status = "pass" if frac_bad <= 0.05 else "fail"
        gates.append(_gate("G2", "physical domain", status, value=frac_bad,
                           threshold=(f"<= 0.05 of valid cells outside "
                                      f"0<h<{policy.max_thickness:g} m or sigma/h>1"),
                           reason="sigma/h<=1 is required for a positive thickness at 1 sigma"))
    else:
        gates.append(_gate("G2", "physical domain", "not_assessable",
                           reason="no declared sigma to test sigma/h"))

    # G3 --------------------------------------------------------- signal existence
    # Two sigma-free quantities only (SPEC.md 5.1): the lag-1 nugget and the
    # sigma-free block excess.  Everything normalised by the declared sigma
    # moved to G5, so a product with no declared uncertainty can be screened
    # here instead of being withheld as not_assessable.
    nu1 = stats.get("nu_lag1")
    be_free = stats.get("block_excess_free_primary")
    be_sig = stats.get("block_excess_primary")   # recorded, not used by G3
    snr = stats.get("snr_median")                # recorded, not used by G3
    block_cells = int(stats.get("block_cells") or 8)
    ext_x = stats.get("block_extent_km_x")
    ext_y = stats.get("block_extent_km_y")
    target_km = float(getattr(policy, "block_km", 200.0))
    # The threshold (1.8, defined on the sigma-free block excess; SPEC.md 4.1)
    # was calibrated on a 200 km block.  If the grid cannot realise that
    # physical scale on both axes, the statistic is not the one that was
    # calibrated and no signal-existence verdict is issued.
    block_ok = bool(
        ext_x is not None and ext_y is not None
        and np.isfinite(ext_x) and np.isfinite(ext_y)
        and 0.75 <= ext_x / target_km <= 1.25
        and 0.75 <= ext_y / target_km <= 1.25)
    label = (f"block_excess_free({block_cells}x{block_cells} cells = "
             f"{ext_x:.0f}x{ext_y:.0f} km) >= {G3_BLOCK_EXCESS_MIN:g}"
             if np.isfinite(ext_x) and np.isfinite(ext_y) else
             f"block_excess_free({block_cells}x{block_cells} cells) >= "
             f"{G3_BLOCK_EXCESS_MIN:g}")
    # bool(...) keeps numpy.bool_ out of the JSON payload.
    cond_nu = bool(nu1 is not None and np.isfinite(nu1) and nu1 <= G3_NU_MAX)
    cond_be = bool(block_ok and be_free is not None and np.isfinite(be_free)
                   and be_free >= G3_BLOCK_EXCESS_MIN)
    if nu1 is None or not np.isfinite(nu1):
        status = "not_assessable"
        reason = "no usable adjacent pairs: the field's structure cannot be measured"
    elif not block_ok:
        status = "not_assessable"
        reason = (f"the grid cannot realise the {target_km:g} km block on which the "
                  f"block-excess threshold was calibrated (best block "
                  f"{ext_x:.0f} x {ext_y:.0f} km with {block_cells} cells); "
                  f"the constant does not transfer to another block scale")
    elif be_free is None or not np.isfinite(be_free):
        status = "not_assessable"
        reason = ("the block statistic cannot be measured on this mask (fewer than "
                  "20 complete blocks, or too few retained cells); an unmeasurable "
                  "statistic is not evidence of white noise")
    elif cond_nu and cond_be:
        status, reason = "pass", ("resolvable, spatially coherent thickness signal; "
                                  "G3 is a coarse screen, so passing means the field is "
                                  "not structurally dead, not that it is identifiable")
    else:
        status, reason = "fail", ("the field is indistinguishable from white noise at "
                                  "the 200 km block; alpha would be a number without "
                                  "a referent")
    gates.append(_gate(
        "G3", "signal existence", status,
        value=nu1,
        threshold={"nu_lag1 <= 0.8": cond_nu, label: cond_be},
        reason=reason,
        inputs={"nu_lag1": _num(nu1), "nu_lag8": _num(stats.get("nu_lag8")),
                "block_excess_free_primary": _num(be_free),
                "block_excess_primary_sigma_normalised": _num(be_sig),
                "snr_median_recorded_only": _num(snr),
                "block_cells": block_cells,
                "block_extent_km_x": _num(ext_x), "block_extent_km_y": _num(ext_y),
                "block_km_target": target_km, "block_scale_realised": block_ok}))

    # G4 ------------------------------------------------------- nuisance inputs
    if budget is None:
        gates.append(_gate(
            "G4", "nuisance inputs", "not_assessable",
            reason=budget_error_reason or "nuisance budget could not be built"))
    else:
        gates.append(_gate(
            "G4", "nuisance inputs", "pass",
            inputs={"snow_var": (product.snow.name if product.snow is not None else None),
                    "snow_sigma_source": budget.sigma_snow_source,
                    "raw_channel": budget.raw_channel_used,
                    "alpha_object": budget.alpha_object}))

    # G5 ------------------------------------ sigma semantics + error-scale check
    # G5 is now the single outlet for everything that depends on sigma.  Its
    # status still reports what the product's attributes *declare*; the
    # sub-check reports whether the declared sigma *behaves* like a per-cell
    # noise scale on this field.  A sub-check failure is "not assessable", not
    # "invalid input": the product is not wrong, the quantity is unreadable.
    sem, sem_reason = _sigma_semantics(product)
    status = ("pass" if sem == "declared_1sigma"
              else "warn" if sem in ("maximum_estimate", "population_sd")
              else "not_assessable")
    sub_status, phi, sub_reason = _sigma_as_error_scale(stats)
    gates.append(_gate(
        "G5", "declared sigma semantics", status, value=sem, reason=sem_reason,
        threshold="declares a standard error (status) and mean(sigma^2)/Var(h) < 1 "
                  "(sigma_as_error_scale sub-check)",
        inputs={"sigma_as_error_scale": sub_status,
                "noise_share_mean_sigma2_over_var_h": _num(phi),
                "sigma_as_error_scale_reason": sub_reason,
                "block_excess_primary_sigma_normalised": _num(stats.get("block_excess_primary")),
                "block_excess_free_primary": _num(stats.get("block_excess_free_primary")),
                "snr_median": _num(stats.get("snr_median"))}))

    # G6 -------------------------------------------- grid-scale resolvability
    q = stats.get("q_sigma")
    if q is None or not np.isfinite(q):
        gates.append(_gate("G6", "grid-scale resolvability", "not_assessable",
                           reason="declared sigma or valid pairs unavailable"))
    else:
        status = "pass" if q >= 1.0 else "warn"
        gates.append(_gate(
            "G6", "grid-scale resolvability", status, value=q,
            threshold="q_sigma >= 1 (declared sigma not larger than the field's own grid texture)",
            reason=("declared sigma is at or below the grid-scale texture" if status == "pass"
                    else "declared sigma exceeds the field's own grid-scale texture; it "
                         "behaves like a regional/upper-bound quantity and cannot rank cells"),
            inputs={"q_sigma": _num(q)}))

    # G7 -------------------------------------------------------------- reference
    if ref is None:
        gates.append(_gate("G7", "external reference", "not_assessable",
                           reason="no --reference supplied; no absolute calibration possible"))
    elif ref.get("status") != "assessed":
        gates.append(_gate("G7", "external reference", "not_assessable",
                           reason=ref.get("reason", "reference unusable")))
    else:
        gates.append(_gate("G7", "external reference", "pass", value=ref["rms"],
                           inputs={"n_pairs": ref["n_pairs"], "bias": _num(ref["bias"]),
                                   "rms": _num(ref["rms"]), "f_irr": _num(ref["f_irr"]),
                                   "nu_lag1": _num(ref["nu_lag1"])},
                           reason="difference-field statistics computed on the common grid"))

    # G8 ------------------------------------------------- raw channel vs its floor
    # G8 must guard the SAME quantity that alpha uses.  alpha is K/(a^2+K) with
    # K = b^2+c^2+d^2, so the admissible test is a / sqrt(K): how the channel's
    # own raw sigma compares with its own nuisance floor.  Testing the declared
    # *thickness* sigma here was wrong whenever the raw channel is the freeboard
    # one, because a = kappa*sigma_freeboard there and the thickness sigma never
    # enters alpha.  That defect is corrected in r13; it let the freeboard
    # channel through at 0.884, which is the posterior channel's ratio.
    #
    # K is not a constant of the priors: c and d grow with h and h_s, so K
    # varies from cell to cell by a factor of two or more across the domain.
    # A single domain-median ratio can therefore hide a latitude band in which
    # the raw channel is an order of magnitude below its own floor, so the gate
    # is evaluated band by band and the run takes the *worst* band.  The bands
    # are the ones the alpha table reports on, so the gate and the table cannot
    # disagree about which part of the domain drove the refusal.
    if budget is not None and budget.mask.any():
        a_raw = np.asarray(budget.comps["a"], dtype=float)
        floor = np.asarray(budget.sigma_floor, dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio_arr = a_raw / floor
        usable = budget.mask & np.isfinite(ratio_arr) & np.isfinite(floor) & (floor > 0)
        band = band_of(product.grid.lat) if product is not None else None
        per_band = {}
        if band is not None:
            for name in _BANDS:
                sel = usable & (band == name)
                if int(sel.sum()) >= _G8_MIN_CELLS:
                    per_band[name] = float(np.nanmedian(ratio_arr[sel]))
        if not per_band:
            # No band is populated (small fixture, or a grid that misses the
            # bands): fall back to the domain median over admissible cells.
            if usable.any():
                per_band = {"all": float(np.nanmedian(ratio_arr[usable]))}
        if per_band:
            decided = min(per_band, key=lambda k: per_band[k])
            ratio = per_band[decided]
            status = g8_status(ratio)
            if len(per_band) == 1:
                whose = "domain" if decided == "all" else f"{decided}N band"
            else:
                whose = f"{decided}N band, worst of {len(per_band)}"
            reason = g8_reason_text(ratio, whose, status)
        else:
            ratio = None
            status = "not_assessable"
            reason = g8_reason_text(None, "domain", status)
        gates.append(_gate(
            "G8", "raw channel vs its own nuisance floor", status, value=ratio,
            threshold="median(a / sqrt(b^2+c^2+d^2)) per 10-degree band: "
                      ">= 1.0 pass, [0.5,1.0) warn, < 0.5 fail; run status = worst band",
            reason=reason,
            inputs={"ratio_by_band": {k: _num(v) for k, v in per_band.items()},
                    "deciding_band": decided if per_band else None,
                    "band_margin_cells": _G8_MIN_CELLS,
                    "sigma_floor_median": _num(float(np.nanmedian(floor[budget.mask])))
                    if budget.mask.any() else None}))
    else:
        gates.append(_gate("G8", "raw channel vs its own nuisance floor", "not_assessable",
                           reason="no admissible budget: the raw channel or the nuisance "
                                  "terms are unavailable"))

    # ---------------------------------------------------------------- verdict
    statuses = {g["id"]: g["status"] for g in gates}
    g3_reason = next(g["reason"] for g in gates if g["id"] == "G3")
    g8_reason = next(g["reason"] for g in gates if g["id"] == "G8")
    verdict, verdict_reason, alpha_reason_code = compose_verdict(
        statuses, g3_reason, g8_reason,
        next(g["reason"] for g in gates if g["id"] == "G1"))

    claims = _claims(gates, budget, stats, ref, verdict)
    return {"gates": gates, "verdict": verdict, "verdict_reason": verdict_reason,
            "alpha_reason_code": alpha_reason_code,
            "claims_not_supported": claims}


def _claims(gates, budget, stats, ref, verdict) -> list[dict]:
    g = {x["id"]: x for x in gates}
    out = []
    if verdict == "unidentifiable":
        failed = [k for k, ok in g["G3"]["threshold"].items() if not ok]
        out.append({"claim": "the product resolves Arctic sea-ice thickness",
                    "status": "not supported",
                    "why": "the field is statistically indistinguishable from white noise "
                           f"({'; '.join(failed)} failed)"})
    if g["G8"]["status"] in ("fail", "warn"):
        out.append({"claim": "the declared uncertainty is a 1-sigma posterior under the "
                             "recorded nuisance priors",
                    "status": "not supported",
                    "why": f"median(a/sqrt(K))={g['G8']['value']:.3f}" if g['G8']['value'] is not None else "median(a/sqrt(K)) undefined"})
    if g["G6"]["status"] == "warn":
        out.append({"claim": "declared sigma can rank or weight individual cells",
                    "status": "not supported",
                    "why": f"q_sigma={g['G6']['value']:.3f} < 1: the declared sigma exceeds the "
                           "field's own 25 km texture"})
    if g["G5"]["status"] in ("warn", "not_assessable"):
        warn = g["G5"]["status"] == "warn"
        out.append({"claim": "the declared uncertainty is a standard error",
                    "status": ("not supported" if warn else "not assessable"),
                    "why": g["G5"]["reason"]})
    if g["G5"]["inputs"].get("sigma_as_error_scale") == "not_assessable":
        out.append({"claim": "the declared uncertainty can be read as a per-cell noise "
                             "scale (sigma-normalised block excess, median S/N)",
                    "status": "not assessable",
                    "why": g["G5"]["inputs"].get("sigma_as_error_scale_reason", "")})
    if budget is not None and budget.mask.any():
        floor = float(np.nanmedian(budget.sigma_floor[budget.mask]))
        out.append({"claim": f"sigma can be reduced below the nuisance floor ({floor:.2f} m) "
                             "by adding independent observations",
                    "status": "not supported",
                    "why": f"the nuisance floor sqrt(b^2+c^2+d^2) is {floor:.2f} m and is "
                           "invariant to the number of independent repeats (shared nuisance)"})
    if ref is None or ref.get("status") != "assessed":
        out.append({"claim": "the product is calibrated in absolute terms",
                    "status": "not assessable",
                    "why": "no independent reference on the same grid"})
    elif np.isfinite(ref.get("f_irr", float("nan"))):
        out.append({"claim": "product-reference disagreement averages away at 25 km",
                    "status": ("not supported" if ref["f_irr"] > 0.5 else "supported"),
                    "why": f"f_irr={ref['f_irr']:.2f} of the difference variance is spatially "
                           "coherent (lag-1)"})
    return out
