"""Per-cell variance budget and the projector coefficient alpha.

Objects produced here (all definitions in SPEC.md):

  a    raw channel      : the product's declared observation-level sigma
  b    snow depth       : snow_sensitivity * sigma_hs
  c    density          : ice + snow density propagation
  d    penetration      : refraction prior propagated through snow depth

  sigma_post^2 = a^2 + b^2 + c^2 + d^2
  alpha        = 1 - a^2 / sigma_post^2

IMPORTANT: alpha is NOT an independent result.  With K = b^2 + c^2 + d^2 the
prior budget (fixed by the authors' priors, not by the data),

        alpha = K / (a^2 + K) = 1 / (1 + (a/K)),

a strictly monotone *decreasing* function of the single declared number a.
It therefore carries exactly the information of the ratio a/K and nothing
else: raising the declared uncertainty lowers alpha, and no data, no
degrees of freedom and no estimation enter.  The tool ships alpha only as a
footnote to the gates -- with (i) the object label (`alpha_object`), (ii) the
premise gates, and (iii) the measured structure diagnostics -- and it is
withheld entirely (no field, a reason code) when a premise gate fails (G3,
G4 or the nuisance-floor gate G8).  The two object labels are NOT
interchangeable:

  nuisance_share_of_freeboard_channel
      a = kappa * sigma_freeboard.  alpha is the fraction of the freeboard
      information eaten by nuisance (the object used in the predecessor
      analysis; comparable across products that all publish a freeboard
      uncertainty).

  nuisance_share_of_declared_posterior
      a = sigma_thickness (the product's own declared uncertainty).  alpha
      is the share of the declared posterior variance attributable to
      nuisance priors.  It mixes pre- and post-inversion objects and must
      never be compared with the freeboard object.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constants import Priors
from .ncio import NotAssessable, Product


@dataclass
class BudgetPolicy:
    min_concentration: float = 15.0
    min_thickness: float = 0.05
    max_thickness: float = 6.0
    raw_channel: str = "auto"          # auto | freeboard | posterior
    require_snow: bool = True
    sigma_snow_override: float | None = None
    sigma_snow_source: str | None = None
    # Target physical size of the block-variance test (G3).  The calibrated
    # threshold (1.8 on the sigma-free block excess) is defined at 200 km;
    # other values are user choices whose threshold is no longer calibrated
    # and are recorded as such.
    block_km: float = 200.0


@dataclass
class BudgetResult:
    mask: np.ndarray
    alpha: np.ndarray
    sigma_raw: np.ndarray
    sigma_post: np.ndarray
    sigma_floor: np.ndarray
    comps: dict            # a, b, c, d in metres
    shares: dict           # squared shares, sum to 1
    alpha_object: str
    raw_channel_used: str
    sigma_snow: np.ndarray
    sigma_snow_source: str
    flags: dict = field(default_factory=dict)


def resolve_raw_channel(product: Product, policy: BudgetPolicy,
                        priors: Priors) -> tuple[str, np.ndarray]:
    """Return (object_label, raw sigma in metres, (T,Y,X))."""
    has_fb = product.freeboard is not None and product.freeboard_unc is not None
    if policy.raw_channel == "freeboard" and not has_fb:
        raise NotAssessable(
            "--raw-channel freeboard requires both freeboard and freeboard_uncertainty", "G4")
    if policy.raw_channel == "posterior" and product.sigma is None:
        raise NotAssessable(
            "--raw-channel posterior requires the product's declared thickness uncertainty", "G4")
    if policy.raw_channel in ("auto", "freeboard") and has_fb:
        # a = kappa * sigma_freeboard: the thickness uncertainty implied by the
        # declared freeboard uncertainty before any nuisance term is added.
        return ("nuisance_share_of_freeboard_channel",
                priors.kappa * product.freeboard_unc.data * product.freeboard_unc.scale)
    if product.sigma is None:
        raise NotAssessable(
            "no declared thickness uncertainty and no freeboard uncertainty: "
            "the raw channel is undefined", "G4")
    return ("nuisance_share_of_declared_posterior",
            product.sigma.data * product.sigma.scale)


def _snow_sigma(product: Product, policy: BudgetPolicy) -> tuple[np.ndarray, str]:
    if policy.sigma_snow_override is not None:
        src = policy.sigma_snow_source or "user_override(no source declared)"
        return (np.full(product.h.data.shape, float(policy.sigma_snow_override)),
                f"user_override: {src}")
    if product.snow_unc is not None:
        return (product.snow_unc.data * product.snow_unc.scale,
                f"product:{product.snow_unc.name}")
    if policy.require_snow:
        raise NotAssessable(
            "snow depth uncertainty is absent and no --sigma-snow override given; "
            "refusing to substitute a default", "G4")
    raise NotAssessable("snow depth uncertainty is required for the budget", "G4")


def compute_budget(product: Product, priors: Priors,
                   policy: BudgetPolicy) -> BudgetResult:
    h = product.h.data * product.h.scale
    if product.snow is None:
        if policy.require_snow:
            raise NotAssessable(
                "snow depth is absent; the nuisance budget is undefined "
                "(pass --snow or --sigma-snow + --sigma-snow-source)", "G4")
        raise NotAssessable("snow depth is required for the budget", "G4")
    hs = product.snow.data * product.snow.scale
    sig_hs, sig_hs_src = _snow_sigma(product, policy)
    alpha_object, a = resolve_raw_channel(product, policy, priors)

    conc = (product.concentration.data
            if product.concentration is not None else np.full_like(h, np.nan))
    with np.errstate(invalid="ignore"):
        mask = (np.isfinite(h) & np.isfinite(a) & np.isfinite(hs) & np.isfinite(sig_hs)
                & (a > 0) & (sig_hs >= 0)
                & (h >= policy.min_thickness) & (h <= policy.max_thickness))
        if product.concentration is not None:
            mask &= np.isfinite(conc) & (conc > policy.min_concentration)

    b = priors.snow_sensitivity * sig_hs
    c = np.sqrt((h / priors.deff * priors.sig_rho_i) ** 2
                + (hs / priors.deff * priors.sig_rho_s) ** 2)
    d = priors.eps_sd * priors.kappa * (priors.refraction_n0 - 1.0) * hs

    a2, b2, c2, d2 = a ** 2, b ** 2, c ** 2, d ** 2
    post2 = a2 + b2 + c2 + d2
    # alpha = 1 - a^2/(a^2+K) with K = b^2+c^2+d^2: a monotone decreasing
    # rescale of the declared raw channel a against the fixed prior budget K.
    # It is reported only as a footnote to the gates (see gates.evaluate_gates).
    with np.errstate(invalid="ignore", divide="ignore"):
        alpha = np.where(mask, 1.0 - a2 / post2, np.nan)
        sigma_post = np.where(mask, np.sqrt(post2), np.nan)
        sigma_raw = np.where(mask, a, np.nan)
        sigma_floor = np.where(mask, np.sqrt(b2 + c2 + d2), np.nan)
        shares = {k: np.where(mask, v / post2, np.nan)
                  for k, v in zip(("a", "b", "c", "d"), (a2, b2, c2, d2))}

    flags = {
        "n_valid": int(mask.sum()),
        "n_total": int(mask.size),
        "frac_in_budget": float(mask.mean()),
        "h_missing_frac": float((~np.isfinite(h)).mean()),
        "raw_missing_frac": float((~np.isfinite(a)).mean()),
        "raw_zero_frac_of_valid_h": float(
            np.mean((~np.isfinite(a) | (a <= 0))[np.isfinite(h)]) if np.isfinite(h).any() else np.nan),
        "snow_missing_frac": float((~np.isfinite(hs)).mean()),
        "sigma_snow_source": sig_hs_src,
    }
    return BudgetResult(mask=mask, alpha=alpha, sigma_raw=sigma_raw,
                        sigma_post=sigma_post, sigma_floor=sigma_floor,
                        comps={"a": a, "b": b, "c": c, "d": d},
                        shares=shares, alpha_object=alpha_object,
                        raw_channel_used=("freeboard" if alpha_object.endswith("freeboard_channel")
                                          else "declared_posterior"),
                        sigma_snow=sig_hs, sigma_snow_source=sig_hs_src,
                        flags=flags)


def band_of(lat: np.ndarray) -> np.ndarray:
    """60-70 / 70-80 / 80+ latitude bands; NaN outside."""
    lat = np.asarray(lat, dtype="float64")
    out = np.full(lat.shape, "", dtype=object)
    out[np.isfinite(lat) & (lat >= 60) & (lat < 70)] = "60-70"
    out[np.isfinite(lat) & (lat >= 70) & (lat < 80)] = "70-80"
    out[np.isfinite(lat) & (lat >= 80)] = "80+"
    out[~np.isfinite(lat) | (lat < 60)] = ""
    return out
