"""Physical constants and priors, each with an explicit provenance string.

Nothing in this module is estimated from the data.  Every number here is an
*assumption*; the tool records the full dict in gates.json under
``priors`` so that any number in the outputs can be traced back to it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field


@dataclass(frozen=True)
class Priors:
    """Nuisance priors for the hydrostatic freeboard -> thickness chain."""

    rho_w: float = 1027.0
    rho_i: float = 910.0
    rho_s: float = 320.0
    sig_rho_i: float = 30.0
    sig_rho_s: float = 60.0
    # dry-snow refractive index: central value sqrt(1.9)=1.378, prior uniform
    # over [n0-hw, n0+hw] with hw=0.5 -> sd = 0.5/sqrt(3) = 0.2887.
    # This is the coefficient used by the predecessor analysis
    # (0.2887*kappa*(n-1)*h_s).  Adopting the published [1.6, 2.2] interval
    # instead gives hw=0.3 and sd=0.1732; pass --eps-halfwidth 0.3 to switch.
    refraction_n0: float = math.sqrt(1.9)
    eps_halfwidth: float = 0.5

    # provenance, one line per parameter, rendered into report/gates
    provenance: dict = field(default_factory=lambda: {
        "rho_w": "seawater density, S=34.7, T=-1.8 C (EOS-80)",
        "rho_i": "bulk sea-ice density 910 kg m-3 (Timco & Frederking 1996)",
        "rho_s": "snow density on sea ice 320 kg m-3 (Warren et al. 1999)",
        "sig_rho_i": "bulk ice-density spread 30 kg m-3 (Cold Regions Sci. Tech. 2016)",
        "sig_rho_s": "dry-snow density spread 60 kg m-3 (Warren et al. 1999)",
        "refraction_n0": "dry-snow refractive index sqrt(1.9) (Metzler 1984; Willatt 2011)",
        "eps_halfwidth": "uniform prior half-width on the refractive index, sd=hw/sqrt(3)",
    })

    # ---- derived ---------------------------------------------------------
    @property
    def kappa(self) -> float:
        """d(h_i)/d(f_i) under hydrostatic equilibrium."""
        return self.rho_w / (self.rho_w - self.rho_i)

    @property
    def deff(self) -> float:
        """rho_w - rho_i, the density lever arm."""
        return self.rho_w - self.rho_i

    @property
    def snow_sensitivity(self) -> float:
        """d(h_i)/d(h_s): kappa*(n-1) + rho_s/(rho_w-rho_i)."""
        return self.kappa * (self.refraction_n0 - 1.0) + self.rho_s / self.deff

    @property
    def eps_sd(self) -> float:
        """sd of the uniform refractive-index prior."""
        return self.eps_halfwidth / math.sqrt(3.0)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update({
            "kappa": self.kappa,
            "snow_sensitivity": self.snow_sensitivity,
            "eps_sd": self.eps_sd,
            "deff": self.deff,
        })
        return d


# Documented sweep envelopes (SPEC.md section 9).  A prior value outside
# these ranges cannot be checked against the literature, so tool runs
# using it are reported as `not_assessable` rather than given a number.
PRIOR_RANGES = {
    "rho_i": (880.0, 940.0),        # bulk ice density, kg m-3
    "rho_s": (200.0, 400.0),        # snow density on sea ice, kg m-3
    "refraction_n0": (1.30, 1.90),  # dry-snow refractive index
    "eps_halfwidth": (0.20, 0.50),  # uniform-prior half-width on n0
}

# ---------------------------------------------------------------- G3 constants
# Derived in examples/gate_calibration.py; see out/gatesweep/gate_calibration.md.
#   nu    -- admissible window 0.70 (worst field that must pass) to 0.81 (best
#            white field); 0.8 is inside it.
#   block -- THE CONSTANT MUST LIVE ON THE STATISTIC G3 ACTUALLY TESTS.
#            The calibration fields have Var(h) = amp^2 and sigma = amp*sqrt(w)
#            (w = white-noise fraction), hence
#                be_free = be8 * w          [be8 = Var(block mean)/mean(sigma^2)]
#            On be8 the admissible window was (1.33, 3.83]; on be_free it is
#            (1.33, 1.92].  The old value 3.0 belongs to the *sigma-normalised*
#            scale and is NOT comparable with a be_free reading: the weakest
#            field that had to pass reads 1.92 there, so 3.0 would reject it.
#            1.8 sits inside the window and above the white-field value 1.33.
#            Conversion for anyone reconciling old tables: be_free = be8 * w.
#   snr   -- degenerate: a scale guard equivalent to sigma <= median|h| (G2),
#            no longer part of G3.
G3_NU_MAX = 0.8
G3_BLOCK_EXCESS_MIN = 1.8
G3_SNR_MIN = 1.0


def outside_envelope(priors: Priors) -> list[str]:
    """Names of priors that fall outside the documented sweep envelopes."""
    out = []
    for name, (lo, hi) in PRIOR_RANGES.items():
        v = getattr(priors, name)
        if not (lo <= v <= hi):
            out.append(name)
    return out
