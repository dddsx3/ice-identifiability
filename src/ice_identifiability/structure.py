"""Data-derived structure diagnostics.

These quantities never use the declared uncertainty to *produce* a number;
they measure the field itself.  They are the empirical layer that separates
"a product with resolvable, spatially coherent thickness" from a white-noise
field that would otherwise receive a flattering alpha.

Definitions (SPEC.md section 4):

  nu(lag) = 0.5 * E[(h(x+lag) - h(x))^2] / Var(h)
      = 1 for white noise at every lag; -> 0 for a smooth field at short lag.
  block_excess(B) = Var(block mean at B x B) / [ mean(sigma^2) / n_cells ]
      = 1 if the field were exactly the declared white noise; >> 1 when a
      coherent signal exists at that scale.  Depends on the declared sigma.
  block_excess_free(B) = Var(block mean) * n_cells / Var(h)
      = the same statistic with the declared sigma removed.  = 1 for a white
      field at any n_cells; grows with the coherent variance at the block
      scale.  This is the quantity G3 uses; see SPEC.md section 4.
  snr_median = median(|h| / sigma_declared)
  q_sigma = 0.5 * E[(h(x+1)-h(x))^2] / mean(sigma_declared^2)
      < 1 means the declared sigma exceeds the field's own grid-scale texture,
      i.e. the declared sigma cannot be a *local* standard error.
"""

from __future__ import annotations

import numpy as np

_MIN_PAIRS = 1000


def _pairs(h: np.ndarray, mask: np.ndarray, lag: int, axis: int):
    """Valid adjacent pairs along one axis at a given lag."""
    a = np.take(h, range(0, h.shape[axis] - lag), axis=axis)
    b = np.take(h, range(lag, h.shape[axis]), axis=axis)
    ma = np.take(mask, range(0, mask.shape[axis] - lag), axis=axis)
    mb = np.take(mask, range(lag, mask.shape[axis]), axis=axis)
    ok = ma & mb & np.isfinite(a) & np.isfinite(b)
    return a[ok], b[ok]


def nugget_ratio(h: np.ndarray, mask: np.ndarray, lag: int = 1) -> float:
    """0.5*E[d^2]/Var(h), pooled over both horizontal axes. NaN if too few pairs."""
    hv = h[mask]
    if hv.size < _MIN_PAIRS:
        return float("nan")
    var = float(np.var(hv))
    if var <= 0:
        return float("nan")
    sq = []
    for axis in (1, 2):
        a, b = _pairs(h, mask, lag, axis)
        if a.size:
            sq.append((a - b) ** 2)
    if not sq:
        return float("nan")
    d2 = np.concatenate(sq)
    if d2.size < _MIN_PAIRS:
        return float("nan")
    # trim the top 0.1% to keep isolated spikes from dominating
    hi = np.percentile(d2, 99.9)
    d2 = d2[d2 <= hi]
    return float(0.5 * d2.mean() / var)


def q_sigma(h: np.ndarray, mask: np.ndarray, sigma: np.ndarray | None) -> float:
    """Grid-scale white level implied by the field divided by declared sigma^2."""
    if sigma is None:
        return float("nan")
    ok = mask & np.isfinite(sigma) & (sigma > 0)
    if ok.sum() < _MIN_PAIRS:
        return float("nan")
    d2 = []
    for axis in (1, 2):
        a, b = _pairs(h, ok, 1, axis)
        if a.size:
            d2.append((a - b) ** 2)
    if not d2:
        return float("nan")
    white = 0.5 * float(np.concatenate(d2).mean())
    return float(white / float(np.mean(sigma[ok] ** 2)))


def _block_partition(h: np.ndarray, mask: np.ndarray, block: int):
    """Split the last two axes into B x B blocks and return their valid means.

    Shared by both block-excess statistics so that they are computed on the
    same blocks and their ratio is the declared noise share
    ``mean(sigma^2)/Var(h)`` (up to the 10 % cell-count tolerance).
    """
    T, Y, X = h.shape
    ny, nx = Y // block, X // block
    if ny < 2 or nx < 2:
        return None
    hh = h[:, :ny * block, :nx * block].reshape(T, ny, block, nx, block)
    mm = mask[:, :ny * block, :nx * block].reshape(T, ny, block, nx, block)
    cnt = mm.sum(axis=(2, 4)).astype("float64")
    sum_h = np.where(mm, hh, 0.0).sum(axis=(2, 4))
    mean_h = np.where(cnt > 0, sum_h / np.maximum(cnt, 1), np.nan)
    good = (cnt >= 0.9 * block * block) & np.isfinite(mean_h)
    return hh, mm, cnt, mean_h, good


def block_excess(h: np.ndarray, mask: np.ndarray, sigma: np.ndarray | None,
                 block: int = 8) -> float:
    """Var(block mean) / white-noise prediction, for one block size."""
    if sigma is None:
        return float("nan")
    part = _block_partition(h, mask, block)
    if part is None:
        return float("nan")
    _, mm, cnt, mean_h, base_good = part
    T, Y, X = h.shape
    ny, nx = Y // block, X // block
    ss = np.where(np.isfinite(sigma), sigma, np.nan)[:, :ny * block, :nx * block]
    ss = ss.reshape(T, ny, block, nx, block)
    var_sig = np.where(mm & np.isfinite(ss), ss ** 2, np.nan)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_sig2 = np.nanmean(var_sig, axis=(2, 4))

    good = base_good & np.isfinite(mean_sig2)
    if good.sum() < 20:
        return float("nan")
    vals = mean_h[good]
    pred = mean_sig2[good] / cnt[good]
    obs_var = float(np.var(vals))
    pred_var = float(np.mean(pred))
    if pred_var <= 0:
        return float("nan")
    return obs_var / pred_var


def block_excess_free(h: np.ndarray, mask: np.ndarray, block: int = 8) -> float:
    """Var(block mean) * n_cells / Var(h): the block excess with sigma removed.

    Dimensionless (m^2 / m^2).  The null is 1 at every block size and on every
    grid: a white field's block mean has variance Var(h)/n_cells.  It needs no
    declared sigma, which is why G3 can be a sigma-free coarse screen.

    Construction and limits are specified in SPEC.md section 4.1: Var(block
    mean) is a population variance over retained blocks (>= 90 % valid cells,
    at least 20 blocks), n_cells is the mean valid-cell count per retained
    block, and Var(h) uses the masked cells *inside those same blocks*.  The
    ratio therefore cannot exceed about n_cells, and a field that is smooth
    within blocks but differs between them saturates it.  That saturation, not
    the null, is what changes with grid spacing.
    """
    part = _block_partition(h, mask, block)
    if part is None:
        return float("nan")
    hh, mm, cnt, mean_h, good = part
    if good.sum() < 20:
        return float("nan")
    keep = good[:, :, None, :, None] & mm
    hv = hh[keep]
    if hv.size < _MIN_PAIRS:
        return float("nan")
    var_h = float(np.var(hv))
    if var_h <= 0:
        return float("nan")
    n_eff = float(cnt[good].mean())
    return float(np.var(mean_h[good]) * n_eff / var_h)


def snr_median(h, mask, sigma) -> float:
    if sigma is None:
        return float("nan")
    ok = mask & np.isfinite(sigma) & (sigma > 0)
    if ok.sum() < _MIN_PAIRS:
        return float("nan")
    return float(np.median(np.abs(h[ok]) / sigma[ok]))


def cell_extent_km(lat, lon, mask=None) -> tuple[float, float]:
    """Median great-circle cell size in km along x (columns) and y (rows).

    The block test is defined in kilometres; a fixed number of cells is only
    meaningful while the cell size is constant.  These medians convert a
    kilometre target into a number of cells for the actual grid.
    """
    lat = np.asarray(lat, dtype="float64")
    lon = np.asarray(lon, dtype="float64")
    if lat.ndim != 2 or lon.shape != lat.shape:
        return float("nan"), float("nan")

    def _gc(la1, lo1, la2, lo2):
        p1, p2 = np.radians(la1), np.radians(la2)
        dp = p2 - p1
        dl = np.radians(lo2 - lo1)
        a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
        return 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))

    dx = _gc(lat[:, :-1], lon[:, :-1], lat[:, 1:], lon[:, 1:])
    dy = _gc(lat[:-1, :], lon[:-1, :], lat[1:, :], lon[1:, :])
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.ndim == 3:
            m = m[0]
        if m.shape == lat.shape:
            dx = np.where(m[:, :-1] & m[:, 1:], dx, np.nan)
            dy = np.where(m[:-1, :] & m[1:, :], dy, np.nan)
    with np.errstate(invalid="ignore"):
        return (float(np.nanmedian(dx)) if np.isfinite(dx).any() else float("nan"),
                float(np.nanmedian(dy)) if np.isfinite(dy).any() else float("nan"))


def block_cells_for(lat, lon, mask=None, block_km: float = 200.0):
    """Cells needed to realise ``block_km`` on this grid, plus cell extents in km.

    The G3 constant was calibrated on a 200 km block.  Keeping that scale in
    kilometres makes the test transferable: on EASE2 25 km it is the original
    8 x 8 cells (196 x 200 km), on a coarser grid it is fewer cells.
    """
    dx, dy = cell_extent_km(lat, lon, mask)
    if np.isfinite(dx) and np.isfinite(dy):
        cell_km = float(np.sqrt(dx * dy))
        n = max(2, int(round(float(block_km) / cell_km))) if cell_km > 0 else 8
    else:
        n = 8
    return n, dx, dy


def structure_report(h, mask, sigma_declared, block_cells: int = 8,
                     lat=None, lon=None) -> dict:
    stats = {
        "nu_lag1": nugget_ratio(h, mask, 1),
        "nu_lag2": nugget_ratio(h, mask, 2),
        "nu_lag4": nugget_ratio(h, mask, 4),
        "nu_lag8": nugget_ratio(h, mask, 8),
        "block_excess_4": block_excess(h, mask, sigma_declared, 4),
        "block_excess_8": block_excess(h, mask, sigma_declared, 8),
        "block_excess_16": block_excess(h, mask, sigma_declared, 16),
        "block_excess_primary": block_excess(h, mask, sigma_declared, block_cells),
        # sigma-free twin of the four statistics above; defined without sigma
        "block_excess_free_4": block_excess_free(h, mask, 4),
        "block_excess_free_8": block_excess_free(h, mask, 8),
        "block_excess_free_16": block_excess_free(h, mask, 16),
        "block_excess_free_primary": block_excess_free(h, mask, block_cells),
        "block_cells": int(block_cells),
        "q_sigma": q_sigma(h, mask, sigma_declared),
        "snr_median": snr_median(h, mask, sigma_declared),
        "n_valid": int(mask.sum()),
    }
    if lat is not None and lon is not None:
        dx, dy = cell_extent_km(lat, lon, mask)
        stats["cell_km_x"] = dx
        stats["cell_km_y"] = dy
        stats["block_extent_km_x"] = (dx * block_cells
                                      if np.isfinite(dx) else float("nan"))
        stats["block_extent_km_y"] = (dy * block_cells
                                      if np.isfinite(dy) else float("nan"))
    return stats


def reference_report(h, ref, mask_p, mask_r) -> dict:
    """Difference-field audit: bias, RMS, and the irreducible (coherent) share.

    f_irr = 1 - nu_lag1(d): the fraction of the *difference* variance that is
    spatially coherent at the 25 km grid and therefore cannot be removed by
    cell averaging.  On a white-noise difference field f_irr -> 0.
    """
    ok = mask_p & mask_r & np.isfinite(h) & np.isfinite(ref)
    if ok.sum() < _MIN_PAIRS:
        return {"status": "not_assessable",
                "reason": "too few common valid cells", "n_pairs": int(ok.sum())}
    d = (h - ref)[ok]
    var = float(np.var(d))
    nu = nugget_ratio(h - ref, ok, 1) if var > 0 else float("nan")
    return {
        "status": "assessed",
        "n_pairs": int(ok.sum()),
        "bias": float(np.mean(d)),
        "rms": float(np.sqrt(np.mean(d ** 2))),
        "added_std": float(np.sqrt(max(var - 0.0, 0.0))),
        "nu_lag1": nu,
        "f_irr": float(1.0 - nu) if np.isfinite(nu) else float("nan"),
    }
