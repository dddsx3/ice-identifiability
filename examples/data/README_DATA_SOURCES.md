# Data sources and synthetic fixtures

This directory contains the four redistributable synthetic fixtures and their
ground-truth record. **No observational data and no restricted product is
included.** Everything here is generated from a fixed seed by the script named
below, contains no third-party content, and is distributed under the same MIT
licence as the rest of the package (see `../../LICENSE`).

This file is the authoritative record of what the shipped fixtures are, how
they were generated, and how to check that they have not changed.

## Contents

| File | Bytes | SHA-256 | What it declares | Expected exit code |
|---|---:|---|---|---:|
| `synth_sigma_grid.nc` | 1,284,707 | `8b69a41e09fa91363c0bd2b670c2c9eac6ec23abd4f62e8bc69c8afe270993c0` | thickness, thickness uncertainty, snow, snow uncertainty | 0 |
| `synth_nosigma_grid.nc` | 765,565 | `fdc2e3e0b90a2c1ac69ff3e9e83c9772a2a657b55dbf4aeb8c3d536552a75cad` | thickness, snow, snow uncertainty; **no** thickness uncertainty | 2 |
| `synth_nosnow_grid.nc` | 588,575 | `6356321514a6b1699e0db1a1dd6678a34ed980203e19db7bd7d1dff5226e8d8c` | thickness, thickness uncertainty; **no** snow | 2 |
| `synth_white_noise.nc` | 938,301 | `26150aec29564498e899bb07896abf494707925a063e7de3de86659a5ebec308` | thickness (white noise), thickness uncertainty, snow, snow uncertainty | 3 |
| `synth_truth.json` | 1,677 | `1ce35467143d165cffe96d194932012d7899048f1e7108ee6ce8c0e0a8845bdb` | target alpha, correlation length, per-file checksums | — |

Total fixture size: 3,577,148 bytes (3.58 MB decimal), consistent with the
value stated in the manuscript.

The fifth synthetic path in the paper (the freeboard channel, exit code 0)
uses `synth_sigma_grid.nc` with the channel selector set to freeboard; it needs
no additional file.

## Generation

Produced in one run by:

```bash
python examples/make_synthetic_fixtures.py --out examples/data
```

- Seed: `20261108` (fixed; no other randomness enters the fixtures).
- Grid: 60 x 60 EASE2 window at 25 km, twelve monthly steps.
- Thickness and snow are generated at 6.25 km and block-averaged to 25 km, so
  the product grid sees the same aggregation that a real product sees.
- Thickness is the sum of a coherent component (correlation length about 90 km)
  and a small grid-scale texture; snow is smoother and shallower; both are
  clipped to physically plausible ranges.
- For each latitude band the generator chooses a target alpha and inverts the
  identity for the snow-depth uncertainty, so the shipped fields carry the
  target to machine precision at the generation scale. The target values and
  the correlation length are written to `synth_truth.json`.
- `synth_white_noise.nc` replaces the coherent component with white noise while
  keeping the same declared uncertainty and the same marginal distribution.

`examples/make_noise_fixture.py` is an auxiliary generator that rebuilds a
white-noise fixture on a real product's grid. It is **not** needed for the
shipped set and is not required by any result in the paper.

## Verifying the shipped files

PowerShell:

```powershell
Get-FileHash examples/data/*.nc, examples/data/synth_truth.json -Algorithm SHA256
```

Python:

```python
import hashlib, json, pathlib
root = pathlib.Path("examples/data")
truth = json.loads((root / "synth_truth.json").read_text())
for name, expected in truth["files"].items():
    digest = hashlib.sha256((root / name).read_bytes()).hexdigest()
    print(name, digest == expected)
```

The five values in the table above are the ones recorded in
`synth_truth.json`; if a script reports a mismatch, the file has been
regenerated or altered and the result is not the one reported in the paper.

**Caveat on byte-identical regeneration.** NetCDF files carry container-level
metadata (HDF5 library version, creation order, chunk layout). Regenerating the
fixtures with a different `netCDF4`/HDF5 build can therefore change the file
hash even when every array value is identical. The hashes above were produced
in the pinned environment described in the manuscript (Python 3.14, NumPy 2.4,
`netCDF4`, `xarray`). If you regenerate in a different environment, compare the
arrays rather than the bytes, and record the new hashes in `synth_truth.json`
instead of reporting a false mismatch.

## Reproducing the paper's synthetic results without restricted data

With the shipped fixtures in place, no regeneration and no real product is
needed:

```powershell
powershell -File examples/reproduce_synthetic.ps1 -KeepFixtures
```

This runs the five synthetic audits on the shipped files, checks the exit codes
(0, 0, 2, 2, 3), and executes the regression suite (`pytest tests -q`). The
manuscript's Data availability statement points to this path.

## Real products (not included)

ESA CCI v4.0, AWI CS2SMOS v206, PIOMAS and the CPOM NTC point cloud are **not**
redistributed here. They remain under their producers' terms and must be
obtained from the producers; the scripts expect them at the paths documented in
`run_all.ps1`. The four determinations in the manuscript's real-product section
require those files; every synthetic determination requires only this directory.
