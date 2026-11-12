# Reproduce the synthetic path with no restricted data.
#
#   powershell -ExecutionPolicy Bypass -File examples\reproduce_synthetic.ps1
#
# Generates the fixtures (fixed seed), runs the five audits that the paper
# reports, prints the exit codes and then runs the regression tests.
param(
  [string]$Root = (Split-Path -Parent $PSScriptRoot),
  [string]$Out  = "out_synth",
  [switch]$KeepFixtures
)
$ErrorActionPreference = "Stop"
$env:PYTHONPATH = Join-Path $Root "src"
Set-Location $Root

$data = Join-Path $Root "examples\data"
if (-not $KeepFixtures -or -not (Test-Path (Join-Path $data "synth_truth.json"))) {
  Write-Host "[1/3] generate fixtures (seed 20261108)"
  python examples\make_synthetic_fixtures.py --out $data
} else {
  Write-Host "[1/3] keep existing fixtures in $data"
}

Write-Host "[2/3] five audits"
$runs = @(
  @{n = "S1_synth_sigma";           f = "synth_sigma_grid.nc";  a = @("--raw-channel", "posterior") },
  @{n = "S2_synth_sigma_freeboard"; f = "synth_sigma_grid.nc";  a = @("--raw-channel", "freeboard") },
  @{n = "S3_synth_nosigma";         f = "synth_nosigma_grid.nc"; a = @() },
  @{n = "S4_synth_nosnow";          f = "synth_nosnow_grid.nc";  a = @("--raw-channel", "posterior") },
  @{n = "S5_synth_noise";           f = "synth_white_noise.nc";  a = @("--raw-channel", "posterior") }
)
foreach ($r in $runs) {
  python -m ice_identifiability run --thickness (Join-Path $data $r.f) `
    --out (Join-Path $Out $r.n) --label $r.n @($r.a)
  Write-Host ("      {0} -> exit {1}" -f $r.n, $LASTEXITCODE)
}

Write-Host "[3/3] regression tests"
python -m pytest tests -q
