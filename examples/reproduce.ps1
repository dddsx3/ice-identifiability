# Reproduce every artifact cited in VERDICT.md with one command.
#
#   powershell -ExecutionPolicy Bypass -File examples\reproduce.ps1
#
# Optional:
#   -CciDir <path>  -Cs2Dir <path>  -Root <path to this tool>
#
param(
  [string]$CciDir = "<WORKROOT>\hard_innovation\d2_first_principles\data\cci",
  [string]$Cs2Dir = "<WORKROOT>\hard_innovation\d2_first_principles\data\cs2smos",
  [string]$Root = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = Join-Path $Root "src"
$Out = Join-Path $Root "out"
$Derived = Join-Path $Out "derived"
$Months = @("2018-04", "2019-04", "2020-04")

# The CCI and CS2SMOS files are not redistributable.  Without them this script
# cannot regenerate a single number, so it stops instead of emitting defaults.
if (-not (Test-Path $CciDir) -or -not (Test-Path $Cs2Dir)) {
  Write-Host "Data not found."
  Write-Host "  CCI: $CciDir"
  Write-Host "  CS2SMOS: $Cs2Dir"
  Write-Host "Pass -CciDir and -Cs2Dir, or run only the unit tests (pytest -q)."
  exit 2
}

Write-Host "[1/12] monthly CS2SMOS composites"
New-Item -ItemType Directory -Force -Path (Join-Path $Derived "cs2smos"), (Join-Path $Derived "noise") | Out-Null
foreach ($m in $Months) {
  python (Join-Path $Root "examples\make_monthly.py") --dir $Cs2Dir --month $m `
    --out (Join-Path $Derived "cs2smos\cs2smos_$m.nc")
}

Write-Host "[2/12] CCI, freeboard channel (native snow)"
python -m ice_identifiability run --thickness $CciDir --months 2019-04 `
  --out (Join-Path $Out "example\cci_freeboard") --label CCI-freeboard

Write-Host "[3/12] CCI, declared-posterior channel"
python -m ice_identifiability run --thickness $CciDir --months 2019-04 `
  --raw-channel posterior --out (Join-Path $Out "example\cci_posterior") --label CCI-posterior

Write-Host "[4/12] CS2SMOS native (must be not_assessable: no snow field)"
python -m ice_identifiability run --thickness (Join-Path $Derived "cs2smos") --months 2019-04 `
  --raw-channel posterior --out (Join-Path $Out "example\cs2_native") --label CS2SMOS-native

Write-Host "[5/12] CS2SMOS on the declared shared-snow basis"
python -m ice_identifiability run --thickness (Join-Path $Derived "cs2smos") --months 2019-04 `
  --snow $CciDir --raw-channel posterior `
  --out (Join-Path $Out "example\cs2_common") --label CS2SMOS-common-basis

Write-Host "[6/12] white-noise fixture (must be unidentifiable)"
$tmpl = (Get-ChildItem $CciDir -Filter "*201904*.nc" | Select-Object -First 1).FullName
python (Join-Path $Root "examples\make_noise_fixture.py") --template $tmpl `
  --out (Join-Path $Derived "noise\noise_2019-04.nc")
python -m ice_identifiability run --thickness (Join-Path $Derived "noise\noise_2019-04.nc") `
  --raw-channel posterior --out (Join-Path $Out "example\noise") --label white-noise-fixture

Write-Host "[7/12] discrimination suite"
python (Join-Path $Root "examples\discriminate.py")

Write-Host "[8/12] detection-power calibration (ROC, delta at 80% power)"
python (Join-Path $Root "examples\power_calibration.py")

Write-Host "[9/12] prior sensitivity (flip list, gridpoint envelope)"
python (Join-Path $Root "examples\prior_sensitivity.py")

Write-Host "[10/12] block bootstrap (error bars, year-to-year)"
python (Join-Path $Root "examples\bootstrap_ci.py")

Write-Host "[11/12] G3 threshold traceability and block scale"
python (Join-Path $Root "examples\gate_calibration.py") --cci $CciDir `
  --out (Join-Path $Out "gatesweep")

Write-Host "[12/12] sigma provenance: per-sensor sigma, weight share, version"
python (Join-Path $Root "examples\sigma_provenance.py") --cs2 $Cs2Dir --cci $CciDir `
  --out (Join-Path $Out "sigmaprov")

Write-Host ""
Write-Host "Done. Read:"
Write-Host "  $(Join-Path $Out 'discrimination\discrimination.md')"
Write-Host "  $(Join-Path $Out 'example\cci_freeboard\report.md')"
Write-Host "  $(Join-Path $Out 'example\cs2_common\report.md')"
Write-Host "  $(Join-Path $Out 'example\noise\report.md')"
Write-Host "  $(Join-Path $Out 'power\power.md')"
Write-Host "  $(Join-Path $Out 'stats\ci.md')"
Write-Host "  $(Join-Path $Out 'sensitivity\sensitivity.md')"
Write-Host "  $(Join-Path $Out 'gatesweep\gate_calibration.md')"
