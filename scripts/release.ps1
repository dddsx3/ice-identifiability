# Release skeleton.  It prepares a tag and a build; it does not publish,
# and it does not invent a DOI.
#
#   powershell -ExecutionPolicy Bypass -File scripts\release.ps1
#
# After the first Zenodo deposit, add the record DOI to CITATION.cff (the
# commented block at the bottom) and re-tag with the DOI in the tag message.

param(
  [string]$Root = (Split-Path -Parent $PSScriptRoot),
  [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
Set-Location $Root

$line = Select-String -Path (Join-Path $Root "pyproject.toml") -Pattern '^version = "([^"]+)"' |
  Select-Object -First 1
if (-not $line) { throw "version not found in pyproject.toml" }
$version = $line.Matches[0].Groups[1].Value
$tag = "v$version"
Write-Host "version $version -> tag $tag"

if (-not $SkipTests) {
  Write-Host "[1/3] unit tests"
  python -m pytest -q
  if ($LASTEXITCODE -ne 0) { throw "tests failed; not tagging" }
} else {
  Write-Host "[1/3] tests skipped by -SkipTests"
}

Write-Host "[2/3] build sdist + wheel"
python -c "import build"
if ($LASTEXITCODE -eq 0) {
  python -m build
} else {
  Write-Host "module 'build' is not installed; run 'pip install build' then rerun."
}

Write-Host "[3/3] annotated tag"
$inGit = $false
try { git rev-parse --is-inside-work-tree *> $null; $inGit = ($LASTEXITCODE -eq 0) } catch { $inGit = $false }
if ($inGit) {
  $dirty = git status --porcelain
  if ($dirty) { Write-Host "WARNING: working tree is not clean; tag anyway? Ctrl+C to abort." }
  git tag -a $tag -m "ice-identifiability $version"
  if ($LASTEXITCODE -eq 0) { Write-Host "tagged $tag (not pushed)" }
  else { Write-Host "tag $tag already exists or tagging failed" }
} else {
  Write-Host "Not a git repository: no tag created. Initialise and commit, then rerun."
}

Write-Host ""
Write-Host "Next (manual, not performed by this script):"
Write-Host "  1. git push origin $tag"
Write-Host "  2. create a GitHub release from $tag"
Write-Host "  3. archive the release on Zenodo; Zenodo mints the DOI"
Write-Host "  4. put the DOI in CITATION.cff and in the tag annotation, then re-release"
