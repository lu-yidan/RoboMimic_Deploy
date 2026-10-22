param(
    [string]$OutputDir = ('outputs/cycle_reproduction_' + (Get-Date -Format 'yyyyMMdd_HHmmss')),
    [string]$Plan = 'outputs/compact_cycle_plan_v13/plan.npz',
    [string]$ConfigName = 'mujoco_climb_cycle',
    [switch]$SkipVideo
)
$ErrorActionPreference = 'Stop'
$repoPath = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $repoPath
try {
    $env:OPENBLAS_NUM_THREADS = '1'
    $env:OMP_NUM_THREADS = '1'
    $pythonPath = Join-Path $repoPath '.venv/Scripts/python.exe'
    if (!(Test-Path -LiteralPath $Plan)) {
        throw "Reference missing: $Plan. See docs/baselines/compact_gait_20261009.md for the complete planning command."
    }
    & $pythonPath -X utf8 -u deploy_mujoco/deploy_mujoco.py --config-name $ConfigName hydra.job.chdir=false "climb_cycle.plan=$Plan" "climb_cycle.output_dir=$OutputDir"
    if ($LASTEXITCODE -ne 0) { throw 'Simulation process failed.' }
    $summary = Get-Content -LiteralPath (Join-Path $OutputDir 'summary.json') -Raw | ConvertFrom-Json
    & $pythonPath -X utf8 tools/audit_climb_cycle.py --run $OutputDir
    if ($LASTEXITCODE -ne 0) { throw 'Independent audit process failed.' }
    $audit = Get-Content -LiteralPath (Join-Path $OutputDir 'independent_audit.json') -Raw | ConvertFrom-Json
    if (!$SkipVideo) {
        & $pythonPath -X utf8 tools/render_climb_cycle.py --run $OutputDir
        if ($LASTEXITCODE -ne 0) { throw 'Video rendering failed; simulation logs remain saved.' }
        & $pythonPath -X utf8 tools/render_climb_cycle.py --run $OutputDir --speed 4
        if ($LASTEXITCODE -ne 0) { throw 'Preview rendering failed; full video remains saved.' }
    }
    if (!$summary.passed -or !$audit.passed) { throw "Cycle acceptance FAILED. Inspect $OutputDir/summary.json and independent_audit.json." }
    Write-Output "Cycle acceptance PASSED: $OutputDir"
}
finally { Pop-Location }
