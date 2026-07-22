param(
    [string]$DataRoot = 'D:\Data\dermamnist',
    [string]$OutputRoot = 'output\dermamnist_coop_native_vptdeep_search',
    [int]$BatchSize = 32
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$python = 'D:\Anaconda\python.exe'

Set-Location $repo
& $python -u scripts\coopvpt\search_dermamnist.py `
    --data-root $DataRoot `
    --output-root $OutputRoot `
    --batch-size $BatchSize `
    --python $python
if ($LASTEXITCODE -ne 0) {
    throw "CoOp/VPT parameter search failed with exit code $LASTEXITCODE"
}
