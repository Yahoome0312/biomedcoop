param(
    [string]$DataRoot = 'D:\Data\dermamnist',
    [string]$SearchRoot = 'output\dermamnist_coop_native_vptdeep_search',
    [string]$OutputRoot = 'output\dermamnist_coop_native_vptdeep_adamw',
    [int]$BatchSize = 32
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$python = 'D:\Anaconda\python.exe'
Set-Location $repo

$arguments = @(
    '-u', 'scripts\coopvpt\reproduce_dermamnist.py',
    '--data-root', $DataRoot,
    '--search-root', $SearchRoot,
    '--output-root', $OutputRoot,
    '--batch-size', "$BatchSize",
    '--python', $python
)
& $python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "CoOp/VPT reproduction failed with exit code $LASTEXITCODE"
}
