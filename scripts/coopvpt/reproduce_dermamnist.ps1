param(
    [string]$DataRoot = 'D:\Data\dermamnist',
    [string]$SearchRoot = 'output\dermamnist_coopvpt_search',
    [string]$OutputRoot = 'output\dermamnist_coopvpt_final',
    [int]$BatchSize = 16,
    [string[]]$Methods = @('PureCoOp_AdamW', 'CoOp_VPT_Shallow', 'CoOp_VPT_Deep')
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
    '--python', $python,
    '--methods'
) + $Methods
& $python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "CoOp/VPT reproduction failed with exit code $LASTEXITCODE"
}
