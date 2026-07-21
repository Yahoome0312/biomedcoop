param(
    [string]$DataRoot = 'D:\Data\dermamnist',
    [string]$OutputRoot = 'output\dermamnist_repro_final',
    [string]$Model = 'BiomedCLIP',
    [string]$ConsoleLogRoot = "$env:TEMP\biomedcoop_dermamnist_logs"
)

# Windows reproduction driver for the official BiomedCoOp few-shot setup.
# [Reproduction addition] The upstream scripts documented up to 16-shot;
# 32-shot is explicitly included here.
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$python = 'D:\Anaconda\python.exe'
$trainer = "BiomedCoOp_$Model"
$experiment = 'nctx4_cscFalse_ctpend'
$shotsList = @(1, 2, 4, 8, 16, 32)
$seedsList = @(1, 2, 3)

if (-not [IO.Path]::IsPathRooted($OutputRoot)) {
    $OutputRoot = Join-Path $repo $OutputRoot
}
New-Item -ItemType Directory -Force -Path $OutputRoot, $ConsoleLogRoot | Out-Null

$env:PYTHONPATH = Join-Path $repo 'Dassl.pytorch'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
Set-Location $repo

foreach ($shots in $shotsList) {
    foreach ($seed in $seedsList) {
        $runDir = Join-Path $OutputRoot "shots_$shots\$trainer\$experiment\seed$seed"
        $logPath = Join-Path $runDir 'log.txt'
        $checkpoint = Join-Path $runDir 'prompt_learner\model.pth.tar-$([int]100)'
        $consoleLog = Join-Path $ConsoleLogRoot "shots${shots}_seed${seed}.log"

        if ((Test-Path -LiteralPath $checkpoint) -and
            (Test-Path -LiteralPath $logPath) -and
            (Select-String -LiteralPath $logPath -Pattern '\* auc:' -Quiet)) {
            Write-Output "SKIP shots=$shots seed=$seed (complete result exists)"
            continue
        }

        Write-Output "RUN shots=$shots seed=$seed"
        $trainArgs = @(
            '-u', 'train.py',
            '--root', $DataRoot,
            '--seed', "$seed",
            '--trainer', $trainer,
            '--dataset-config-file', 'configs/datasets/dermamnist.yaml',
            '--config-file', 'configs/trainers/BiomedCoOp/few_shot/dermamnist.yaml',
            '--output-dir', $runDir,
            'TRAINER.BIOMEDCOOP.N_CTX', '4',
            'TRAINER.BIOMEDCOOP.CSC', 'False',
            'TRAINER.BIOMEDCOOP.CLASS_TOKEN_POSITION', 'end',
            'DATASET.NUM_SHOTS', "$shots",
            'DATALOADER.NUM_WORKERS', '4',
            'DATALOADER.PERSISTENT_WORKERS', 'True'
        )

        & $python @trainArgs *> $consoleLog
        if ($LASTEXITCODE -ne 0) {
            throw "BiomedCoOp failed for shots=$shots seed=$seed; see $consoleLog"
        }
        Write-Output "DONE shots=$shots seed=$seed"
    }
}

Write-Output "All requested DermaMNIST BiomedCoOp runs completed."
