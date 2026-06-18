#!/usr/bin/env pwsh
param(
    [Parameter(ValueFromRemainingArguments=$true)]
    $Args
)
$venv = Join-Path $PSScriptRoot '.venv312'
$python = Join-Path $venv 'Scripts\python.exe'
if (-not (Test-Path $python)) {
    Write-Error "Python venv not found at $python. Create it with: python -m venv .venv312 && .\.venv312\Scripts\pip install -r requirements.txt"
    exit 1
}
& $python (Join-Path $PSScriptRoot 'Main.py') @Args