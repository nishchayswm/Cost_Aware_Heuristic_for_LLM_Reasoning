param([int]$Limit=100,[int]$Seed=42,[string]$Python="python")
$ErrorActionPreference="Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
& $Python -u -m costbench run --limit $Limit --seed $Seed
exit $LASTEXITCODE
