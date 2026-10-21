# Builds the f64 CLI and publishes it to the spektrafilm Python repo, so the GUI
# no longer depends on this repository being checked out next to it.
param(
    [string]$Destination = (Join-Path (Split-Path $PSScriptRoot -Parent | Split-Path -Parent) 'spektrafilm\bin')
)

$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
Push-Location $root
try {
    cargo build --release -p spektrafilm-cli --bin spektrafilm-f64
    if ($LASTEXITCODE -ne 0) { throw 'cargo build failed' }
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    Copy-Item (Join-Path $root 'target\release\spektrafilm-f64.exe') (Join-Path $Destination 'spektrafilm-f64.exe') -Force
    $dataDestination = Join-Path $Destination 'data'
    if (Test-Path $dataDestination) { Remove-Item $dataDestination -Recurse -Force }
    Copy-Item (Join-Path $root 'data') $dataDestination -Recurse -Force
    Write-Host "Published spektrafilm-f64.exe to $Destination"
} finally {
    Pop-Location
}
