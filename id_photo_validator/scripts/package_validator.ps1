# Package the validator for transfer without copying large datasets.
# It creates dist/validator_bundle.zip containing code, models, and deps lists (but excludes data/, .venv/, caches).

param(
    [string]$Destination = "dist/validator_bundle.zip"
)

$root = Resolve-Path "$PSScriptRoot/.."
Set-Location $root

$excludeDirs = @('data', '.git', '.venv', 'dist', '__pycache__', '.mypy_cache', '.pytest_cache', '.idea', '.vscode')

function ShouldInclude([IO.FileInfo]$item) {
    $relative = $item.FullName.Substring($root.Path.Length + 1)
    $parts = $relative -split '[\\/]+'
    foreach($ex in $excludeDirs){
        if($parts -contains $ex){ return $false }
    }
    return $true
}

$files = Get-ChildItem -Path $root -Recurse -File | Where-Object { ShouldInclude $_ }

$destPath = Resolve-Path . -ErrorAction SilentlyContinue
if(-not (Test-Path "dist")){ New-Item -ItemType Directory -Path "dist" | Out-Null }
$destFull = Join-Path $root $Destination
if(Test-Path $destFull){ Remove-Item $destFull }

Compress-Archive -Path $files.FullName -DestinationPath $destFull -Force
Write-Host "Created bundle at $destFull"
Write-Host "Included $(($files | Measure-Object).Count) files (models/quality_head.joblib, code, requirements)."
Write-Host "Excluded data/, .venv/, dist/, cache folders."
