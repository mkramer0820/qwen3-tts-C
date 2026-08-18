[CmdletBinding()]
param(
    [ValidateSet("setup", "doctor", "build", "shell")]
    [string]$Action = "setup",
    [string]$Distro = "Ubuntu-24.04"
)

$ErrorActionPreference = "Stop"
$repoRoot = $PSScriptRoot

function Write-Step([string]$Message) {
    Write-Host "`n==> $Message" -ForegroundColor Cyan
}

Write-Step "Checking the Windows GPU"
$windows = Get-CimInstance Win32_OperatingSystem
if ($windows.Caption -notmatch "Windows 11") {
    throw "AMD's supported ROCm WSL path requires Windows 11; detected '$($windows.Caption)'."
}
$gpus = @(Get-CimInstance Win32_VideoController)
$gpuNames = @($gpus | ForEach-Object Name)
$gpus | ForEach-Object { Write-Host "    $($_.Name) (driver $($_.DriverVersion))" }
if (-not ($gpuNames -match "AMD.*Radeon")) {
    Write-Warning "No AMD Radeon GPU was reported. Continuing so the Linux doctor can provide a specific error."
} elseif ($gpuNames -match "Radeon AI PRO R9700") {
    Write-Host "    Target confirmed: AMD Radeon AI PRO R9700 (RDNA4/gfx1201)" -ForegroundColor Green
}

Write-Step "Checking for Ubuntu 24.04 under WSL2"
$distros = @(wsl.exe --list --quiet 2>$null | ForEach-Object { $_.Replace([char]0, "").Trim() } | Where-Object { $_ })
if ($LASTEXITCODE -ne 0) {
    throw "WSL could not list distributions. Run this script from a normal or Administrator PowerShell window."
}

if ($Distro -notin $distros) {
    Write-Host "Only docker-desktop is not enough for AMD ROCm. Installing $Distro now."
    wsl.exe --install --distribution $Distro
    if ($LASTEXITCODE -ne 0) { throw "WSL failed to install $Distro." }
    Write-Host "Restart Windows if requested, open $Distro once to create its Linux user, then run this same command again."
    exit 0
}

$distroTable = (wsl.exe --list --verbose | Out-String).Replace([char]0, "")
$distroLine = $distroTable -split "`r?`n" | Where-Object { $_ -match [regex]::Escape($Distro) } | Select-Object -First 1
if ($distroLine -notmatch "\s2\s*$") {
    Write-Step "Converting $Distro to WSL2"
    wsl.exe --set-version $Distro 2
    if ($LASTEXITCODE -ne 0) { throw "Could not convert $Distro to WSL2." }
}

Write-Step "Starting the ROCm WSL workflow"
$linuxRepo = (wsl.exe --distribution $Distro -- wslpath -a "$repoRoot").Replace([char]0, "").Trim()
if (-not $linuxRepo) { throw "Could not translate the repository path into a WSL path." }

wsl.exe --distribution $Distro --cd $linuxRepo -- bash training/expressivity-lora/docker/rocm-wsl.sh $Action
if ($LASTEXITCODE -ne 0) { throw "ROCm WSL action '$Action' failed. Read the first ERROR line above for the fix." }
