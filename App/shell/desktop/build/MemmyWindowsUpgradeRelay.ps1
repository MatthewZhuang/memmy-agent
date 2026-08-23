param(
  [Parameter(Mandatory = $true)][string]$InstallerPath,
  [Parameter(Mandatory = $true)][string]$InstallDir,
  [Parameter(Mandatory = $true)][int]$OriginalInstallerPid,
  [Parameter(Mandatory = $true)][int]$LegacyHelperPid,
  [Parameter(Mandatory = $true)][string]$ExpectedVersion,
  [Parameter(Mandatory = $true)][ValidateSet('0', '1')][string]$ReopenAfterInstall,
  [Parameter(Mandatory = $true)][string]$ReadyPath,
  [Parameter(Mandatory = $true)][string]$WorkDir,
  [Parameter(Mandatory = $true)][string]$LogPath
)

$ErrorActionPreference = 'Stop'
$dataPath = Join-Path $InstallDir 'data'
$backupPath = Join-Path $WorkDir 'data-backup'
$stagingRoot = Split-Path -Parent $WorkDir
$lockPath = Join-Path $stagingRoot 'active.lock'
$appExe = Join-Path $InstallDir 'Memmy.exe'
$installerExit = 1
$dataMoved = $false
$dataRestored = $false
$lockAcquired = $false
$resolvedReopenAfterInstall = $ReopenAfterInstall

function Write-MemmyUpgradeLog([string]$Message) {
  $logDirectory = Split-Path -Parent $LogPath
  if ($logDirectory) {
    New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
  }
  Add-Content -LiteralPath $LogPath -Value ('[{0:O}] {1}' -f (Get-Date), $Message)
}

function Resolve-MemmyLegacyHelperReopenIntent([int]$HelperPid, [string]$MarkerPath, [string]$Fallback) {
  try {
    $helper = Get-CimInstance -ClassName Win32_Process -Filter ("ProcessId = {0}" -f $HelperPid) -ErrorAction Stop
    $commandLine = [string]$helper.CommandLine
    if (-not $commandLine) {
      throw "legacy helper command line is unavailable"
    }
    $pattern = '(?:^|\s)"?(?<intent>[01])"?\s+"?' + [Regex]::Escape($MarkerPath) + '"?(?:\s|$)'
    $match = [Regex]::Match($commandLine, $pattern, [Text.RegularExpressions.RegexOptions]::IgnoreCase)
    if (-not $match.Success) {
      throw "legacy helper reopen argument is unavailable"
    }
    $intent = $match.Groups['intent'].Value
    Write-MemmyUpgradeLog "reopen intent resolved from legacy helper pid ${HelperPid}: $intent"
    return $intent
  } catch {
    Write-MemmyUpgradeLog "reopen intent fallback=$Fallback legacyHelperPid=$HelperPid reason=$($_.Exception.Message)"
    return $Fallback
  }
}

function Wait-MemmyProcessExit([int]$ProcessId, [int]$TimeoutSeconds) {
  $process = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
  if ($null -eq $process) {
    return
  }
  Write-MemmyUpgradeLog "waiting for original installer pid $ProcessId"
  if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
    throw "original installer pid $ProcessId did not exit"
  }
}

function Move-MemmyDirectory([string]$Source, [string]$Destination) {
  New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Destination) | Out-Null
  for ($attempt = 1; $attempt -le 120; $attempt++) {
    try {
      [System.IO.Directory]::Move($Source, $Destination)
      return
    } catch {
      if ($attempt -eq 120) {
        throw
      }
      Start-Sleep -Milliseconds 500
    }
  }
}

function Restore-MemmyData {
  if (-not $dataMoved) {
    $script:dataRestored = $true
    return
  }
  if (-not (Test-Path -LiteralPath $backupPath -PathType Container)) {
    if (Test-Path -LiteralPath $dataPath -PathType Container) {
      $script:dataRestored = $true
      Write-MemmyUpgradeLog "data restore verified by child installer $dataPath"
      return
    }
    throw "data backup is missing: $backupPath"
  }
  if (Test-Path -LiteralPath $dataPath) {
    $installerDataPath = Join-Path $WorkDir 'installer-created-data'
    if (Test-Path -LiteralPath $installerDataPath) {
      throw "installer-created data backup already exists: $installerDataPath"
    }
    Move-MemmyDirectory -Source $dataPath -Destination $installerDataPath
    Write-MemmyUpgradeLog "preserved installer-created data at $installerDataPath"
  }
  New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
  Move-MemmyDirectory -Source $backupPath -Destination $dataPath
  if (-not (Test-Path -LiteralPath $dataPath -PathType Container)) {
    throw "restored data directory is unavailable: $dataPath"
  }
  $script:dataRestored = $true
  Write-MemmyUpgradeLog "data restore verified $dataPath"
}

function Get-MemmyInstalledVersion {
  if (-not (Test-Path -LiteralPath $appExe -PathType Leaf)) {
    return ''
  }
  $versionInfo = [System.Diagnostics.FileVersionInfo]::GetVersionInfo($appExe)
  foreach ($value in @($versionInfo.ProductVersion, $versionInfo.FileVersion)) {
    if ($value) {
      return $value
    }
  }
  return ''
}

function Clear-MemmyUpdateMarkers {
  $markerPath = Join-Path $dataPath 'Memmy\prepared-required-update.json'
  foreach ($path in @($markerPath, "$markerPath.lock", "$markerPath.prompt", "$markerPath.attempt")) {
    Remove-Item -LiteralPath $path -Recurse -Force -ErrorAction SilentlyContinue
  }
}

function Start-MemmyInstalledApp {
  if (-not (Test-Path -LiteralPath $appExe -PathType Leaf)) {
    Write-MemmyUpgradeLog "app executable is unavailable for reopen: $appExe"
    return
  }
  Start-Process -FilePath $appExe -WorkingDirectory $InstallDir -WindowStyle Normal
  Write-MemmyUpgradeLog "started app $appExe"
}

function Test-MemmyInstalledAppRunning {
  $expectedPath = [System.IO.Path]::GetFullPath($appExe)
  foreach ($process in @(Get-Process -Name 'Memmy' -ErrorAction SilentlyContinue)) {
    try {
      if ([string]::Equals([System.IO.Path]::GetFullPath($process.Path), $expectedPath, [System.StringComparison]::OrdinalIgnoreCase)) {
        return $true
      }
    } catch {
      continue
    }
  }
  return $false
}

function Ensure-MemmyInstalledAppStarted {
  for ($attempt = 1; $attempt -le 4; $attempt++) {
    if (Test-MemmyInstalledAppRunning) {
      Write-MemmyUpgradeLog "app already started by child installer $appExe"
      return
    }
    if ($attempt -lt 4) {
      Start-Sleep -Milliseconds 100
    }
  }
  Start-MemmyInstalledApp
}

function Schedule-MemmyStagingCleanup {
  $cleanupScriptPath = Join-Path $WorkDir 'MemmyWindowsUpgradeCleanup.ps1'
  if (-not (Test-Path -LiteralPath $cleanupScriptPath -PathType Leaf)) {
    return
  }
  $powershellPath = Join-Path $PSHOME 'powershell.exe'
  $cleanupArguments = "-NoProfile -NonInteractive -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$cleanupScriptPath`" -WorkDir `"$WorkDir`""
  Start-Process -FilePath $powershellPath -ArgumentList $cleanupArguments -WorkingDirectory $stagingRoot -WindowStyle Hidden
}

try {
  New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null
  Write-MemmyUpgradeLog "relay starting installer=$InstallerPath installDir=$InstallDir expected=$ExpectedVersion reopenFallback=$ReopenAfterInstall"
  $markerPath = Join-Path $dataPath 'Memmy\prepared-required-update.json'
  $resolvedReopenAfterInstall = Resolve-MemmyLegacyHelperReopenIntent -HelperPid $LegacyHelperPid -MarkerPath $markerPath -Fallback $ReopenAfterInstall
  New-Item -ItemType Directory -Path $lockPath -ErrorAction Stop | Out-Null
  $lockAcquired = $true
  [System.IO.File]::WriteAllText($ReadyPath, $resolvedReopenAfterInstall)
  Write-MemmyUpgradeLog "relay ready reopen=$resolvedReopenAfterInstall"
  Wait-MemmyProcessExit -ProcessId $OriginalInstallerPid -TimeoutSeconds 120

  if (Test-Path -LiteralPath $dataPath -PathType Container) {
    if (Test-Path -LiteralPath $backupPath) {
      throw "refusing to overwrite existing data backup: $backupPath"
    }
    Move-MemmyDirectory -Source $dataPath -Destination $backupPath
    $dataMoved = $true
    Write-MemmyUpgradeLog "data moved to $backupPath"
  }

  $arguments = @('/S', '--updated', '--memmy-upgrade-relayed', '/currentuser', ('/D=' + $InstallDir))
  $env:MEMMY_UPGRADE_WORK_DIR = $WorkDir
  $env:MEMMY_UPGRADE_REOPEN_AFTER_INSTALL = $resolvedReopenAfterInstall
  Write-MemmyUpgradeLog "child installer context workDir=$env:MEMMY_UPGRADE_WORK_DIR reopen=$env:MEMMY_UPGRADE_REOPEN_AFTER_INSTALL"
  $installerProcess = Start-Process -FilePath $InstallerPath -ArgumentList $arguments -PassThru -WindowStyle Hidden
  $installerProcess.WaitForExit()
  $installerExit = if ($null -eq $installerProcess.ExitCode) { 1 } else { $installerProcess.ExitCode }
  Write-MemmyUpgradeLog "installer exit $installerExit"
} catch {
  Write-MemmyUpgradeLog ('relay error: ' + ($_ | Out-String))
  if ($installerExit -eq 0) {
    $installerExit = 1
  }
} finally {
  try {
    Restore-MemmyData
  } catch {
    Write-MemmyUpgradeLog ('data restore failed: ' + ($_ | Out-String))
    $dataRestored = $false
  }
  if ($lockAcquired) {
    Remove-Item -LiteralPath $lockPath -Recurse -Force -ErrorAction SilentlyContinue
  }
}

if (-not $dataRestored) {
  Write-MemmyUpgradeLog "upgrade stopped with recoverable data backup $backupPath"
  exit 3
}

$installedVersion = Get-MemmyInstalledVersion
$upgradeVerified = $installerExit -eq 0 -and $installedVersion.StartsWith($ExpectedVersion, [System.StringComparison]::OrdinalIgnoreCase)
if ($upgradeVerified) {
  Clear-MemmyUpdateMarkers
  Write-MemmyUpgradeLog "upgrade verified installedVersion=$installedVersion"
  if ($resolvedReopenAfterInstall -eq '1') {
    Ensure-MemmyInstalledAppStarted
  }
  Schedule-MemmyStagingCleanup
  exit 0
}

Write-MemmyUpgradeLog "upgrade not verified installedVersion=$installedVersion installerExit=$installerExit"
if ($resolvedReopenAfterInstall -eq '1') {
  Start-MemmyInstalledApp
}
exit $(if ($installerExit -ne 0) { $installerExit } else { 4 })
