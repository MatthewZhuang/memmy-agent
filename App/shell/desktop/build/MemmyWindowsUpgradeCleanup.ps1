param(
  [Parameter(Mandatory = $true)][string]$WorkDir
)

Start-Sleep -Seconds 3
Remove-Item -LiteralPath $WorkDir -Recurse -Force -ErrorAction SilentlyContinue
