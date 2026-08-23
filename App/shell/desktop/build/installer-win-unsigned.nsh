!include "FileFunc.nsh"
!include "getProcessInfo.nsh"

!ifndef MEMMY_WM_SETTINGCHANGE
  !define MEMMY_WM_SETTINGCHANGE 0x001A
!endif

!macro customHeader
  !ifdef BUILD_UNINSTALLER
    ; Keep unsigned QA uninstallers usable if Windows or transfer tools touch the NSIS stub.
    CRCCheck off
  !endif
!macroend

; Force per-user installation by skipping the all-users versus current-user choice and using
; %LOCALAPPDATA%\Programs\Memmy, which remains writable by the current user.
; 1) Program Files requires elevation, and a locked uninstaller can make overwrite installation fail.
; 2) Silent background upgrades use NSIS /currentuser and must match the original installation scope.
; 3) A writable per-user directory enables zero-click silent upgrades without elevation.
!macro customInstallMode
  StrCpy $isForceCurrentInstall "1"
!macroend

!ifndef BUILD_UNINSTALLER
  Var MemmyIsRelayedUpgrade
  Var MemmyUpgradeWorkDir
  Var MemmyUpgradeReopenAfterInstall

  ; The 1.0.8 updater starts the downloaded installer from $INSTDIR\data. electron-builder's
  ; normal upgrade path asks the old uninstaller to move all of $INSTDIR, so that running
  ; installer keeps the directory busy and the old uninstaller exits with code 2. Relay only
  ; this legacy --updated invocation through a copy outside $INSTDIR. The relayed child carries
  ; an explicit marker so normal installs and future upgrades continue through electron-builder.
  !macro customInit
    Call MemmyRelayLegacyUpgrade
  !macroend

  !macro customInstall
    Call MemmyRestoreRelayedUpgradeData
    Call MemmyAddCliToUserPath
    Call MemmyInstallLaunchProxy
    !insertmacro MemmyPointShortcutsToLaunchProxy
    Call MemmyClearRelayedUpgradeMarkers
    Call MemmyLaunchRelayedUpgrade
    Call MemmyScheduleRelayedUpgradeCleanup
  !macroend
!endif

!ifdef BUILD_UNINSTALLER
  !macro customUnInstall
    Call un.MemmyRemoveCliFromUserPath
    Call un.MemmyRemoveLaunchProxy
  !macroend
!endif

!ifndef BUILD_UNINSTALLER
Function MemmyRelayLegacyUpgrade
  ${GetParameters} $R0

  ClearErrors
  ${GetOptions} $R0 "--memmy-upgrade-relayed" $R1
  IfErrors memmy_relay_check_legacy
  Call MemmyReadRelayedUpgradeContext
  Return

  memmy_relay_check_legacy:
    ClearErrors
    ${GetOptions} $R0 "--updated" $R1
    IfErrors memmy_relay_done

    System::Call 'kernel32::GetCurrentProcessId() i .r2'
    StrCmp $2 "" memmy_relay_failed
    StrCmp $2 "0" memmy_relay_failed
    ${GetProcessInfo} $2 $2 $3 $4 $5 $6
    StrCmp $3 "" memmy_relay_failed
    StrCmp $3 "0" memmy_relay_failed
    StrCpy $R1 "$LOCALAPPDATA\Memmy\upgrade-staging\$2"
    StrCpy $R3 "$R1\MemmyWindowsUpgradeRelay.ps1"
    StrCpy $R4 "$R1\Memmy-${VERSION}-upgrade.exe"
    StrCpy $R5 "$SYSDIR\WindowsPowerShell\v1.0\powershell.exe"
    StrCpy $R7 "$LOCALAPPDATA\Memmy\upgrade-logs\windows-upgrade.log"
    StrCpy $R8 "$R1\relay-ready"

    ; Direct/manual updates reopen Memmy. A prepared update without the boot-attempt marker was
    ; launched during normal app shutdown and must remain closed. Boot fallback writes .attempt.
    StrCpy $R6 "1"
    IfFileExists "$INSTDIR\data\Memmy\prepared-required-update.json" 0 memmy_relay_stage
    IfFileExists "$INSTDIR\data\Memmy\prepared-required-update.json.attempt" memmy_relay_stage
    StrCpy $R6 "0"

  memmy_relay_stage:
    ; Refresh the launch proxy before the relay moves $INSTDIR\data. The 1.0.8 proxy only knows
    ; the install-local marker lock, which disappears with that move; the refreshed proxy also
    ; recognizes the relay lock outside $INSTDIR and keeps desktop launches blocked throughout.
    Call MemmyInstallLaunchProxy
    ClearErrors
    CreateDirectory "$R1"
    SetOutPath "$R1"
    File /oname=MemmyWindowsUpgradeRelay.ps1 "${BUILD_RESOURCES_DIR}\MemmyWindowsUpgradeRelay.ps1"
    IfErrors memmy_relay_failed
    IfFileExists "$R3" 0 memmy_relay_failed
    File /oname=MemmyWindowsUpgradeCleanup.ps1 "${BUILD_RESOURCES_DIR}\MemmyWindowsUpgradeCleanup.ps1"
    IfErrors memmy_relay_failed
    IfFileExists "$R1\MemmyWindowsUpgradeCleanup.ps1" 0 memmy_relay_failed

    ClearErrors
    CopyFiles /SILENT "$EXEPATH" "$R4"
    IfErrors memmy_relay_failed
    IfFileExists "$R4" 0 memmy_relay_failed
    IfFileExists "$R5" 0 memmy_relay_failed
    Delete "$R8"

    ClearErrors
    ExecShell "open" "$R5" '-NoProfile -NonInteractive -ExecutionPolicy Bypass -WindowStyle Hidden -File $\"$R3$\" -InstallerPath $\"$R4$\" -InstallDir $\"$INSTDIR$\" -OriginalInstallerPid $2 -LegacyHelperPid $3 -ExpectedVersion $\"${VERSION}$\" -ReopenAfterInstall $R6 -ReadyPath $\"$R8$\" -WorkDir $\"$R1$\" -LogPath $\"$R7$\"' SW_HIDE
    IfErrors memmy_relay_failed
    StrCpy $R9 "0"

  memmy_relay_wait_ready:
    IfFileExists "$R8" memmy_relay_ready
    Sleep 100
    IntOp $R9 $R9 + 1
    IntCmp $R9 100 memmy_relay_failed memmy_relay_wait_ready memmy_relay_failed

  memmy_relay_ready:

    ; The old 1.0.8 helper must not reopen the old executable while the relay owns the upgrade.
    SetErrorLevel 1602
    Quit

  memmy_relay_failed:
    SetOutPath "$INSTDIR"
    SetErrorLevel 2
    Quit

  memmy_relay_done:
FunctionEnd

Function MemmyReadRelayedUpgradeContext
  StrCpy $MemmyIsRelayedUpgrade "0"
  ReadEnvStr $MemmyUpgradeWorkDir "MEMMY_UPGRADE_WORK_DIR"
  ReadEnvStr $MemmyUpgradeReopenAfterInstall "MEMMY_UPGRADE_REOPEN_AFTER_INSTALL"
  StrCmp $MemmyUpgradeWorkDir "" memmy_relay_context_failed
  StrCmp $MemmyUpgradeReopenAfterInstall "0" memmy_relay_context_validate_path
  StrCmp $MemmyUpgradeReopenAfterInstall "1" memmy_relay_context_validate_path memmy_relay_context_failed

  memmy_relay_context_validate_path:
    StrCpy $0 "$LOCALAPPDATA\Memmy\upgrade-staging\"
    StrLen $1 $0
    StrLen $2 $MemmyUpgradeWorkDir
    IntCmp $2 $1 memmy_relay_context_failed memmy_relay_context_failed memmy_relay_context_compare_path

  memmy_relay_context_compare_path:
    StrCpy $3 $MemmyUpgradeWorkDir $1
    StrCmp $3 $0 0 memmy_relay_context_failed
    StrCpy $MemmyIsRelayedUpgrade "1"
    Return

  memmy_relay_context_failed:
    SetErrorLevel 2
    Quit
FunctionEnd

Function MemmyRestoreRelayedUpgradeData
  StrCmp $MemmyIsRelayedUpgrade "1" 0 memmy_restore_relayed_data_done
  StrCpy $0 "$MemmyUpgradeWorkDir\data-backup"
  StrCpy $1 "$INSTDIR\data"
  StrCpy $2 "$MemmyUpgradeWorkDir\installer-created-data"
  IfFileExists "$0\*" 0 memmy_restore_relayed_data_failed
  IfFileExists "$2\*" memmy_restore_relayed_data_failed
  IfFileExists "$1\*" 0 memmy_restore_relayed_data_backup
  ClearErrors
  Rename "$1" "$2"
  IfErrors memmy_restore_relayed_data_failed

  memmy_restore_relayed_data_backup:
    ClearErrors
    Rename "$0" "$1"
    IfErrors memmy_restore_relayed_data_failed
    IfFileExists "$1\*" memmy_restore_relayed_data_done memmy_restore_relayed_data_failed

  memmy_restore_relayed_data_failed:
    SetErrorLevel 3
    Quit

  memmy_restore_relayed_data_done:
FunctionEnd

Function MemmyClearRelayedUpgradeMarkers
  StrCmp $MemmyIsRelayedUpgrade "1" 0 memmy_clear_relayed_markers_done
  StrCpy $0 "$INSTDIR\data\Memmy\prepared-required-update.json"
  Delete "$0"
  RMDir /r "$0.lock"
  Delete "$0.prompt"
  Delete "$0.attempt"
  RMDir /r "$LOCALAPPDATA\Memmy\upgrade-staging\active.lock"

  memmy_clear_relayed_markers_done:
FunctionEnd

Function MemmyLaunchRelayedUpgrade
  StrCmp $MemmyIsRelayedUpgrade "1" 0 memmy_launch_relayed_upgrade_done
  StrCmp $MemmyUpgradeReopenAfterInstall "1" 0 memmy_launch_relayed_upgrade_done
  Exec '$\"$INSTDIR\${PRODUCT_FILENAME}.exe$\" --updated'

  memmy_launch_relayed_upgrade_done:
FunctionEnd

Function MemmyScheduleRelayedUpgradeCleanup
  StrCmp $MemmyIsRelayedUpgrade "1" 0 memmy_schedule_relayed_cleanup_done
  StrCpy $0 "$MemmyUpgradeWorkDir\MemmyWindowsUpgradeCleanup.ps1"
  StrCpy $1 "$SYSDIR\WindowsPowerShell\v1.0\powershell.exe"
  IfFileExists "$0" 0 memmy_schedule_relayed_cleanup_done
  IfFileExists "$1" 0 memmy_schedule_relayed_cleanup_done
  ExecShell "open" "$1" '-NoProfile -NonInteractive -ExecutionPolicy Bypass -WindowStyle Hidden -File $\"$0$\" -WorkDir $\"$MemmyUpgradeWorkDir$\"' SW_HIDE

  memmy_schedule_relayed_cleanup_done:
FunctionEnd

Function MemmyPathContains
  Exch $R1
  Exch
  Exch $R0
  StrCpy $R5 "0"
  StrLen $R2 $R1
  StrLen $R3 $R0
  StrCpy $R4 0

  memmy_path_contains_loop:
    StrCpy $R5 $R0 $R2 $R4
    StrCmp $R5 $R1 memmy_path_contains_found
    IntOp $R4 $R4 + 1
    IntCmp $R4 $R3 memmy_path_contains_not_found memmy_path_contains_loop memmy_path_contains_not_found

  memmy_path_contains_found:
    StrCpy $R5 "1"
    Goto memmy_path_contains_done

  memmy_path_contains_not_found:
    StrCpy $R5 "0"

  memmy_path_contains_done:
    Pop $R1
    Exch $R5
FunctionEnd

Function MemmyAddCliToUserPath
  StrCpy $0 "$INSTDIR\resources\cli"
  IfFileExists "$0\memmy.cmd" 0 memmy_add_cli_done
  IfFileExists "$0\memmy-memory.cmd" 0 memmy_add_cli_done

  ReadRegStr $1 HKCU "Environment" "Path"
  StrCmp $1 "" memmy_add_cli_empty

  Push ";$1;"
  Push ";$0;"
  Call MemmyPathContains
  Pop $2
  StrCmp $2 "1" memmy_add_cli_done
  WriteRegExpandStr HKCU "Environment" "Path" "$1;$0"
  Goto memmy_add_cli_broadcast

  memmy_add_cli_empty:
    WriteRegExpandStr HKCU "Environment" "Path" "$0"

  memmy_add_cli_broadcast:
    System::Call 'user32::SendMessageTimeout(i 0xffff, i ${MEMMY_WM_SETTINGCHANGE}, i 0, t "Environment", i 0x0002, i 5000, *i .r0)'

  memmy_add_cli_done:
FunctionEnd

Function MemmyInstallLaunchProxy
  StrCpy $0 "$LOCALAPPDATA\Memmy\launcher"
  CreateDirectory "$0"
  SetOutPath "$0"
  File /oname=Memmy.ico "${BUILD_RESOURCES_DIR}\icon.ico"
  File /oname=MemmyUpdatePrompt.ps1 "${BUILD_RESOURCES_DIR}\MemmyUpdatePrompt.ps1"

  FileOpen $1 "$0\MemmyLauncher.vbs" w
  FileWrite $1 "Set shell = CreateObject($\"WScript.Shell$\")$\r$\n"
  FileWrite $1 "Set fso = CreateObject($\"Scripting.FileSystemObject$\")$\r$\n"
  FileWrite $1 "appExe = $\"$INSTDIR\${PRODUCT_FILENAME}.exe$\"$\r$\n"
  FileWrite $1 "dataRoot = $\"$INSTDIR\data$\"$\r$\n"
  FileWrite $1 "powerShellPath = shell.ExpandEnvironmentStrings($\"%SystemRoot%$\") & $\"\System32\WindowsPowerShell\v1.0\powershell.exe$\"$\r$\n"
  FileWrite $1 "promptPath = $\"$0\MemmyUpdatePrompt.ps1$\"$\r$\n"
  FileWrite $1 "languagePath = dataRoot & $\"\Memmy\update-prompt-language.txt$\"$\r$\n"
  FileWrite $1 "markerPath = dataRoot & $\"\Memmy\prepared-required-update.json$\"$\r$\n"
  FileWrite $1 "lockPath = markerPath & $\".lock$\"$\r$\n"
  FileWrite $1 "relayLockPath = shell.ExpandEnvironmentStrings($\"%LOCALAPPDATA%$\") & $\"\Memmy\upgrade-staging\active.lock$\"$\r$\n"
  FileWrite $1 "If fso.FolderExists(relayLockPath) Then$\r$\n"
  FileWrite $1 "  lockPath = relayLockPath$\r$\n"
  FileWrite $1 "End If$\r$\n"
  FileWrite $1 "promptMarkerPath = markerPath & $\".prompt$\"$\r$\n"
  FileWrite $1 "If fso.FolderExists(lockPath) And fso.FileExists(promptMarkerPath) Then$\r$\n"
  FileWrite $1 "  If fso.FileExists(powerShellPath) And fso.FileExists(promptPath) Then$\r$\n"
  FileWrite $1 "    shell.Run Chr(34) & powerShellPath & Chr(34) & $\" -STA -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File $\" & Chr(34) & promptPath & Chr(34) & $\" -LockPath $\" & Chr(34) & lockPath & Chr(34) & $\" -AppExe $\" & Chr(34) & appExe & Chr(34) & $\" -LanguagePath $\" & Chr(34) & languagePath & Chr(34), 0, True$\r$\n"
  FileWrite $1 "  End If$\r$\n"
  FileWrite $1 "  WScript.Quit 0$\r$\n"
  FileWrite $1 "End If$\r$\n"
  FileWrite $1 "If fso.FolderExists(lockPath) Or Not fso.FileExists(appExe) Then$\r$\n"
  FileWrite $1 "  WScript.Quit 0$\r$\n"
  FileWrite $1 "End If$\r$\n"
  FileWrite $1 "shell.CurrentDirectory = fso.GetParentFolderName(appExe)$\r$\n"
  FileWrite $1 "shell.Run Chr(34) & appExe & Chr(34), 1, False$\r$\n"
  FileClose $1

  SetOutPath "$INSTDIR"
FunctionEnd

!macro MemmyPointShortcutsToLaunchProxy
  StrCpy $0 "$LOCALAPPDATA\Memmy\launcher"
  StrCpy $1 "$0\MemmyLauncher.vbs"
  StrCpy $2 "$0\Memmy.ico"
  StrCpy $4 "0"
  IfFileExists "$1" 0 memmy_point_shortcuts_done

  StrCpy $3 "$newStartMenuLink"
  IfFileExists "$3" 0 memmy_point_desktop_shortcut
  CreateShortCut "$3" "$SYSDIR\wscript.exe" "$\"$1$\"" "$2" 0 "" "" "${APP_DESCRIPTION}"
  ClearErrors
  StrCpy $4 "1"

  memmy_point_desktop_shortcut:
    Push "$CMDLINE"
    Push "no-desktop-shortcut"
    Call MemmyPathContains
    Pop $5
    StrCmp $5 "1" memmy_point_shortcuts_done

    StrCmp $keepShortcuts "false" memmy_point_new_desktop_shortcut
    StrCmp $oldDesktopLink $newDesktopLink memmy_point_existing_new_desktop_shortcut
    IfFileExists "$oldDesktopLink" 0 memmy_point_existing_new_desktop_shortcut
    Rename "$oldDesktopLink" "$newDesktopLink"
    ClearErrors
    Goto memmy_point_new_desktop_shortcut

  memmy_point_existing_new_desktop_shortcut:
    IfFileExists "$newDesktopLink" 0 memmy_point_shortcuts_done

  memmy_point_new_desktop_shortcut:
    StrCpy $3 "$newDesktopLink"
    CreateShortCut "$3" "$SYSDIR\wscript.exe" "$\"$1$\"" "$2" 0 "" "" "${APP_DESCRIPTION}"
    ClearErrors
    StrCpy $4 "1"

  memmy_point_shortcuts_done:
    StrCmp $4 "1" 0 memmy_point_no_shortcut_refresh
    System::Call 'Shell32::SHChangeNotify(i 0x08000000, i 0, i 0, i 0)'

  memmy_point_no_shortcut_refresh:
!macroend
!endif

!ifdef BUILD_UNINSTALLER
Function un.MemmyRemoveCliFromUserPath
  StrCpy $0 "$INSTDIR\resources\cli"
  ReadRegStr $1 HKCU "Environment" "Path"
  StrCmp $1 "" memmy_remove_cli_done

  Push "$1"
  Push "$0;"
  Push ""
  Call un.MemmyRemovePathSegment
  Pop $1
  Push "$1"
  Push ";$0"
  Push ""
  Call un.MemmyRemovePathSegment
  Pop $1
  Push "$1"
  Push "$0"
  Push ""
  Call un.MemmyRemovePathSegment
  Pop $1
  WriteRegExpandStr HKCU "Environment" "Path" "$1"
  System::Call 'user32::SendMessageTimeout(i 0xffff, i ${MEMMY_WM_SETTINGCHANGE}, i 0, t "Environment", i 0x0002, i 5000, *i .r0)'

  memmy_remove_cli_done:
FunctionEnd

Function un.MemmyRemovePathSegment
  Exch $R2
  Exch
  Exch $R1
  Exch
  Exch 2
  Exch $R0
  Push $R3
  Push $R4
  Push $R5
  Push $R6
  Push $R7
  StrCpy $R3 ""
  StrLen $R4 $R0
  StrLen $R5 $R1
  StrCpy $R6 0

  un_memmy_remove_path_loop:
    StrCpy $R7 $R0 $R5 $R6
    StrCmp $R7 $R1 un_memmy_remove_path_match
    StrCpy $R7 $R0 1 $R6
    StrCpy $R3 "$R3$R7"
    IntOp $R6 $R6 + 1
    IntCmp $R6 $R4 un_memmy_remove_path_done un_memmy_remove_path_loop un_memmy_remove_path_done

  un_memmy_remove_path_match:
    StrCpy $R3 "$R3$R2"
    IntOp $R6 $R6 + $R5
    IntCmp $R6 $R4 un_memmy_remove_path_done un_memmy_remove_path_loop un_memmy_remove_path_done

  un_memmy_remove_path_done:
    StrCpy $R0 $R3
    Pop $R7
    Pop $R6
    Pop $R5
    Pop $R4
    Pop $R3
    Pop $R2
    Pop $R1
    Exch $R0
FunctionEnd

Function un.MemmyRemoveLaunchProxy
  Push $R0
  Push $R1
  Push $R2
  Push $R3
  Push $R4
  Push $R5

  StrCpy $R0 "$CMDLINE"
  StrCpy $R1 "keep-shortcuts"
  StrLen $R2 $R1
  StrLen $R3 $R0
  StrCpy $R4 0

  un_memmy_keep_shortcuts_loop:
    StrCpy $R5 $R0 $R2 $R4
    StrCmp $R5 $R1 un_memmy_keep_launch_proxy
    IntOp $R4 $R4 + 1
    IntCmp $R4 $R3 un_memmy_remove_launch_proxy un_memmy_keep_shortcuts_loop un_memmy_remove_launch_proxy

  un_memmy_keep_launch_proxy:
    Pop $R5
    Pop $R4
    Pop $R3
    Pop $R2
    Pop $R1
    Pop $R0
    Return

  un_memmy_remove_launch_proxy:
    Pop $R5
    Pop $R4
    Pop $R3
    Pop $R2
    Pop $R1
    Pop $R0
    ReadRegStr $0 SHELL_CONTEXT "Software\${APP_GUID}" "ShortcutName"
    StrCmp $0 "" 0 un_memmy_delete_old_desktop_shortcut
    StrCpy $0 "${PRODUCT_FILENAME}"

  un_memmy_delete_old_desktop_shortcut:
    Delete "$DESKTOP\$0.lnk"
    Delete "$DESKTOP\${SHORTCUT_NAME}.lnk"
    ClearErrors
    RMDir /r "$LOCALAPPDATA\Memmy\launcher"
FunctionEnd
!endif
