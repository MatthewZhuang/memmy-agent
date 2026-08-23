import { execFile as execFileCallback, spawn, type ChildProcess } from "node:child_process";
import { existsSync } from "node:fs";
import { copyFile, mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { promisify } from "node:util";
import { afterEach, describe, expect, it } from "vitest";

const execFile = promisify(execFileCallback);
const relayScriptPath = resolve(import.meta.dirname, "../build/MemmyWindowsUpgradeRelay.ps1");
const cleanupScriptPath = resolve(import.meta.dirname, "../build/MemmyWindowsUpgradeCleanup.ps1");
const temporaryDirectories: string[] = [];
const helperProcesses: ChildProcess[] = [];
const descendantProcessIds: number[] = [];
const describeOnWindows = process.platform === "win32" ? describe : describe.skip;

afterEach(async () => {
  await Promise.all(helperProcesses.splice(0).map(async (process) => {
    if (process.exitCode !== null) return;
    await new Promise<void>((resolvePromise) => {
      process.once("exit", () => resolvePromise());
      process.kill();
    });
  }));
  for (const processId of descendantProcessIds.splice(0)) {
    try {
      process.kill(processId);
    } catch {
      // The descendant may already have exited.
    }
    for (let attempt = 0; attempt < 30; attempt += 1) {
      try {
        process.kill(processId, 0);
        await new Promise((resolvePromise) => setTimeout(resolvePromise, 100));
      } catch {
        break;
      }
    }
  }
  await Promise.all(temporaryDirectories.splice(0).map((directory) => rm(directory, {
    recursive: true,
    force: true,
    maxRetries: 20,
    retryDelay: 100
  })));
});

const createRelayFixture = async (installerExitCode: number, options: { spawnLongLivedDescendant?: boolean } = {}) => {
  const root = await mkdtemp(join(tmpdir(), "memmy-upgrade-relay-"));
  temporaryDirectories.push(root);
  const installDir = join(root, "installed Memmy");
  const dataDir = join(installDir, "data", "Memmy");
  const workDir = join(root, "relay-work");
  const logPath = join(root, "logs", "windows-upgrade.log");
  const installerPath = join(root, `fake-installer-${installerExitCode}.cmd`);
  const descendantPidPath = join(root, "installer-descendant.pid");
  const descendantScriptPath = join(root, "installer-descendant.ps1");
  await mkdir(dataDir, { recursive: true });
  await mkdir(workDir, { recursive: true });
  await copyFile(cleanupScriptPath, join(workDir, "MemmyWindowsUpgradeCleanup.ps1"));
  await writeFile(join(dataDir, "sentinel.txt"), "keep-me", "utf8");
  await writeFile(join(dataDir, "prepared-required-update.json"), "{}", "utf8");
  await mkdir(join(dataDir, "prepared-required-update.json.lock"), { recursive: true });
  await writeFile(join(dataDir, "prepared-required-update.json.prompt"), "prompt", "utf8");
  await writeFile(join(dataDir, "prepared-required-update.json.attempt"), "1.0.9", "utf8");
  const windowsDirectory = process.env.SystemRoot ?? "C:\\Windows";
  const appStubPath = join(windowsDirectory, "System32", "where.exe");
  await writeFile(join(installDir, "Memmy.exe"), await readFile(appStubPath));
  const escapedInstallDir = installDir.replaceAll("%", "%%");
  const escapedAppStubPath = appStubPath.replaceAll("%", "%%");
  const installerLines = [
    "@echo off",
  ];
  if (options.spawnLongLivedDescendant) {
    const powershellPath = join(windowsDirectory, "System32", "WindowsPowerShell", "v1.0", "powershell.exe");
    await writeFile(descendantScriptPath, [
      `$PID | Set-Content -LiteralPath '${descendantPidPath.replaceAll("'", "''")}'`,
      "Start-Sleep -Seconds 20",
      ""
    ].join("\r\n"), "utf8");
    installerLines.push(`start "" /b "${powershellPath.replaceAll("%", "%%")}" -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "${descendantScriptPath.replaceAll("%", "%%")}"`);
  }
  installerLines.push(
    `rmdir /s /q "${escapedInstallDir}"`,
    `mkdir "${escapedInstallDir}"`,
    `copy /y "${escapedAppStubPath}" "${escapedInstallDir}\\Memmy.exe" >nul`,
    "if not defined MEMMY_UPGRADE_WORK_DIR goto installer_done",
    ">\"%MEMMY_UPGRADE_WORK_DIR%\\child-reopen-intent.txt\" echo(%MEMMY_UPGRADE_REOPEN_AFTER_INSTALL%",
    `move /y "%MEMMY_UPGRADE_WORK_DIR%\\data-backup" "${escapedInstallDir}\\data" >nul`,
    ":installer_done",
    `exit /b ${installerExitCode}`,
    ""
  );
  await writeFile(installerPath, installerLines.join("\r\n"), "utf8");
  return { root, installDir, dataDir, workDir, logPath, installerPath, descendantPidPath };
};

const runRelay = async (
  fixture: Awaited<ReturnType<typeof createRelayFixture>>,
  options: { legacyHelperPid?: number; reopenAfterInstall?: "0" | "1" } = {}
) => {
  const powershellPath = join(process.env.SystemRoot ?? "C:\\Windows", "System32", "WindowsPowerShell", "v1.0", "powershell.exe");
  return execFile(powershellPath, [
    "-NoProfile",
    "-ExecutionPolicy",
    "Bypass",
    "-File",
    relayScriptPath,
    "-InstallerPath",
    fixture.installerPath,
    "-InstallDir",
    fixture.installDir,
    "-OriginalInstallerPid",
    "2147483647",
    "-LegacyHelperPid",
    String(options.legacyHelperPid ?? 2147483647),
    "-ExpectedVersion",
    "10.",
    "-ReopenAfterInstall",
    options.reopenAfterInstall ?? "0",
    "-ReadyPath",
    join(fixture.workDir, "relay-ready"),
    "-WorkDir",
    fixture.workDir,
    "-LogPath",
    fixture.logPath
  ], { timeout: 30_000, windowsHide: true });
};

const startLegacyUpdateHelper = async (fixture: Awaited<ReturnType<typeof createRelayFixture>>, reopenAfterInstall: "0" | "1") => {
  const powershellPath = join(process.env.SystemRoot ?? "C:\\Windows", "System32", "WindowsPowerShell", "v1.0", "powershell.exe");
  const helperPath = join(fixture.root, "legacy update helper.ps1");
  const markerPath = join(fixture.dataDir, "prepared-required-update.json");
  await writeFile(helperPath, "param($AppPid, $OpenAfterInstall, $MarkerPath)\nStart-Sleep -Seconds 20\n", "utf8");
  const helper = spawn(powershellPath, [
    "-NoProfile",
    "-File",
    helperPath,
    "43188",
    reopenAfterInstall,
    markerPath
  ], { windowsHide: true, stdio: "ignore" });
  helperProcesses.push(helper);
  await new Promise((resolvePromise) => setTimeout(resolvePromise, 250));
  if (!helper.pid) {
    throw new Error("legacy update helper did not start");
  }
  return helper.pid;
};

const waitForPathAbsent = async (path: string) => {
  const deadline = Date.now() + 10_000;
  while (existsSync(path) && Date.now() < deadline) {
    await new Promise((resolvePromise) => setTimeout(resolvePromise, 100));
  }
};

const waitForPathPresent = async (path: string) => {
  const deadline = Date.now() + 10_000;
  while (!existsSync(path) && Date.now() < deadline) {
    await new Promise((resolvePromise) => setTimeout(resolvePromise, 100));
  }
};

describeOnWindows("Windows upgrade relay", () => {
  it("restores install-local data and clears prepared markers after a verified upgrade", async () => {
    expect(existsSync(relayScriptPath)).toBe(true);
    const fixture = await createRelayFixture(0);
    await runRelay(fixture);

    expect(await readFile(join(fixture.dataDir, "sentinel.txt"), "utf8")).toBe("keep-me");
    expect(existsSync(join(fixture.installDir, "Memmy.exe"))).toBe(true);
    expect(existsSync(join(fixture.dataDir, "prepared-required-update.json"))).toBe(false);
    expect(existsSync(join(fixture.dataDir, "prepared-required-update.json.lock"))).toBe(false);
    expect(existsSync(join(fixture.dataDir, "prepared-required-update.json.prompt"))).toBe(false);
    expect(existsSync(join(fixture.dataDir, "prepared-required-update.json.attempt"))).toBe(false);
    const log = await readFile(fixture.logPath, "utf8");
    expect(log).toContain("data restore verified");
    expect(log).toContain("upgrade verified");
    await waitForPathAbsent(fixture.workDir);
    expect(existsSync(fixture.workDir)).toBe(false);
  });

  it("restores data and retains update markers when the staged installer exits 2", async () => {
    expect(existsSync(relayScriptPath)).toBe(true);
    const fixture = await createRelayFixture(2);
    await expect(runRelay(fixture)).rejects.toMatchObject({ code: 2 });

    expect(await readFile(join(fixture.dataDir, "sentinel.txt"), "utf8")).toBe("keep-me");
    expect(existsSync(join(fixture.dataDir, "prepared-required-update.json"))).toBe(true);
    const log = await readFile(fixture.logPath, "utf8");
    expect(log).toContain("installer exit 2");
    expect(log).toContain("data restore verified");
    expect(log).not.toContain("upgrade verified");
  });

  it("uses the legacy helper's explicit manual reopen intent and lets the child installer restore data", async () => {
    const fixture = await createRelayFixture(0);
    const helperPid = await startLegacyUpdateHelper(fixture, "1");

    await runRelay(fixture, { legacyHelperPid: helperPid, reopenAfterInstall: "0" });

    expect(await readFile(join(fixture.workDir, "child-reopen-intent.txt"), "utf8")).toBe("1\r\n");
    expect(await readFile(join(fixture.workDir, "relay-ready"), "utf8")).toBe("1");
    expect(await readFile(join(fixture.dataDir, "sentinel.txt"), "utf8")).toBe("keep-me");
    const log = await readFile(fixture.logPath, "utf8");
    expect(log).toContain(`reopen intent resolved from legacy helper pid ${helperPid}: 1`);
    expect(log).toContain("data restore verified by child installer");
    await waitForPathAbsent(fixture.workDir);
  });

  it("waits for the installer itself without waiting for its long-lived descendant", async () => {
    const fixture = await createRelayFixture(0, { spawnLongLivedDescendant: true });
    const startedAt = Date.now();

    await runRelay(fixture);

    expect(Date.now() - startedAt).toBeLessThan(10_000);
    await waitForPathPresent(fixture.descendantPidPath);
    expect(existsSync(fixture.descendantPidPath)).toBe(true);
    descendantProcessIds.push(Number.parseInt((await readFile(fixture.descendantPidPath, "utf8")).trim(), 10));
    await waitForPathAbsent(fixture.workDir);
  });
});
