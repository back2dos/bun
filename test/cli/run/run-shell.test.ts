import { describe, expect, test } from "bun:test";
import { chmodSync, mkdirSync } from "fs";
import { bunEnv, bunExe, isWindows, tempDir, tmpdirSync } from "harness";
import { join } from "path";

describe.concurrent("run-shell", () => {
  test("running a shell script works", async () => {
    const dir = tmpdirSync();
    mkdirSync(dir, { recursive: true });
    await Bun.write(join(dir, "something.sh"), "echo wah");
    await using proc = Bun.spawn({
      cmd: [bunExe(), join(dir, "something.sh")],
      cwd: dir,
      env: bunEnv,
      stderr: "pipe",
      stdout: "pipe",
    });
    const stdout = await proc.stdout.text();
    const stderr = await proc.stderr.text();
    console.log(stderr);
    expect(stdout).toEqual("wah\n");
  });

  test("invalid syntax reports the error correctly", async () => {
    const dir = tmpdirSync("bun-shell-test-error");
    mkdirSync(dir, { recursive: true });
    const shellScript = `-h)
  echo "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"`;
    await Bun.write(join(dir, "scripts", "script.sh"), shellScript);
    await using proc = Bun.spawn({
      cmd: [bunExe(), join(dir, "scripts", "script.sh")],
      cwd: dir,
      env: bunEnv,
      stderr: "pipe",
      stdout: "pipe",
    });
    const stderr = await proc.stderr.text();
    expect(stderr).toBe("error: Failed to run script.sh due to error Unexpected ')'\n");
  });
});

// Bun's shell runs package.json scripts by default on Windows. The system
// shell rows show that both shells give the same output.
describe.each(isWindows ? ["bun"] : ["bun", "system"])("package script with --shell=%s", shell => {
  const run = async (script: string, env: Record<string, string>, ...args: string[]) => {
    using dir = tempDir("run-shell-braced", {
      "package.json": JSON.stringify({ name: "braced-fixture", version: "1.2.3", scripts: { braced: script } }),
    });
    await using proc = Bun.spawn({
      cmd: [bunExe(), "run", `--shell=${shell}`, "braced", ...args],
      cwd: String(dir),
      env: { ...bunEnv, ...env },
      stdout: "pipe",
      stderr: "pipe",
    });
    const [stdout, stderr, exitCode] = await Promise.all([proc.stdout.text(), proc.stderr.text(), proc.exited]);
    return { stdout, stderr, exitCode };
  };

  test("expands ${NAME}", async () => {
    const { stdout, exitCode } = await run(
      'echo ${BRACED_NAME} "${BRACED_VERSION}" ${npm_lifecycle_event}-x',
      { BRACED_NAME: "braced-fixture", BRACED_VERSION: "1.2.3" },
      "${BRACED_NAME}",
    );
    // The argument after the script name is data. It stays text.
    expect(stdout).toBe("braced-fixture 1.2.3 braced-x ${BRACED_NAME}\n");
    expect(exitCode).toBe(0);
  });

  test("expands ${NAME:-word} and ${NAME:+word}", async () => {
    const script = "echo --port ${BRACED_PORT:-3002} ${BRACED_HOST:+--host $BRACED_HOST} ${BRACED_NONE:+--none}";
    expect(await run(script, { BRACED_HOST: "localhost" })).toMatchObject({
      stdout: "--port 3002 --host localhost\n",
      exitCode: 0,
    });
    expect(await run(script, { BRACED_PORT: "8080" })).toMatchObject({ stdout: "--port 8080\n", exitCode: 0 });
  });

  test("a ${NAME:?} guard stops the script when NAME is not set", async () => {
    const script = 'true "${BRACED_DEPLOY_ENV:?}" && echo deploying to $BRACED_DEPLOY_ENV';
    const unset = await run(script, {});
    expect(unset.stderr).toContain("BRACED_DEPLOY_ENV: parameter");
    expect(unset.stdout).toBe("");
    expect(unset.exitCode).not.toBe(0);
    expect(await run(script, { BRACED_DEPLOY_ENV: "staging" })).toMatchObject({
      stdout: "deploying to staging\n",
      exitCode: 0,
    });
  });
});

test.skipIf(isWindows)(
  "package script shell interpreter is resolved from the original PATH, not node_modules/.bin",
  async () => {
    // A dependency can place arbitrary executables named "bash"/"sh"/"zsh" into
    // node_modules/.bin via its "bin" field. The interpreter that runs
    // package.json scripts must never be picked up from there.
    const fakeShell = "#!/bin/sh\necho FAKE_SHELL_USED\n";
    using dir = tempDir("run-shell-interpreter", {
      "package.json": JSON.stringify({
        name: "shell-interpreter-fixture",
        version: "1.0.0",
        scripts: {
          "say-hi": "echo real-shell-ran",
        },
      }),
      "node_modules/.bin/bash": fakeShell,
      "node_modules/.bin/sh": fakeShell,
      "node_modules/.bin/zsh": fakeShell,
    });
    for (const name of ["bash", "sh", "zsh"]) {
      chmodSync(join(String(dir), "node_modules", ".bin", name), 0o755);
    }

    await using proc = Bun.spawn({
      cmd: [bunExe(), "run", "say-hi"],
      cwd: String(dir),
      env: bunEnv,
      stdout: "pipe",
      stderr: "pipe",
    });

    const [stdout, stderr, exitCode] = await Promise.all([proc.stdout.text(), proc.stderr.text(), proc.exited]);

    // The script must run under a real system shell, not the executables a
    // dependency dropped into node_modules/.bin.
    expect(stdout).not.toContain("FAKE_SHELL_USED");
    expect(stderr).not.toContain("FAKE_SHELL_USED");
    expect(stdout).toContain("real-shell-ran");
    expect(exitCode).toBe(0);
  },
);
