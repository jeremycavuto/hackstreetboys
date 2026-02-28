// app/api/start-battery/route.js
//
// Called by the dashboard on mount. Checks if battery_api.py is already
// running on port 8000 — if not, spawns it as a detached background process.
// The process keeps running even after this request completes.

import { spawn } from "child_process";
import { NextResponse } from "next/server";
import path from "path";
import net from "net";

const BATTERY_PORT = 8000;
const BATTERY_SCRIPT = path.join(process.cwd(), "battery-backend", "battery_api.py");

// Check if something is already listening on port 8000
function isPortInUse(port) {
  return new Promise((resolve) => {
    const tester = net.createConnection({ port, host: "127.0.0.1" });
    tester.once("connect", () => { tester.destroy(); resolve(true); });
    tester.once("error",   () => { resolve(false); });
  });
}

export async function POST() {
  try {
    const alreadyRunning = await isPortInUse(BATTERY_PORT);

    if (alreadyRunning) {
      return NextResponse.json({ status: "already_running" });
    }

    // Spawn python as a detached background process so it outlives this request
    const child = spawn("python", [BATTERY_SCRIPT], {
      detached: true,
      stdio:    "ignore",   // don't pipe stdout/stderr into Next.js
      cwd:      path.dirname(BATTERY_SCRIPT),
    });

    child.unref(); // let Node.js exit without waiting for this process

    // Poll for up to 15s (PyTorch + FastAPI can take a few seconds to load)
    let nowRunning = false;
    for (let i = 0; i < 15; i++) {
      await new Promise((r) => setTimeout(r, 1000));
      nowRunning = await isPortInUse(BATTERY_PORT);
      if (nowRunning) break;
    }

    if (nowRunning) {
      return NextResponse.json({ status: "started" });
    } else {
      return NextResponse.json(
        { status: "failed", message: "Process spawned but port 8000 not responding. Check that python and battery_api.py are accessible." },
        { status: 500 }
      );
    }
  } catch (err) {
    return NextResponse.json(
      { status: "error", message: err.message },
      { status: 500 }
    );
  }
}