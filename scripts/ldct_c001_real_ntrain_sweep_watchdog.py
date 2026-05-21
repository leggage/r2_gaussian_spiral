#!/usr/bin/env python3
"""
Re-run ``run_ldct_c001_real_spiral_ntrain_methods_sweep.py`` until it exits 0.
The sweep script skips R2/SAX jobs that already finished, so restarts are safe after fixes.

Example::

  cd /path/to/r2_gaussian
  nohup env CUDA_VISIBLE_DEVICES=0 SAX_TORCH_CUDA_ARCH_LIST=8.9 \\
    /path/to/r2_gaussian_n/bin/python scripts/ldct_c001_real_ntrain_sweep_watchdog.py \\
    --interval_sec 180 --max_attempts 200 \\
    --pidfile output/by_experiment/re_ldctc001_spiral_different_methods_ntrains/.sweep_watchdog.pid \\
    -- \\
    --ntrains 50 100 200 400 800 1000 --only all \\
    --sax_methods naf intratomo lineformer \\
    >> output/.../watchdog.log 2>&1 &

Cron (re-start if killed): run ``scripts/ldct_c001_maybe_start_sweep_watchdog.sh`` every 10 minutes.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

R2_ROOT = Path(__file__).resolve().parents[1]
SWEEP = R2_ROOT / "scripts" / "run_ldct_c001_real_spiral_ntrain_methods_sweep.py"
DEFAULT_PIDFILE = (
    R2_ROOT
    / "output/by_experiment/re_ldctc001_spiral_different_methods_ntrains/.sweep_watchdog.pid"
)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Loop sweep until success; forwards extra args to the sweep script."
    )
    ap.add_argument(
        "--interval_sec",
        type=int,
        default=180,
        help="Sleep after a non-zero sweep exit before retrying.",
    )
    ap.add_argument(
        "--max_attempts",
        type=int,
        default=100,
        help="Maximum sweep invocations (0 = unlimited).",
    )
    ap.add_argument(
        "--pidfile",
        type=str,
        default=str(DEFAULT_PIDFILE),
        help="Write daemon pid while running (empty string to disable).",
    )
    args, passthrough = ap.parse_known_args()
    if passthrough[:1] == ["--"]:
        passthrough = passthrough[1:]

    py = os.environ.get("PY_R2", sys.executable)
    cmd = [py, str(SWEEP), *passthrough]

    pidfile: Path | None = None
    if args.pidfile:
        pidfile = Path(args.pidfile)
        pidfile.parent.mkdir(parents=True, exist_ok=True)
        pidfile.write_text(str(os.getpid()), encoding="utf-8")

    def _clear_pidfile() -> None:
        if not pidfile or not pidfile.is_file():
            return
        try:
            if pidfile.read_text(encoding="utf-8").strip() == str(os.getpid()):
                pidfile.unlink()
        except OSError:
            pass

    try:
        attempt = 0
        while True:
            attempt += 1
            if args.max_attempts > 0 and attempt > args.max_attempts:
                print("[watchdog] max_attempts reached without success", flush=True)
                return 1
            tail = f"/{args.max_attempts}" if args.max_attempts > 0 else ""
            print(
                f"[watchdog] attempt {attempt}{tail}: {' '.join(cmd)}",
                flush=True,
            )
            r = subprocess.run(cmd, cwd=str(R2_ROOT))
            if r.returncode == 0:
                print("[watchdog] sweep finished with exit 0", flush=True)
                return 0
            print(
                f"[watchdog] sweep exited {r.returncode}; "
                f"sleep {args.interval_sec}s then retry",
                flush=True,
            )
            time.sleep(args.interval_sec)
    finally:
        _clear_pidfile()


if __name__ == "__main__":
    sys.exit(main())
