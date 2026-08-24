"""Stage 2 preflight: refuse to time anything on a contaminated machine.

The contract requires an idle baseline taken "with no other work at all, agent
sessions included", and a per-session contamination check keyed on external
state rather than on the timings themselves. This is the executable form of
that clause, and it runs *before* any timing so a contaminated campaign is
never produced rather than being detected afterwards.

It fails on this machine while an agent session is running, which is the point:
an interactive assistant burning a third of a core is exactly the "other work"
the clause excludes, and a benchmark that timed itself alongside one would be
measuring the assistant.

Usage::

    uv run python benchmarks/timing/preflight.py            # full 30-sample baseline
    uv run python benchmarks/timing/preflight.py --probe    # single instantaneous read
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.timing.accuracy_thresholds import (  # noqa: E402
    CONTAMINATION,
    benchmark_host_mismatches,
)

#: Processes whose presence invalidates a baseline outright, whatever the load
#: average happens to read. A quiet-looking average does not make an agent
#: session absent; it makes it briefly idle.
DISQUALIFYING_AGENT_PROCESSES = ("claude", "codex")
DISQUALIFYING_BUSY_PROCESSES = ("pytest", "uv")


def load_average() -> float:
    """One-minute load average."""
    output = subprocess.run(["uptime"], capture_output=True, text=True, check=True).stdout
    match = re.search(r"load averages?:\s*([\d.]+)", output)
    if not match:
        raise RuntimeError(f"could not parse load average from {output!r}")
    return float(match.group(1))


def thermal_warnings() -> Dict[str, object]:
    """``pmset -g therm``. Binary on Apple Silicon: a warning was recorded, or not."""
    output = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True).stdout
    recorded = [line.strip() for line in output.splitlines() if "No " not in line and line.strip()]
    return {"warnings_recorded": bool(recorded), "detail": recorded}


def competing_processes() -> List[str]:
    """Named processes that disqualify a baseline, with their CPU share."""
    output = subprocess.run(["ps", "axo", "pcpu,comm"], capture_output=True, text=True, check=True).stdout
    found = []
    for line in output.splitlines()[1:]:
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            continue
        share, command = parts
        try:
            share_value = float(share)
        except ValueError:
            continue
        process_name = Path(command).name.lower()
        if process_name in DISQUALIFYING_AGENT_PROCESSES or (
            share_value >= 1.0 and process_name in DISQUALIFYING_BUSY_PROCESSES
        ):
            found.append(f"{command} ({share_value:.1f}%)")
    return found


def observed_benchmark_host() -> Dict[str, object]:
    """Read the five host fields frozen by Stage 1."""
    model = subprocess.run(["sysctl", "-n", "hw.model"], capture_output=True, text=True, check=True).stdout.strip()
    pmset = shutil.which("pmset")
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "machine_model": model,
        "logical_cpu_count": os.cpu_count(),
        "thermal_command": f"{pmset} -g therm" if pmset else None,
    }


def baseline(samples: int, interval_s: float) -> Dict[str, object]:
    readings = []
    for index in range(samples):
        readings.append(load_average())
        if index + 1 < samples:
            time.sleep(interval_s)
    return {
        "samples": readings,
        "median": statistics.median(readings),
        "max": max(readings),
        "min": min(readings),
    }


def evaluate(
    result: Dict[str, object],
    processes: List[str],
    thermal: Dict[str, object],
    host_problems: List[str] | None = None,
) -> List[str]:
    problems = []
    problems.extend(host_problems or [])
    limit = float(CONTAMINATION["baseline_median_max"])
    if result["median"] > limit:
        problems.append(f"baseline median {result['median']:.2f} exceeds the frozen bound {limit}")
    if processes:
        problems.append("competing work present, which disqualifies a baseline: " + ", ".join(processes))
    if thermal["warnings_recorded"]:
        problems.append(f"thermal or performance warning recorded: {thermal['detail']}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true", help="One instantaneous reading instead of the baseline.")
    parser.add_argument("--json", type=Path, help="Write the reading to this path.")
    args = parser.parse_args()

    samples = 1 if args.probe else int(CONTAMINATION["baseline_samples"])
    interval = 0.0 if args.probe else float(CONTAMINATION["baseline_interval_s"])
    print(f"[preflight] {samples} sample(s) at {interval:g} s")

    result = baseline(samples, interval)
    processes = competing_processes()
    thermal = thermal_warnings()
    host = observed_benchmark_host()
    host_problems = benchmark_host_mismatches(host)
    problems = evaluate(result, processes, thermal, host_problems)

    print(f"[preflight] load median {result['median']:.2f}  min {result['min']:.2f}  max {result['max']:.2f}")
    print(f"[preflight] thermal warnings recorded: {thermal['warnings_recorded']}")
    print(f"[preflight] competing work: {', '.join(processes) if processes else 'none'}")
    print(f"[preflight] host matches frozen contract: {not host_problems}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({**result, "processes": processes, "thermal": thermal, "host": host}, indent=2))

    if problems:
        print("\n[preflight] REFUSING to time on this machine:")
        for problem in problems:
            print(f"  - {problem}")
        print("\nRun this with the agent session closed and the machine otherwise idle.")
        raise SystemExit(1)
    print("\n[preflight] machine is fit to time")


if __name__ == "__main__":
    main()
