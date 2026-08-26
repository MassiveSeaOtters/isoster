#!/usr/bin/env python
"""Run the frozen Part B Stage 4 three-way timing campaign."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.timing.accuracy_thresholds import CONTAMINATION, benchmark_host_mismatches  # noqa: E402
from benchmarks.timing.preflight import (  # noqa: E402
    baseline,
    competing_processes,
    evaluate,
    observed_benchmark_host,
    thermal_warnings,
)
from benchmarks.timing.run_stage2_calibration import (  # noqa: E402
    MAX_RETRIES,
    _arm_label,
    _indicator_sample,
    _interpreter_calibration,
    _json_default,
    _mark_attempt_contaminated,
    _run_monitored_session,
    _set_accuracy_outcomes,
    _summary,
    _wait_for_clean_retry,
)
from benchmarks.timing.stage3_parameters import (  # noqa: E402
    FROZEN_STAGE3_PARAMETERS,
    load_stage3_parameters,
)


def _timing_summaries(records):
    grouped = defaultdict(list)
    for record in records:
        if record["timing_eligible"]:
            grouped[_arm_label(record)].append(record)
    summaries = {}
    for label, arm_records in sorted(grouped.items()):
        session_medians = []
        for session_index in sorted({record["session_index"] for record in arm_records}):
            session_medians.append(
                statistics.median(
                    record["fit_plus_harness_s"] for record in arm_records if record["session_index"] == session_index
                )
            )
        summaries[label] = {
            "fit_only_s": _summary(record["fit_only_s"] for record in arm_records),
            "fit_plus_harness_s": _summary(record["fit_plus_harness_s"] for record in arm_records),
            "session_medians_s": session_medians,
            "session_median_interval_s": [min(session_medians), max(session_medians)],
            "calls_per_batch": arm_records[0]["calls_per_batch"],
            "ring_count_median": statistics.median(record["ring_count"] for record in arm_records),
            "coverage_fraction_min": min(record["coverage_fraction"] for record in arm_records),
        }
    return summaries


def _set_timing_eligibility(records):
    for record in records:
        record["timing_eligible"] = (
            record["execution_status"] == "ok"
            and record["coverage_status"] == "complete"
            and record["contamination_status"] == "clean"
        )


def _preflight(output):
    sample_count = int(CONTAMINATION["baseline_samples"])
    interval_s = float(CONTAMINATION["baseline_interval_s"])
    print(f"[stage4] preflight: {sample_count} load samples at {interval_s:g} s", flush=True)
    readings = baseline(sample_count, interval_s)
    processes = competing_processes()
    thermal = thermal_warnings()
    host = observed_benchmark_host()
    problems = evaluate(readings, processes, thermal, benchmark_host_mismatches(host))
    payload = {**readings, "processes": processes, "thermal": thermal, "host": host}
    (output / "baseline.json").write_text(json.dumps(payload, indent=2))
    if problems:
        raise SystemExit("Stage 4 preflight failed: " + "; ".join(problems))
    print(f"[stage4] preflight passed; load median {readings['median']:.2f}", flush=True)
    return payload


def _run(args):
    parameters = load_stage3_parameters(args.parameters)
    expected_records = (
        len(parameters["calls_per_batch_by_arm"]) * parameters["sessions"] * parameters["repetitions_per_session"]
    )
    expected_calls = (
        sum(parameters["calls_per_batch_by_arm"].values())
        * parameters["sessions"]
        * parameters["repetitions_per_session"]
    )
    print(
        f"[stage4] frozen plan: {len(parameters['calls_per_batch_by_arm'])} arms, "
        f"{expected_records} records, {expected_calls} timed calls",
        flush=True,
    )
    if args.dry_run:
        return
    if args.output.exists():
        raise SystemExit(f"Stage 4 output already exists: {args.output}")
    host_problems = benchmark_host_mismatches(observed_benchmark_host())
    if host_problems:
        raise SystemExit("Stage 4 host does not match the frozen contract: " + "; ".join(host_problems))
    if _indicator_sample()["contaminated"]:
        raise SystemExit("Stage 4 preflight found a thermal warning or competing process")
    args.output.mkdir(parents=True)
    preflight = _preflight(args.output)

    attempts = []
    accepted = None
    session_runner = Path(__file__).with_name("run_stage2_calibration.py")
    for attempt_index in range(MAX_RETRIES + 1):
        attempt_dir = args.output / f"attempt_{attempt_index + 1:02d}"
        attempt_dir.mkdir()
        trace = [_indicator_sample()]
        attempt = {"attempt": attempt_index + 1, "trace": trace, "status": "running"}
        attempts.append(attempt)
        if attempt_index:
            _wait_for_clean_retry(trace)
        status = "contaminated" if trace[-1]["contaminated"] else "clean"
        interpreter = None
        if status == "clean":
            try:
                interpreter = _interpreter_calibration(args.autoprof_python)
            except Exception as error:  # noqa: BLE001
                status = "failed"
                attempt["error"] = f"interpreter calibration failed: {type(error).__name__}: {error}"
        for session_index in range(parameters["sessions"]):
            if status != "clean":
                break
            print(f"[stage4] starting session {session_index}", flush=True)
            session_output = attempt_dir / f"session_{session_index:02d}.json"
            command = [
                sys.executable,
                str(session_runner),
                "--session-index",
                str(session_index),
                "--repetitions",
                str(parameters["repetitions_per_session"]),
                "--session-output",
                str(session_output),
                "--stage",
                "campaign",
                "--timing-parameters",
                str(args.parameters),
            ]
            if args.autoprof_python:
                command.extend(["--autoprof-python", args.autoprof_python])
            status = _run_monitored_session(command, trace)
            print(f"[stage4] session {session_index} finished with status: {status}", flush=True)
        attempt.update({"status": status, "interpreter": interpreter})
        if status == "contaminated":
            _mark_attempt_contaminated(attempt_dir)
        (attempt_dir / "attempt.json").write_text(json.dumps(attempt, indent=2))
        if status == "clean":
            accepted = attempt_dir
            break
        if status == "failed":
            raise SystemExit("Stage 4 session failed; inspect the retained attempt")
    if accepted is None:
        raise SystemExit("Stage 4 remained contaminated after three retries")

    records = []
    sessions = []
    for path in sorted(accepted.glob("session_*.json")):
        payload = json.loads(path.read_text())
        sessions.append({key: value for key, value in payload.items() if key != "records"})
        records.extend(payload["records"])
    _set_timing_eligibility(records)
    accuracy_failures = _set_accuracy_outcomes(records)
    summary = {
        "stage": "Part B Stage 4 three-way timing campaign",
        "note": "It is not a survey; accuracy is descriptive and remains separate from timing eligibility.",
        "parameters": parameters,
        "host": observed_benchmark_host(),
        "baseline": preflight,
        "attempts": attempts,
        "accepted_attempt": str(accepted),
        "sessions": sessions,
        "records": records,
        "accuracy_evaluation_failures": accuracy_failures,
        "timing_summaries": _timing_summaries(records),
    }
    summary_path = args.output / "stage4_timing_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=_json_default, allow_nan=True))
    eligible = sum(record["timing_eligible"] for record in records)
    print(f"[stage4] retained {len(records)} records; {eligible} timing-eligible", flush=True)
    print(f"[stage4] summary: {summary_path}", flush=True)
    if len(records) != expected_records or eligible != expected_records:
        raise SystemExit("Stage 4 execution, coverage, or contamination check failed")
    if len(summary["timing_summaries"]) != len(parameters["calls_per_batch_by_arm"]):
        raise SystemExit("Stage 4 timing summary is missing one or more arms")
    print("[stage4] campaign complete; accuracy outcomes are retained descriptively", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parameters", type=Path, default=FROZEN_STAGE3_PARAMETERS)
    parser.add_argument("--autoprof-python")
    parser.add_argument("--dry-run", action="store_true")
    _run(parser.parse_args())


if __name__ == "__main__":
    main()
