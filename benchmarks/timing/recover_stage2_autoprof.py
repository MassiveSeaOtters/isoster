#!/usr/bin/env python
"""Recover failed AutoProf records from a retained Stage 2 summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.timing.accuracy_thresholds import (  # noqa: E402
    ENSEMBLE_REALIZATIONS,
    benchmark_host_mismatches,
)
from benchmarks.timing.preflight import observed_benchmark_host  # noqa: E402
from benchmarks.timing.run_stage2_calibration import (  # noqa: E402
    _indicator_sample,
    _interpreter_calibration,
    _json_default,
    _recommendations,
    _request_id,
    _run_monitored_session,
    _set_accuracy_outcomes,
)


def _select_failed_autoprof_records(records):
    selected = {}
    for record in records:
        if record.get("tool") != "autoprof" or record.get("execution_status") != "failed":
            continue
        request_id = _request_id(record)
        if request_id in selected:
            raise ValueError(f"duplicate failed AutoProf request id: {request_id}")
        selected[request_id] = record
    if not selected:
        raise ValueError("source summary contains no failed AutoProf records")
    return selected


def _merge_records(source_records, recovered_records, selected, source_summary):
    recovered = {_request_id(record): record for record in recovered_records}
    if len(recovered) != len(recovered_records):
        raise ValueError("recovery output contains duplicate request ids")
    if set(recovered) != set(selected):
        missing = sorted(set(selected) - set(recovered))
        unexpected = sorted(set(recovered) - set(selected))
        raise ValueError(f"recovery record mismatch; missing={missing}, unexpected={unexpected}")
    merged = []
    for original in source_records:
        request_id = _request_id(original)
        if request_id not in selected:
            merged.append(original)
            continue
        replacement = recovered[request_id]
        replacement["recovery"] = {
            "source_summary": str(source_summary),
            "original_error": original.get("error"),
            "original_failed_wall_s": original.get("failed_wall_s"),
        }
        merged.append(replacement)
    return merged


def _print_selection(selected):
    counts = {}
    for record in selected.values():
        counts[record["session_index"]] = counts.get(record["session_index"], 0) + 1
    print(f"[recovery] selected {len(selected)} failed AutoProf records")
    for session_index, count in sorted(counts.items()):
        print(f"[recovery] session {session_index}: {count} record(s)")


def _run(args):
    if not args.source.is_file():
        raise SystemExit(f"source summary does not exist: {args.source}")
    source = json.loads(args.source.read_text())
    source_records = source.get("records")
    if not isinstance(source_records, list):
        raise SystemExit("source summary has no records list")
    try:
        selected = _select_failed_autoprof_records(source_records)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    _print_selection(selected)
    if args.dry_run:
        return
    if args.output.exists():
        raise SystemExit(f"recovery output already exists: {args.output}")
    host_problems = benchmark_host_mismatches(observed_benchmark_host())
    if host_problems:
        raise SystemExit("recovery host does not match the frozen contract: " + "; ".join(host_problems))

    args.output.mkdir(parents=True)
    recovery = {
        "source_summary": str(args.source),
        "selected_records": len(selected),
        "sessions": [],
        "interpreter": _interpreter_calibration(args.autoprof_python),
    }
    recovered_records = []
    runner = Path(__file__).with_name("run_stage2_calibration.py")
    for session_index in sorted({record["session_index"] for record in selected.values()}):
        request_ids = sorted(
            request_id for request_id, record in selected.items() if record["session_index"] == session_index
        )
        request_path = args.output / f"session_{session_index:02d}_request_ids.json"
        session_path = args.output / f"session_{session_index:02d}.json"
        request_path.write_text(json.dumps(request_ids, indent=2))
        trace = [_indicator_sample()]
        if trace[-1]["contaminated"]:
            status = "contaminated"
        else:
            command = [
                sys.executable,
                str(runner),
                "--session-index",
                str(session_index),
                "--repetitions",
                str(ENSEMBLE_REALIZATIONS),
                "--session-output",
                str(session_path),
                "--request-ids",
                str(request_path),
            ]
            if args.autoprof_python:
                command.extend(["--autoprof-python", args.autoprof_python])
            status = _run_monitored_session(command, trace)
        session_metadata = {"session_index": session_index, "status": status, "trace": trace}
        recovery["sessions"].append(session_metadata)
        (args.output / "recovery.json").write_text(json.dumps(recovery, indent=2, default=_json_default))
        if status != "clean":
            raise SystemExit(f"recovery session {session_index} was {status}; retained output is not merged")
        recovered_records.extend(json.loads(session_path.read_text())["records"])

    merged_records = _merge_records(source_records, recovered_records, selected, args.source)
    failures = _set_accuracy_outcomes(merged_records)
    remaining_failed = [
        record for record in merged_records if record["tool"] == "autoprof" and record["execution_status"] != "ok"
    ]
    recovery.update(
        {
            "recovered_records": len(selected) - len(remaining_failed),
            "remaining_failed_autoprof_records": len(remaining_failed),
        }
    )
    merged_summary = {
        **source,
        "records": merged_records,
        "accuracy_evaluation_failures": failures,
        "recommendations_for_stage3": _recommendations(merged_records),
        "autoprof_recovery": recovery,
    }
    summary_path = args.output / "stage2_calibration_summary_recovered.json"
    summary_path.write_text(json.dumps(merged_summary, indent=2, default=_json_default, allow_nan=True))
    print(f"[recovery] wrote {len(merged_records)} merged records to {summary_path}")
    if remaining_failed:
        raise SystemExit(f"recovery retained {len(remaining_failed)} failed AutoProf record(s)")
    print("[recovery] AutoProf execution coverage restored; accuracy verdicts remain unchanged and recorded")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--autoprof-python")
    parser.add_argument("--dry-run", action="store_true")
    _run(parser.parse_args())


if __name__ == "__main__":
    main()
