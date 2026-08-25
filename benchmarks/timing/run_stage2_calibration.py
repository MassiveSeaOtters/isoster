#!/usr/bin/env python
"""Run Part B Stage 2: timing calibration, never the benchmark campaign.

The calibration retains every timing, checks it against the frozen Stage 1
science contract outside the timed region, and recommends only batching and
session structure for Stage 3. Fixed-aperture and natural end-to-end results
remain separate throughout.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import selectors
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.pycache_prefix = os.environ.get("ISOSTER_PYCACHE_PREFIX", "/tmp/isoster_pycache")

from benchmarks.autoprof_env import resolve_autoprof_python  # noqa: E402
from benchmarks.harmonic_scale.adapters import (  # noqa: E402
    assert_rings_match_request,
    measure_isoster_fixed,
    measure_photutils_fixed,
)
from benchmarks.harmonic_scale.conventions import (  # noqa: E402
    raw_from_autoprof,
    raw_from_bender,
    rotate_raw_to_major_axis,
)
from benchmarks.timing.accuracy_thresholds import (  # noqa: E402
    CONTAMINATION,
    ENSEMBLE_REALIZATIONS,
    MAX_APERTURE_DISPLACEMENT_PX,
    MIN_COVERAGE_FRACTION,
    SCIENTIFIC_INPUT,
    TARGET_INTERVAL_R_E,
    accuracy_family_members,
    benchmark_host_mismatches,
    evaluate_accuracy_family,
    evaluate_geometry_accuracy_family,
    evaluate_systematic_accuracy_family,
    harmonic_absolute_error_limit_on_aperture,
    headline_eligible,
    ideal_sigma_by_family_member,
    ring_intensity_absolute_error_limit_on_aperture,
)
from benchmarks.timing.preflight import (  # noqa: E402
    competing_processes,
    load_average,
    observed_benchmark_host,
    thermal_warnings,
)
from benchmarks.timing.profile_evaluation import (  # noqa: E402
    interpolate_profile_to_evaluation_radii,
)
from benchmarks.timing.stage1_fixtures import (  # noqa: E402
    NOISE_ARMS,
    fixed_aperture_radii,
    render_stage1_fixture,
    stage1_fixtures,
)
from benchmarks.utils.sersic_model import (  # noqa: E402
    analytic_truth_on_aperture,
    aperture_displacement_error_px,
)

DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "benchmark_timing" / "stage2_calibration"
DEFAULT_BASELINE = REPO_ROOT / "outputs" / "benchmark_timing" / "baseline.json"
PILOT_SESSIONS = 3
MIN_BATCH_SECONDS = 0.1
TARGET_RELATIVE_HALF_WIDTH = 0.05
MAX_RETRIES = 3
MONITOR_INTERVAL_S = 10.0
ORDERS = (3, 4)
THREAD_LIMITS = SCIENTIFIC_INPUT["thread_limits"]


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


class AutoprofClient:
    """One persistent AutoProf interpreter for one calibration session."""

    def __init__(self, workspace: Path, python: str | None = None):
        workspace.mkdir(parents=True, exist_ok=True)
        self.log_handle = (workspace / "worker.stderr.log").open("w")
        worker = Path(__file__).with_name("autoprof_timing_worker.py")
        start = time.perf_counter()
        self.process = subprocess.Popen(
            [resolve_autoprof_python(python), str(worker)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log_handle,
            text=True,
            bufsize=1,
            cwd=REPO_ROOT,
        )
        ready_line = self.process.stdout.readline() if self.process.stdout else ""
        self.startup_s = time.perf_counter() - start
        self.environment = json.loads(ready_line)
        if self.environment.get("status") != "ready":
            raise RuntimeError(f"AutoProf worker did not become ready: {ready_line!r}")

    def request(self, job: dict) -> tuple[dict, float]:
        if not self.process.stdin or not self.process.stdout:
            raise RuntimeError("AutoProf worker pipes are unavailable")
        start = time.perf_counter()
        self.process.stdin.write(json.dumps(job, default=_json_default, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            if not selector.select(timeout=600):
                self.process.terminate()
                raise TimeoutError(f"AutoProf worker timed out for {job['request_id']}")
        response = json.loads(self.process.stdout.readline())
        elapsed = time.perf_counter() - start
        if response.get("status") != "ok":
            raise RuntimeError(f"AutoProf worker failed: {response.get('error')}\n{response.get('traceback', '')}")
        if response.get("request_id") != job["request_id"]:
            raise RuntimeError("AutoProf worker returned a mismatched request id")
        return response["result"], elapsed

    def close(self):
        if self.process.poll() is None and self.process.stdin and self.process.stdout:
            self.process.stdin.write('{"command":"shutdown"}\n')
            self.process.stdin.flush()
            self.process.stdout.readline()
            self.process.wait(timeout=10)
        self.log_handle.close()


def _ring_request(fixture: str) -> list[dict]:
    spec = stage1_fixtures()[fixture]
    center = spec["galaxy"]["center"]
    return [
        {
            "sma": sma,
            "x0": float(center[0]),
            "y0": float(center[1]),
            "eps": float(spec["reference_eps"]),
            "pa": float(spec["reference_pa"]),
        }
        for sma in fixed_aperture_radii(fixture)
    ]


def _fixed_rows(
    tool: str,
    image: np.ndarray,
    fixture: str,
    harmonics_enabled: bool,
    background_noise: float,
    client,
    workspace,
    request_id,
):
    request = _ring_request(fixture)
    orders = ORDERS if harmonics_enabled else ()
    start = time.perf_counter()
    if tool == "isoster":
        rows = measure_isoster_fixed(image, request, orders)
        fit_plus_harness_s = time.perf_counter() - start
        fit_only_s = fit_plus_harness_s
    elif tool == "photutils":
        rows = measure_photutils_fixed(image, request, orders)
        fit_plus_harness_s = time.perf_counter() - start
        fit_only_s = fit_plus_harness_s
    else:
        job = {
            "request_id": request_id,
            "scope": "fixed_aperture",
            "name": request_id,
            "output_dir": str(workspace / request_id),
            "rings": [
                {
                    "sma_pix": row["sma"],
                    "eps": row["eps"],
                    "pa_deg_astro": (math.degrees(row["pa"]) - 90.0) % 180.0,
                }
                for row in request
            ],
            "orders": list(orders),
            "pixel_scale": 1.0,
            "zeropoint": 22.5,
            "isoclip": True,
            "interpolate_start": 1000.0,
            "set_psf": None,
            "x0": request[0]["x0"],
            "y0": request[0]["y0"],
            "image": image,
            "extra_options": {
                "ap_set_background": 0.0,
                "ap_set_background_noise": float(background_noise),
            },
        }
        result, fit_plus_harness_s = client.request(job)
        rows = []
        for wanted, entry in zip(request, result["rows"]):
            measured_pa = math.radians((float(entry["pa_deg_astro"]) + 90.0) % 180.0)
            row = {
                "sma": float(entry["sma_pix"]),
                "x0": float(wanted["x0"]),
                "y0": float(wanted["y0"]),
                "eps": float(entry["eps"]),
                "pa": measured_pa,
                "mean_intensity": float(entry["b0"]),
                "status": "measured",
                "harmonic_sampling_mode": (
                    "line_interpolated"
                    if entry.get("interpolated") is True
                    else "line_nearest_pixel"
                    if entry.get("interpolated") is False
                    else "unknown"
                ),
            }
            for order in orders:
                s_sky, c_sky = raw_from_autoprof(entry[f"a{order}"], entry[f"b{order}"], entry["b0"])
                s_raw, c_raw = rotate_raw_to_major_axis(s_sky, c_sky, order=order, pa_rad=measured_pa)
                row[f"s{order}_raw"], row[f"c{order}_raw"] = s_raw, c_raw
            rows.append(row)
        fit_only_s = float(result["timing"]["pipeline_steps_s"].get("isophoteextract forced", math.nan))

    assert_rings_match_request(rows, request)
    normalized = []
    for row in rows:
        converted = {
            "sma": float(row["sma"]),
            "x0": float(row["x0"]),
            "y0": float(row["y0"]),
            "eps": float(row["eps"]),
            "pa": float(row["pa"]),
            "ring_mean": float(row["mean_intensity"]),
            "harmonic_conversion_valid": True,
        }
        for order in orders:
            converted[f"s{order}_raw_major"] = float(row[f"s{order}_raw"])
            converted[f"c{order}_raw_major"] = float(row[f"c{order}_raw"])
        if tool == "autoprof" and orders:
            converted["harmonic_conversion_valid"] = row.get("harmonic_sampling_mode") == "line_interpolated"
        normalized.append(converted)
    return normalized, fit_only_s, fit_plus_harness_s


def _isoster_end_to_end(image, spec, harmonics_enabled, output_path):
    from isoster import fit_image
    from isoster.config import IsosterConfig

    center = spec["galaxy"]["center"]
    config = IsosterConfig(
        x0=float(center[0]),
        y0=float(center[1]),
        eps=float(spec["reference_eps"]),
        pa=float(spec["reference_pa"]),
        sma0=max(2.0, 0.3 * float(spec["galaxy"]["R_e"])),
        compute_deviations=harmonics_enabled,
        harmonic_orders=list(ORDERS),
        use_eccentric_anomaly=False,
        debug=harmonics_enabled,
    )
    start = time.perf_counter()
    isophotes = fit_image(image, config=config)["isophotes"]
    rows = []
    for iso in isophotes:
        if not math.isfinite(float(iso["sma"])) or float(iso["sma"]) <= 0.0:
            continue
        row = {
            "sma": float(iso["sma"]),
            "x0": float(iso["x0"]),
            "y0": float(iso["y0"]),
            "eps": float(iso["eps"]),
            "pa": float(iso["pa"]),
            "ring_mean": float(iso["intens"]),
            "ndata": int(iso.get("ndata", 0)),
            "niter": int(iso.get("niter", 0)),
            "stop_code": int(iso.get("stop_code", 0)),
            "harmonic_conversion_valid": True,
        }
        if harmonics_enabled:
            gradient = float(iso.get("grad", math.nan))
            for order in ORDERS:
                s_raw, c_raw = raw_from_bender(iso[f"a{order}"], iso[f"b{order}"], iso["sma"], gradient)
                row[f"s{order}_raw_major"] = s_raw
                row[f"c{order}_raw_major"] = c_raw
            row["harmonic_conversion_valid"] = all(
                math.isfinite(row[f"{prefix}{order}_raw_major"]) for order in ORDERS for prefix in ("s", "c")
            )
        rows.append(row)
    output_path.write_text(json.dumps(rows, allow_nan=True, separators=(",", ":")))
    return rows, time.perf_counter() - start


def _photutils_end_to_end(image, spec, harmonics_enabled, output_path):
    from photutils.isophote import Ellipse, EllipseGeometry
    from photutils.isophote.harmonics import fit_upper_harmonic

    center = spec["galaxy"]["center"]
    geometry = EllipseGeometry(
        x0=float(center[0]),
        y0=float(center[1]),
        sma=max(2.0, 0.3 * float(spec["galaxy"]["R_e"])),
        eps=float(spec["reference_eps"]),
        pa=float(spec["reference_pa"]),
    )
    start = time.perf_counter()
    isolist = Ellipse(image, geometry).fit_image()
    rows = []
    for iso in isolist:
        if not math.isfinite(float(iso.sma)) or float(iso.sma) <= 0.0:
            continue
        row = {
            "sma": float(iso.sma),
            "x0": float(iso.x0),
            "y0": float(iso.y0),
            "eps": float(iso.eps),
            "pa": float(iso.pa),
            "ring_mean": float(iso.intens),
            "ndata": int(getattr(iso, "ndata", 0)),
            "niter": int(getattr(iso, "niter", 0)),
            "stop_code": int(getattr(iso, "stop_code", 0)),
            "harmonic_conversion_valid": True,
        }
        if harmonics_enabled:
            for order in ORDERS:
                coefficients, _ = fit_upper_harmonic(iso.sample.values[0], iso.sample.values[2], order)
                row[f"s{order}_raw_major"] = float(coefficients[1])
                row[f"c{order}_raw_major"] = float(coefficients[2])
            row["harmonic_conversion_valid"] = all(
                math.isfinite(row[f"{prefix}{order}_raw_major"]) for order in ORDERS for prefix in ("s", "c")
            )
        rows.append(row)
    output_path.write_text(json.dumps(rows, allow_nan=True, separators=(",", ":")))
    return rows, time.perf_counter() - start


def _end_to_end_rows(tool, image, fixture, harmonics_enabled, client, workspace, request_id):
    spec = stage1_fixtures()[fixture]
    output_dir = workspace / request_id
    output_dir.mkdir(parents=True, exist_ok=True)
    if tool == "isoster":
        rows, elapsed = _isoster_end_to_end(image, spec, harmonics_enabled, output_dir / "profile.json")
        return rows, elapsed, elapsed
    if tool == "photutils":
        rows, elapsed = _photutils_end_to_end(image, spec, harmonics_enabled, output_dir / "profile.json")
        return rows, elapsed, elapsed
    center = spec["galaxy"]["center"]
    job = {
        "request_id": request_id,
        "scope": "end_to_end",
        "name": request_id,
        "output_dir": str(output_dir),
        "orders": list(ORDERS if harmonics_enabled else ()),
        "pixel_scale": 1.0,
        "zeropoint": 22.5,
        "x0": float(center[0]),
        "y0": float(center[1]),
        "eps": float(spec["reference_eps"]),
        "pa": float(spec["reference_pa"]),
        "image": image,
    }
    result, total_s = client.request(job)
    return result["rows"], float(result["timing"]["pipeline_wall_s"]), total_s


def _coverage(fixture: str, rows: list[dict], scope: str) -> tuple[str, float]:
    if scope == "fixed_aperture":
        expected = fixed_aperture_radii(fixture)
        complete = len(rows) == len(expected) and all(
            math.isclose(row["sma"], radius, rel_tol=0.0, abs_tol=1e-6) for row, radius in zip(rows, expected)
        )
        return ("complete" if complete else "partial"), (1.0 if complete else len(rows) / len(expected))
    if not rows:
        return "partial", 0.0
    r_e = float(stage1_fixtures()[fixture]["galaxy"]["R_e"])
    low, high = (value * r_e for value in TARGET_INTERVAL_R_E)
    returned_low = min(float(row["sma"]) for row in rows)
    returned_high = max(float(row["sma"]) for row in rows)
    overlap = max(0.0, min(high, returned_high) - max(low, returned_low))
    fraction = overlap / (high - low)
    return ("complete" if fraction >= MIN_COVERAGE_FRACTION else "partial"), fraction


def _accuracy_inputs(fixture, rows, scope, harmonics_enabled, truth_meta):
    geometry_free = scope == "end_to_end"
    if geometry_free:
        r_e = float(stage1_fixtures()[fixture]["galaxy"]["R_e"])
        radii = [fraction * r_e for fraction in (0.3, 0.5, 1.0, 1.8, 3.0)]
        required = {"sma", "x0", "y0", "eps", "pa", "ring_mean"}
        if harmonics_enabled:
            required.update(f"{prefix}{order}_raw_major" for order in ORDERS for prefix in ("s", "c"))
        usable_rows = [
            row
            for row in rows
            if (not harmonics_enabled or row.get("harmonic_conversion_valid") is True)
            and required.issubset(row)
            and all(math.isfinite(float(row[name])) for name in required)
        ]
        evaluated = interpolate_profile_to_evaluation_radii(
            usable_rows, radii, orders=ORDERS if harmonics_enabled else ()
        )
    else:
        radii = fixed_aperture_radii(fixture)
        evaluated = rows
    families = accuracy_family_members(radii, harmonics_enabled=harmonics_enabled, geometry_free=geometry_free)
    residuals = {family: {member: [] for member in members} for family, members in families.items()}
    limits = {family: {} for family in families}
    spec = stage1_fixtures()[fixture]
    reference_center = spec["galaxy"]["center"]
    for row in evaluated:
        radius = float(row["sma"])
        radius_label = f"{radius:g}"
        truth = analytic_truth_on_aperture(truth_meta, row["x0"], row["y0"], radius, row["eps"], row["pa"], ORDERS)
        intensity_member = f"ring_mean@sma={radius_label}"
        residuals["intensity"][intensity_member].append(float(row["ring_mean"]) - truth[3]["mean_intensity"])
        limits["intensity"][intensity_member] = ring_intensity_absolute_error_limit_on_aperture(fixture, radius)
        if harmonics_enabled:
            for order in ORDERS:
                for prefix, truth_name in (("s", "s_raw"), ("c", "c_raw")):
                    member = f"{prefix}{order}_raw_major@sma={radius_label}"
                    residuals["harmonic"][member].append(
                        float(row[f"{prefix}{order}_raw_major"]) - truth[order][truth_name]
                    )
                    limits["harmonic"][member] = harmonic_absolute_error_limit_on_aperture(fixture, radius)
        if geometry_free:
            member = f"aperture_displacement@sma={radius_label}"
            reference = {
                "x0": float(reference_center[0]),
                "y0": float(reference_center[1]),
                "sma": radius,
                "eps": float(spec["reference_eps"]),
                "pa": float(spec["reference_pa"]),
            }
            residuals["geometry"][member].append(aperture_displacement_error_px(reference, row))
            limits["geometry"][member] = MAX_APERTURE_DISPLACEMENT_PX
    return residuals, limits


def _base_record(scope, tool, fixture, noise_arm, harmonics_enabled, session_index, realization_index):
    settings = {
        "isoster": {
            "use_eccentric_anomaly": False,
            "sigma_clipping": "IsosterConfig defaults",
            "interpolation": "isoster native",
        },
        "photutils": {
            "sigma_clipping": "EllipseSample defaults",
            "interpolation": "photutils native",
        },
        "autoprof": {
            "ap_isoclip": True,
            "ap_iso_interpolate_start": 1000.0,
            "ap_isoband_fixed": True,
            "ap_isoband_width": 0.1,
        },
    }[tool]
    return {
        "scope": scope,
        "tool": tool,
        "fixture": fixture,
        "noise_arm": noise_arm,
        "harmonics_enabled": harmonics_enabled,
        "geometry_free": scope == "end_to_end",
        "session_index": session_index,
        "realization_index": realization_index,
        "settings": {
            **settings,
            "background": "supplied zero" if scope == "fixed_aperture" else "tool-native handling",
            "output_writing_timed": scope == "end_to_end",
            "harmonic_orders": list(ORDERS if harmonics_enabled else ()),
        },
        "execution_status": "failed",
        "coverage_status": "partial",
        "harmonic_accuracy_status": "fail" if harmonics_enabled else "not_applicable",
        "intensity_accuracy_status": "fail",
        "geometry_accuracy_status": "fail" if scope == "end_to_end" else "not_applicable",
        "contamination_status": "clean",
        "headline_eligible": False,
    }


def _run_session(session_index: int, repetitions: int, output_path: Path, autoprof_python: str | None):
    workspace = output_path.parent / f"session_{session_index:02d}_work"
    client = AutoprofClient(workspace / "autoprof_worker", autoprof_python)
    records = []
    try:
        fixtures = stage1_fixtures()
        for scope in ("fixed_aperture", "end_to_end"):
            for fixture, spec in fixtures.items():
                if scope == "fixed_aperture" and spec["scope"] == "end_to_end":
                    continue
                for noise_arm in NOISE_ARMS:
                    for harmonics_enabled in (False, True):
                        for realization_index in range(repetitions):
                            image, _, metadata = render_stage1_fixture(
                                fixture,
                                noise_arm,
                                stage="calibration",
                                realization_index=0 if noise_arm == "noiseless" else realization_index,
                            )
                            tools = ["isoster", "photutils", "autoprof"]
                            random.Random(20260824 + session_index * 1000 + realization_index).shuffle(tools)
                            for tool in tools:
                                record = _base_record(
                                    scope,
                                    tool,
                                    fixture,
                                    noise_arm,
                                    harmonics_enabled,
                                    session_index,
                                    realization_index,
                                )
                                request_id = (
                                    f"s{session_index}_{scope}_{tool}_{fixture}_{noise_arm}_"
                                    f"h{int(harmonics_enabled)}_r{realization_index}"
                                )
                                failed_attempt_start = time.perf_counter()
                                try:
                                    if scope == "fixed_aperture":
                                        rows, fit_only_s, total_s = _fixed_rows(
                                            tool,
                                            image,
                                            fixture,
                                            harmonics_enabled,
                                            float(metadata["noise_sigma"]),
                                            client,
                                            workspace,
                                            request_id,
                                        )
                                    else:
                                        rows, fit_only_s, total_s = _end_to_end_rows(
                                            tool,
                                            image,
                                            fixture,
                                            harmonics_enabled,
                                            client,
                                            workspace,
                                            request_id,
                                        )
                                    if not rows:
                                        raise RuntimeError("tool returned no profile")
                                    coverage_status, coverage_fraction = _coverage(fixture, rows, scope)
                                    record.update(
                                        {
                                            "execution_status": "ok",
                                            "coverage_status": coverage_status,
                                            "coverage_fraction": coverage_fraction,
                                            "fit_only_s": fit_only_s,
                                            "fit_plus_harness_s": total_s,
                                            "harness_s": total_s - fit_only_s,
                                            "ring_count": len(rows),
                                            "fit_only_s_per_ring": fit_only_s / len(rows),
                                            "fit_plus_harness_s_per_ring": total_s / len(rows),
                                            "profile": rows,
                                        }
                                    )
                                    try:
                                        residuals, limits = _accuracy_inputs(
                                            fixture, rows, scope, harmonics_enabled, metadata["truth"]
                                        )
                                        record["accuracy_residuals"] = residuals
                                        record["systematic_limits"] = limits
                                    except (KeyError, TypeError, ValueError) as error:
                                        record["accuracy_error"] = f"{type(error).__name__}: {error}"
                                except Exception as error:  # noqa: BLE001
                                    record["error"] = f"{type(error).__name__}: {error}"
                                    record["failed_wall_s"] = time.perf_counter() - failed_attempt_start
                                records.append(record)
                                output_path.write_text(
                                    json.dumps(
                                        {
                                            "session_index": session_index,
                                            "thread_limits": {name: os.environ.get(name) for name in THREAD_LIMITS},
                                            "autoprof_startup_s": client.startup_s,
                                            "autoprof_environment": client.environment,
                                            "records": records,
                                        },
                                        indent=2,
                                        default=_json_default,
                                        allow_nan=True,
                                    )
                                )
    finally:
        client.close()


def _arm_key(record):
    return tuple(record[name] for name in ("scope", "tool", "fixture", "noise_arm", "harmonics_enabled"))


def _set_accuracy_outcomes(records):
    grouped = {}
    for record in records:
        grouped.setdefault((*_arm_key(record), record["session_index"]), []).append(record)
    failures = []
    for key, arm_records in grouped.items():
        harmonics_enabled = bool(key[4])
        geometry_free = key[0] == "end_to_end"
        noise_arm = key[3]
        successful = [record for record in arm_records if record["execution_status"] == "ok"]
        verdicts = {}
        try:
            if noise_arm == "noiseless":
                source = successful[0]
                for family, by_member in source["accuracy_residuals"].items():
                    one_each = {member: values[0] for member, values in by_member.items()}
                    if family == "geometry":
                        passed = all(value <= MAX_APERTURE_DISPLACEMENT_PX for value in one_each.values())
                    else:
                        passed = evaluate_systematic_accuracy_family(one_each, source["systematic_limits"][family])[
                            "family_passed"
                        ]
                    verdicts[family] = bool(passed)
            else:
                if len(successful) != ENSEMBLE_REALIZATIONS:
                    raise ValueError(f"requires {ENSEMBLE_REALIZATIONS} successful realizations")
                members = successful[0]["accuracy_residuals"]
                for family in members:
                    combined = {
                        member: [record["accuracy_residuals"][family][member][0] for record in successful]
                        for member in members[family]
                    }
                    if family == "geometry":
                        passed = evaluate_geometry_accuracy_family(combined)["family_passed"]
                    else:
                        passed = evaluate_accuracy_family(
                            combined, ideal_sigma_by_family_member(key[2], list(combined))
                        )["family_passed"]
                    verdicts[family] = bool(passed)
        except (IndexError, KeyError, ValueError) as error:
            failures.append({"arm": key, "error": str(error)})

        for record in arm_records:
            record["harmonic_accuracy_status"] = (
                ("pass" if verdicts.get("harmonic") else "fail") if harmonics_enabled else "not_applicable"
            )
            record["intensity_accuracy_status"] = "pass" if verdicts.get("intensity") else "fail"
            record["geometry_accuracy_status"] = (
                ("pass" if verdicts.get("geometry") else "fail") if geometry_free else "not_applicable"
            )
            record["headline_eligible"] = headline_eligible(
                record,
                harmonics_enabled=harmonics_enabled,
                geometry_free=geometry_free,
            )
    return failures


def _summary(values):
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return {"raw_s": [], "median_s": None, "iqr_s": None}
    if len(ordered) >= 4:
        q1, _, q3 = statistics.quantiles(ordered, n=4)
    else:
        q1, q3 = ordered[0], ordered[-1]
    return {"raw_s": ordered, "median_s": statistics.median(ordered), "iqr_s": [q1, q3]}


def _recommendations(records):
    grouped = {}
    for record in records:
        if record["execution_status"] == "ok":
            grouped.setdefault(_arm_key(record), []).append(record)
    output = {}
    for key, arm_records in sorted(grouped.items(), key=lambda item: str(item[0])):
        total = _summary(record["fit_plus_harness_s"] for record in arm_records)
        fit = _summary(record["fit_only_s"] for record in arm_records)
        median_s = total["median_s"]
        calls_per_batch = max(1, math.ceil(MIN_BATCH_SECONDS / median_s)) if median_s else None
        session_medians = []
        for session_index in sorted({record["session_index"] for record in arm_records}):
            session_medians.append(
                statistics.median(
                    record["fit_plus_harness_s"] for record in arm_records if record["session_index"] == session_index
                )
            )
        relative_spread = (
            (max(session_medians) - min(session_medians)) / (2.0 * statistics.median(session_medians))
            if session_medians and statistics.median(session_medians)
            else None
        )
        recommended_sessions = (
            max(
                PILOT_SESSIONS,
                math.ceil(PILOT_SESSIONS * (relative_spread / TARGET_RELATIVE_HALF_WIDTH) ** 2),
            )
            if relative_spread is not None
            else None
        )
        label = "|".join(str(value) for value in key)
        output[label] = {
            "fit_only": fit,
            "fit_plus_harness": total,
            "calls_per_batch": calls_per_batch,
            "observed_session_medians_s": session_medians,
            "observed_relative_half_range": relative_spread,
            "recommended_sessions": recommended_sessions,
        }
    return output


CPU_WORKLOAD = """
import json, statistics, time
samples=[]
for _ in range(7):
    start=time.perf_counter()
    value=0
    for index in range(2000000):
        value=(value + index * index) % 1000000007
    samples.append(time.perf_counter()-start)
print(json.dumps({"raw_s": samples, "median_s": statistics.median(samples), "checksum": value}))
"""


def _interpreter_calibration(autoprof_python):
    results = {}
    for label, python in (("project", sys.executable), ("autoprof", resolve_autoprof_python(autoprof_python))):
        completed = subprocess.run([python, "-c", CPU_WORKLOAD], capture_output=True, text=True, check=True)
        results[label] = json.loads(completed.stdout)
        results[label]["python"] = python
    results["autoprof_to_project_ratio"] = results["autoprof"]["median_s"] / results["project"]["median_s"]
    return results


def _indicator_sample(limit):
    load = load_average()
    thermal = thermal_warnings()
    processes = competing_processes()
    return {
        "time": time.time(),
        "load": load,
        "load_limit_exceeded": load > limit,
        "thermal": thermal,
        "processes": processes,
        "contaminated": bool(thermal["warnings_recorded"]) or bool(processes),
    }


def _assess_sample(sample, consecutive_load_samples):
    if sample["contaminated"]:
        return 0, True
    consecutive_load_samples = consecutive_load_samples + 1 if sample["load_limit_exceeded"] else 0
    return (
        consecutive_load_samples,
        consecutive_load_samples >= int(CONTAMINATION["in_session_consecutive_load_samples"]),
    )


def _session_environment():
    environment = os.environ.copy()
    environment.update(THREAD_LIMITS)
    return environment


def _wait_for_clean_retry(trace, limit):
    while trace[-1]["contaminated"] or trace[-1]["load_limit_exceeded"]:
        print(
            f"[stage2] waiting for clean indicators before retry: load {trace[-1]['load']:.2f} (limit {limit:.2f})",
            flush=True,
        )
        time.sleep(MONITOR_INTERVAL_S)
        trace.append(_indicator_sample(limit))


def _run_monitored_session(command, trace, limit):
    process = subprocess.Popen(command, cwd=REPO_ROOT, start_new_session=True, env=_session_environment())
    next_sample = 0.0
    consecutive_load_samples = 0
    while process.poll() is None:
        now = time.monotonic()
        if now >= next_sample:
            sample = _indicator_sample(limit)
            consecutive_load_samples, sample["contaminated"] = _assess_sample(sample, consecutive_load_samples)
            trace.append(sample)
            if sample["contaminated"]:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=30)
                return "contaminated"
            next_sample = now + MONITOR_INTERVAL_S
        time.sleep(0.25)
    sample = _indicator_sample(limit)
    _, sample["contaminated"] = _assess_sample(sample, 0)
    trace.append(sample)
    if sample["contaminated"]:
        return "contaminated"
    return "clean" if process.returncode == 0 else "failed"


def _validate_baseline(path: Path):
    baseline = json.loads(path.read_text())
    problems = benchmark_host_mismatches(baseline.get("host", {}))
    if float(baseline["median"]) > float(CONTAMINATION["baseline_median_max"]):
        problems.append("baseline median exceeds the frozen ceiling")
    if baseline.get("processes"):
        problems.append("baseline recorded competing processes")
    if baseline.get("thermal", {}).get("warnings_recorded"):
        problems.append("baseline recorded a thermal warning")
    if problems:
        raise RuntimeError("invalid baseline: " + "; ".join(problems))
    return baseline


def _mark_attempt_contaminated(attempt_dir: Path):
    for path in sorted(attempt_dir.glob("session_*.json")):
        payload = json.loads(path.read_text())
        for record in payload.get("records", []):
            record["contamination_status"] = "contaminated"
            record["headline_eligible"] = headline_eligible(
                record,
                harmonics_enabled=bool(record["harmonics_enabled"]),
                geometry_free=bool(record["geometry_free"]),
            )
        path.write_text(json.dumps(payload, indent=2, default=_json_default, allow_nan=True))


def _run_parent(args):
    if args.sessions < 2:
        raise SystemExit("Stage 2 requires several sessions; use --sessions 2 or more")
    if args.repetitions != ENSEMBLE_REALIZATIONS:
        raise SystemExit(f"Stage 1 accuracy requires exactly --repetitions {ENSEMBLE_REALIZATIONS}")
    baseline = _validate_baseline(args.baseline)
    args.output.mkdir(parents=True, exist_ok=True)
    attempts = []
    accepted = None
    limit = float(baseline["median"]) + float(CONTAMINATION["in_session_excess_max"])
    for attempt_index in range(MAX_RETRIES + 1):
        attempt_dir = args.output / f"attempt_{attempt_index + 1:02d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        trace = []
        attempt = {"attempt": attempt_index + 1, "trace": trace, "status": "running"}
        attempts.append(attempt)
        trace.append(_indicator_sample(limit))
        if attempt_index:
            _wait_for_clean_retry(trace, limit)
        attempt_status = (
            "clean" if not trace[-1]["contaminated"] and not trace[-1]["load_limit_exceeded"] else "contaminated"
        )
        interpreter = None
        if attempt_status == "clean":
            try:
                interpreter = _interpreter_calibration(args.autoprof_python)
            except Exception as error:  # noqa: BLE001
                attempt_status = "failed"
                attempt["error"] = f"interpreter calibration failed: {type(error).__name__}: {error}"
        if attempt_status == "clean":
            trace.append(_indicator_sample(limit))
            if trace[-1]["contaminated"] or trace[-1]["load_limit_exceeded"]:
                attempt_status = "contaminated"
        for session_index in range(args.sessions):
            if attempt_status != "clean":
                break
            session_output = attempt_dir / f"session_{session_index:02d}.json"
            command = [
                sys.executable,
                str(Path(__file__)),
                "--session-index",
                str(session_index),
                "--repetitions",
                str(args.repetitions),
                "--session-output",
                str(session_output),
            ]
            if args.autoprof_python:
                command.extend(["--autoprof-python", args.autoprof_python])
            session_status = _run_monitored_session(command, trace, limit)
            if session_status != "clean":
                attempt_status = session_status
                break
        attempt["status"] = attempt_status
        attempt["interpreter"] = interpreter
        if attempt_status == "contaminated":
            _mark_attempt_contaminated(attempt_dir)
        (attempt_dir / "attempt.json").write_text(json.dumps(attempt, indent=2))
        if attempt_status == "clean":
            accepted = attempt_dir
            break
        if attempt_status == "failed":
            (args.output / "stage2_calibration_summary.json").write_text(json.dumps({"attempts": attempts}, indent=2))
            raise SystemExit("Stage 2 session failed; inspect the retained attempt before retrying")
    if accepted is None:
        (args.output / "stage2_calibration_summary.json").write_text(json.dumps({"attempts": attempts}, indent=2))
        raise SystemExit("machine remained contaminated after three retries")

    records = []
    sessions = []
    for path in sorted(accepted.glob("session_*.json")):
        payload = json.loads(path.read_text())
        sessions.append({key: value for key, value in payload.items() if key != "records"})
        records.extend(payload["records"])
    failures = _set_accuracy_outcomes(records)
    summary = {
        "stage": "Part B Stage 2 timing calibration; not the benchmark campaign",
        "note": "It is not a survey; it trades coverage for repeated measurements and uncertainty estimates.",
        "host": observed_benchmark_host(),
        "baseline": baseline,
        "attempts": attempts,
        "accepted_attempt": str(accepted),
        "sessions": sessions,
        "records": records,
        "accuracy_evaluation_failures": failures,
        "recommendations_for_stage3": _recommendations(records),
    }
    output_path = args.output / "stage2_calibration_summary.json"
    output_path.write_text(json.dumps(summary, indent=2, default=_json_default, allow_nan=True))
    accuracy_failed = any(
        record["execution_status"] == "ok"
        and (
            record["harmonic_accuracy_status"] == "fail"
            or record["intensity_accuracy_status"] == "fail"
            or record["geometry_accuracy_status"] == "fail"
        )
        for record in records
    )
    print(f"[stage2] retained {len(records)} timings in {output_path}")
    if accuracy_failed or failures:
        raise SystemExit("Stage 2 accuracy check failed; investigate the retained calibration before Stage 3")
    print("[stage2] calibration passed; review and commit the Stage 3 timing parameters")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sessions", type=int, default=PILOT_SESSIONS)
    parser.add_argument("--repetitions", type=int, default=ENSEMBLE_REALIZATIONS)
    parser.add_argument("--autoprof-python")
    parser.add_argument("--session-index", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--session-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.session_index is not None:
        if args.session_output is None:
            raise SystemExit("--session-output is required with --session-index")
        _run_session(args.session_index, args.repetitions, args.session_output, args.autoprof_python)
        return
    _run_parent(args)


if __name__ == "__main__":
    main()
