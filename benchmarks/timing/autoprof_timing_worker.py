#!/usr/bin/env python
"""Persistent AutoProf worker for the Stage 2 calibration.

The process reads one JSON job per line and writes one JSON result per line.
Each job gets a new pipeline and output directory, while the expensive Python
and AutoProf imports stay resident. Pipeline-step timers provide the fit-only
measurement; the caller measures the full JSON/FITS/IPC round trip.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import sys
import time
import traceback

import numpy as np
from astropy.io import fits

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
sys.pycache_prefix = os.environ.get("ISOSTER_PYCACHE_PREFIX", "/tmp/isoster_pycache")

from benchmarks.harmonic_scale.autoprof_worker import (  # noqa: E402, I001
    _attribute_extractions,
    _install_sampling_mode_probe,
    run_job as run_fixed_job,
)
from benchmarks.harmonic_scale.conventions import (  # noqa: E402
    raw_from_autoprof,
    rotate_raw_to_major_axis,
)

END_TO_END_PIPELINE_STEPS = [
    "background",
    "psf",
    "center",
    "isophoteinit",
    "isophotefit",
    "isophoteextract",
    "checkfit",
    "writeprof",
]


def _center_from_aux(path):
    center = {"x0": float("nan"), "y0": float("nan")}
    if not os.path.exists(path):
        return center
    with open(path) as handle:
        for line in handle:
            if line.strip().startswith("center x:"):
                try:
                    parts = line.split(",")
                    center["x0"] = float(parts[0].split(":", 1)[1].strip().split()[0])
                    center["y0"] = float(parts[1].split(":", 1)[1].strip().split()[0])
                except (IndexError, ValueError):
                    pass
                break
    return center


def _run_end_to_end_job(job):
    output_dir = job["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    name = job["name"]
    pixel_scale = float(job.get("pixel_scale", 1.0))
    orders = [int(order) for order in job["orders"]]

    fits_start = time.perf_counter()
    image_path = os.path.join(output_dir, f"{name}.fits")
    fits.PrimaryHDU(np.asarray(job["image"], dtype=np.float64)).writeto(image_path, overwrite=True)
    fits_write_s = time.perf_counter() - fits_start
    events = _install_sampling_mode_probe()

    from autoprof.Pipeline import Isophote_Pipeline

    pipeline = Isophote_Pipeline(loggername=os.path.join(output_dir, f"{name}.log"))
    pipeline.UpdatePipeline(new_pipeline_steps=list(END_TO_END_PIPELINE_STEPS))
    options = {
        "ap_image_file": image_path,
        "ap_name": name,
        "ap_pixscale": pixel_scale,
        "ap_zeropoint": float(job.get("zeropoint", 22.5)),
        "ap_doplot": False,
        "ap_saveto": output_dir + os.sep,
        "ap_fluxunits": "intensity",
        "ap_iso_measurecoefs": orders or None,
        "ap_isoclip": True,
        "ap_isoband_fixed": True,
        "ap_isoband_width": 0.1,
        "ap_iso_interpolate_start": 1000.0,
        "ap_guess_center": {"x": float(job["x0"]), "y": float(job["y0"])},
        "ap_isoinit_pa_set": math.degrees(float(job["pa"])) - 90.0,
        "ap_isoinit_ellip_set": float(job["eps"]),
    }

    pipeline_start = time.perf_counter()
    outcome = pipeline.Process_Image(options=options)
    pipeline_wall_s = time.perf_counter() - pipeline_start
    if outcome == 1:
        raise RuntimeError(f"AutoProf reported failure for {name}")

    parse_start = time.perf_counter()
    profile_path = os.path.join(output_dir, f"{name}.prof")
    data = np.atleast_1d(np.genfromtxt(profile_path, delimiter=",", names=True, skip_header=1))
    columns = set(data.dtype.names or ())
    center = _center_from_aux(os.path.join(output_dir, f"{name}.aux"))
    native_rows = []
    for index in range(len(data)):
        sma = float(data["R"][index]) / pixel_scale
        ring_mean = float(data["b0"][index]) if "b0" in columns else float(data["I"][index])
        if not (math.isfinite(sma) and sma > 0.0 and math.isfinite(ring_mean)):
            continue
        native_rows.append(
            {
                "source_index": index,
                "sma_pix": sma,
                "eps": float(data["ellip"][index]),
                "pa_deg_astro": float(data["pa"][index]),
                "b0": ring_mean,
                "pixels": int(data["pixels"][index]) if "pixels" in columns else None,
            }
        )

    interpolation = _attribute_extractions(events["extractions"], native_rows, pixel_scale)
    rows = []
    for entry, ring_mode in zip(native_rows, interpolation["per_ring"]):
        index = entry["source_index"]
        pa = math.radians((entry["pa_deg_astro"] + 90.0) % 180.0)
        row = {
            "sma": entry["sma_pix"],
            "x0": center["x0"],
            "y0": center["y0"],
            "eps": entry["eps"],
            "pa": pa,
            "ring_mean": entry["b0"],
            "ndata": entry["pixels"],
            "niter": None,
            "harmonic_conversion_valid": not orders or ring_mode["interpolated"] is True,
            "harmonic_sampling_mode": (
                "line_interpolated"
                if ring_mode["interpolated"] is True
                else "line_nearest_pixel"
                if ring_mode["interpolated"] is False
                else "unknown"
            ),
        }
        for order in orders:
            native_a = float(data[f"a{order}"][index])
            native_b = float(data[f"b{order}"][index])
            s_sky, c_sky = raw_from_autoprof(native_a, native_b, entry["b0"])
            s_major, c_major = rotate_raw_to_major_axis(s_sky, c_sky, order=order, pa_rad=pa)
            row[f"s{order}_raw_major"] = s_major
            row[f"c{order}_raw_major"] = c_major
        if orders:
            row["harmonic_conversion_valid"] = row["harmonic_conversion_valid"] and all(
                math.isfinite(row[f"{prefix}{order}_raw_major"]) for order in orders for prefix in ("s", "c")
            )
        rows.append(row)

    return {
        "rows": rows,
        "sampling_mode": {
            "all_rings_line_sampled": events["band_sampling_calls"] == 0,
            "attribution_ok": interpolation["attribution_ok"],
            "all_rings_interpolated": interpolation["all_interpolated"],
        },
        "timing": {
            "fits_write_s": fits_write_s,
            "pipeline_wall_s": pipeline_wall_s,
            "pipeline_steps_s": {key: float(value) for key, value in outcome.items()},
            "profile_parse_s": time.perf_counter() - parse_start,
        },
    }


def _reply(payload):
    sys.stdout.write(json.dumps(payload, allow_nan=True, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def main():
    import autoprof

    _reply(
        {
            "status": "ready",
            "python": sys.version.split()[0],
            "autoprof": getattr(autoprof, "__version__", "unknown"),
            "numpy": np.__version__,
        }
    )
    for line in sys.stdin:
        try:
            job = json.loads(line)
            if job.get("command") == "shutdown":
                _reply({"status": "stopped"})
                return
            with contextlib.redirect_stdout(sys.stderr):
                result = run_fixed_job(job) if job["scope"] == "fixed_aperture" else _run_end_to_end_job(job)
            _reply({"status": "ok", "request_id": job["request_id"], "result": result})
        except Exception as error:  # noqa: BLE001
            _reply(
                {
                    "status": "error",
                    "request_id": job.get("request_id") if "job" in locals() else None,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
            )


if __name__ == "__main__":
    main()
