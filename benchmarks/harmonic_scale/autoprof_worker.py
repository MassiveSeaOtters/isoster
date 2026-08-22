"""Runs inside the AutoProf venv. Never imported by the main environment.

Reads a JSON job, drives AutoProf's forced pipeline over an explicit list of
rings, and writes a JSON result carrying the native coefficients plus enough
provenance to defend them later.

Four things here exist because AutoProf does not behave the way its option
names suggest. Each is load-bearing; none is defensive tidiness.

1. **``ap_process_mode`` is inert.** It selects the forced pipeline only
   through ``Process_ConfigFile`` (``Pipeline.py:322``). Called through
   ``Process_Image``, as here, the option does nothing at all, so the step list
   is installed explicitly with ``UpdatePipeline`` and the realized sequence is
   archived.

2. **The companion ``.aux`` is required even with ``ap_set_center``.**
   ``ap_set_center`` short-circuits ``Center_Forced`` (``Center.py:90``), but
   the next step, ``Isophote_Init_Forced``, opens
   ``ap_forcing_profile[:-4] + "aux"`` unconditionally
   (``Isophote_Initialize.py:74``). We therefore write a minimal one. Its
   *global* ellipticity and position angle are initialization values only --
   the per-ring extraction geometry comes from the CSV rows.

3. **Isophotal-band sampling must be excluded deterministically.** The mode is
   chosen by ``medflux > noise * ap_isoband_start or isobandwidth < 0.5``
   (``Isophote_Extract.py:95``). The first clause fails for a ring with
   non-positive median flux, so raising S/N is not a guarantee; the second is
   an ``or`` and is. ``ap_isoband_fixed=True`` with ``ap_isoband_width < 0.5``
   makes line sampling unconditional.

4. **That has to be verified, not trusted.** The ``.prof`` output carries no
   column saying which mode a ring used, so the check is impossible from stock
   output. We wrap ``_iso_between`` and record every call, distinguishing band
   sampling (a non-zero inner radius) from the curve-of-growth sum (which
   always passes zero). Zero band-sampling calls is the assertion.
"""

import hashlib
import importlib
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
from astropy.io import fits  # noqa: E402

#: Installed exactly, in this order. AutoProf's own forced sequence
#: (``Pipeline.py:322-332``); notably it contains no ``isophotefit forced``,
#: because the forced extractor reads geometry straight from the profile.
FORCED_PIPELINE_STEPS = [
    "background",
    "psf",
    "center forced",
    "isophoteinit forced",
    "isophoteextract forced",
    "writeprof",
]


def _write_forcing_files(directory, name, rings, pixel_scale):
    """Write the forcing CSV and the minimal companion .aux AutoProf needs.

    Units follow ``Isophote_Extract_Forced``: ``R`` is divided by
    ``ap_pixscale`` on read, so it is written in **arcsec**; ``pa`` goes
    through ``PA_shift_convention(..., deg=True)``, so it is written in
    astronomical degrees.
    """
    profile_path = os.path.join(directory, f"{name}_force.prof")
    aux_path = profile_path[:-4] + "aux"

    with open(profile_path, "w") as handle:
        handle.write("R,ellip,pa\n")
        for ring in rings:
            handle.write(
                "%.10f,%.10f,%.10f\n"
                % (
                    ring["sma_pix"] * pixel_scale,
                    ring["eps"],
                    ring["pa_deg_astro"],
                )
            )

    # Parsed by string position in Isophote_Initialize.py:74-95, so the shape
    # of this line matters: first ":" then "+-" then "," then "pa:" ... "deg"
    # then "size:" ... "pix". Values are initialization only.
    first = rings[0]
    with open(aux_path, "w") as handle:
        handle.write(
            "global ellipticity: %.3f +- %.3f, pa: %.3f +- %.3f deg, size: %f pix\n"
            % (first["eps"], 0.01, first["pa_deg_astro"], 1.0, first["sma_pix"])
        )
    return profile_path, aux_path


def _install_sampling_mode_probe():
    """Wrap ``_iso_between`` so band-sampling events can be counted.

    ``_iso_between`` serves two callers in ``_Generate_Profile``: the
    curve-of-growth sum, which always passes an inner radius of 0, and
    isophotal-band sampling, which passes ``R - width``. Only the latter
    changes the estimand, so the inner radius is the discriminator.
    """
    # importlib, not `from autoprof.pipeline_steps import Isophote_Extract`:
    # the package re-exports a *function* under the module's own name, so the
    # plain import binds the function and the module is unreachable.
    extract_module = importlib.import_module("autoprof.pipeline_steps.Isophote_Extract")

    original = extract_module._iso_between
    events = {"band_sampling_calls": 0, "total_calls": 0}

    def probe(image, low, high, *args, **kwargs):
        events["total_calls"] += 1
        if float(low) != 0.0:
            events["band_sampling_calls"] += 1
        return original(image, low, high, *args, **kwargs)

    extract_module._iso_between = probe
    return events


def _source_digest():
    """SHA-256 of the file whose four-line expression this whole study is about."""
    extract_module = importlib.import_module("autoprof.pipeline_steps.Isophote_Extract")

    with open(extract_module.__file__, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def main(job_path):
    with open(job_path) as handle:
        job = json.load(handle)

    output_dir = job["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    name = job["name"]
    rings = job["rings"]
    orders = job["orders"]

    image = np.asarray(job["image"], dtype=np.float64) if "image" in job else None
    image_path = job.get("image_path")
    if image is not None:
        image_path = os.path.join(output_dir, f"{name}.fits")
        fits.PrimaryHDU(image).writeto(image_path, overwrite=True)

    profile_path, aux_path = _write_forcing_files(output_dir, name, rings, job["pixel_scale"])
    events = _install_sampling_mode_probe()

    from autoprof.Pipeline import Isophote_Pipeline

    pipeline = Isophote_Pipeline(loggername=os.path.join(output_dir, f"{name}.log"))
    # ap_process_mode would be ignored on this path; install the steps directly.
    pipeline.UpdatePipeline(new_pipeline_steps=list(FORCED_PIPELINE_STEPS))

    options = {
        "ap_image_file": image_path,
        "ap_name": name,
        "ap_pixscale": job["pixel_scale"],
        "ap_zeropoint": job["zeropoint"],
        "ap_doplot": False,
        "ap_saveto": output_dir + os.sep,
        "ap_fluxunits": "intensity",
        "ap_forcing_profile": profile_path,
        "ap_set_center": {"x": job["x0"], "y": job["y0"]},
        "ap_iso_measurecoefs": list(orders),
        "ap_isoclip": bool(job["isoclip"]),
        # Force line sampling unconditionally: see module docstring, point 3.
        "ap_isoband_fixed": True,
        "ap_isoband_width": 0.1,
    }
    options.update(job.get("extra_options", {}))

    outcome = pipeline.Process_Image(options=options)
    if outcome == 1:
        raise SystemExit(f"AutoProf reported failure for {name}")

    data = np.genfromtxt(os.path.join(output_dir, f"{name}.prof"), delimiter=",", names=True, skip_header=1)
    data = np.atleast_1d(data)

    columns = list(data.dtype.names)
    rows = []
    for index in range(len(data)):
        entry = {
            "sma_arcsec": float(data["R"][index]),
            "sma_pix": float(data["R"][index]) / job["pixel_scale"],
            "eps": float(data["ellip"][index]),
            "pa_deg_astro": float(data["pa"][index]),
            "b0": float(data["b0"][index]) if "b0" in columns else float("nan"),
            "a0": float(data["a0"][index]) if "a0" in columns else float("nan"),
            "pixels": int(data["pixels"][index]) if "pixels" in columns else -1,
        }
        for order in orders:
            for prefix in ("a", "b"):
                key = f"{prefix}{order}"
                entry[key] = float(data[key][index]) if key in columns else float("nan")
        rows.append(entry)

    import autoprof

    result = {
        "rows": rows,
        "columns": columns,
        "provenance": {
            "autoprof_version": getattr(autoprof, "__version__", "unknown"),
            "isophote_extract_sha256": _source_digest(),
            "realized_pipeline_steps": list(FORCED_PIPELINE_STEPS),
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
            "forcing_profile": profile_path,
            "forcing_aux": aux_path,
            "isoclip": bool(job["isoclip"]),
            "ap_isoband_fixed": True,
            "ap_isoband_width": 0.1,
        },
        "sampling_mode": {
            "band_sampling_calls": events["band_sampling_calls"],
            "iso_between_total_calls": events["total_calls"],
            "all_rings_line_sampled": events["band_sampling_calls"] == 0,
        },
    }
    with open(job["result_path"], "w") as handle:
        json.dump(result, handle)


if __name__ == "__main__":
    main(sys.argv[1])
