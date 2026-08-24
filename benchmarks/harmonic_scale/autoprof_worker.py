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

5. **The *other* sampling mode -- interpolated versus rounded to the nearest
   pixel -- is the largest effect in the whole study, and it is likewise
   absent from stock output.** ``_iso_extract`` samples with Lanczos where
   ``Rlim < rad_interp`` and by ``np.rint`` otherwise
   (``SharedFunctions.py:653``), with
   ``rad_interp = ap_iso_interpolate_start * results["psf fwhm"]``
   (``Isophote_Extract.py:110-116``). Note what ``results["psf fwhm"]``
   actually is on this pipeline: the ``psf`` step resolves to ``PSF_Assumed``
   (``Pipeline.py:53``), which **hardcodes 4.0 px** unless ``ap_set_psf`` or
   ``ap_guess_psf`` is given -- it measures nothing. Measuring requires
   selecting ``psf starfind`` or another variant explicitly.

   So the threshold *is* predictable here, at ``5 x 4.0 = 20`` px. We observe
   AutoProf's branch anyway, for two reasons that outlive that convenience:
   the per-ring mode is what the study is about and recomputing it would test
   our arithmetic against itself, and the PSF step is swappable, so a
   prediction that holds today is a prediction, not a measurement.
   ``interpolate_Lanczos`` and ``interpolate_bicubic`` are
   wrapped to count calls, and a ring is recorded as interpolated when the
   interpolator ran during its ``_iso_extract`` call. ``_iso_between`` never
   interpolates, so a call can only have come from the ring extraction.
"""

import hashlib
import importlib
import json
import os
import sys
import time
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
    """Wrap AutoProf's samplers so both sampling modes can be observed per ring.

    Two independent questions, neither answerable from stock ``.prof`` output:

    *Line or isophotal band?* ``_iso_between`` serves two callers in
    ``_Generate_Profile``: the curve-of-growth sum, which always passes an
    inner radius of 0, and isophotal-band sampling, which passes ``R - width``.
    Only the latter changes the estimand, so the inner radius is the
    discriminator.

    *Interpolated or rounded to the nearest pixel?* Decided inside
    ``_iso_extract`` by ``Rlim < rad_interp``. Rather than recompute that
    comparison here -- which would test our arithmetic against itself, and
    would need the PSF AutoProf measured -- we watch whether the interpolator
    actually ran: the two interpolators are wrapped in ``SharedFunctions``,
    where ``_iso_extract`` resolves them, and each ``_iso_extract`` call is
    attributed the calls made during it. ``_iso_between`` does not
    interpolate, so attribution is unambiguous.

    Returns a dict shared with the caller by reference, filled as the pipeline
    runs.
    """
    # importlib, not `from autoprof.pipeline_steps import Isophote_Extract`:
    # the package re-exports a *function* under the module's own name, so the
    # plain import binds the function and the module is unreachable.
    extract_module = importlib.import_module("autoprof.pipeline_steps.Isophote_Extract")
    shared_module = importlib.import_module("autoprof.autoprofutils.SharedFunctions")

    events = {
        "band_sampling_calls": 0,
        "total_calls": 0,
        "interpolator_calls": 0,
        "extractions": [],
    }

    original_between = extract_module._iso_between

    def between_probe(image, low, high, *args, **kwargs):
        events["total_calls"] += 1
        if float(low) != 0.0:
            events["band_sampling_calls"] += 1
        return original_between(image, low, high, *args, **kwargs)

    extract_module._iso_between = between_probe

    # ``_iso_extract`` is imported into Isophote_Extract's namespace, so this
    # is the binding ``_Generate_Profile`` actually calls.
    original_extract = extract_module._iso_extract

    def extract_probe(image, sma, *args, **kwargs):
        before = events["interpolator_calls"]
        try:
            return original_extract(image, sma, *args, **kwargs)
        finally:
            # Recorded in the ``finally`` so a ring that raised still leaves a
            # trace; a missing entry would silently shorten the ring list and
            # misalign it against the profile rows.
            events["extractions"].append(
                {
                    "sma_pix": float(sma),
                    # AutoProf's default when the option is absent
                    # (SharedFunctions.py:606); recorded, not assumed.
                    "rad_interp_pix": float(kwargs.get("rad_interp", 30)),
                    "interp_method": str(kwargs.get("interp_method", "lanczos")),
                    "interpolated": events["interpolator_calls"] > before,
                }
            )

    extract_module._iso_extract = extract_probe

    def wrap_interpolator(name):
        original = getattr(shared_module, name)

        def interpolator_probe(*args, **kwargs):
            events["interpolator_calls"] += 1
            return original(*args, **kwargs)

        setattr(shared_module, name, interpolator_probe)

    for interpolator in ("interpolate_Lanczos", "interpolate_bicubic"):
        wrap_interpolator(interpolator)

    return events


def _source_digest():
    """SHA-256 of the file whose four-line expression this whole study is about."""
    extract_module = importlib.import_module("autoprof.pipeline_steps.Isophote_Extract")

    with open(extract_module.__file__, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _measured_psf_fwhm(output_dir, name):
    """The PSF FWHM AutoProf measured, read back from its own .aux output.

    The interpolation threshold is this number times
    ``ap_iso_interpolate_start``, so a campaign that records only the setting
    has not recorded where the switch actually fell. ``PSF.py`` writes the
    line as ``psf fwhm: %.3f pix``; three decimals is AutoProf's own
    precision, which is why the exact threshold is taken from the probe
    instead and this value is reported alongside it as context.
    """
    aux_path = os.path.join(output_dir, name + ".aux")
    if not os.path.exists(aux_path):
        return None
    for line in open(aux_path):
        if line.strip().startswith("psf fwhm:"):
            try:
                return float(line.split(":", 1)[1].strip().split()[0])
            except (IndexError, ValueError):
                return None
    return None


def _attribute_extractions(extractions, rows, pixel_scale):
    """Line up the probe's per-call records with the profile's rows.

    ``_Generate_Profile`` extracts the rings in profile order, so the two
    lists should correspond one to one. That is checked rather than assumed:
    a mismatch in length, or a ring whose recorded semi-major axis is not the
    one in the profile, means the attribution is wrong and every per-ring mode
    below it is offset. In that case the modes are returned as ``None`` with a
    stated reason, because a silently misaligned label is worse than an
    absent one.
    """
    unknown = [{"interpolated": None, "rad_interp_pix": None} for _ in rows]

    if len(extractions) != len(rows):
        return {
            "per_ring": unknown,
            "all_interpolated": None,
            "all_nearest": None,
            "attribution_ok": False,
            "note": (
                "%d _iso_extract calls for %d profile rows; cannot attribute per-ring "
                "sampling mode" % (len(extractions), len(rows))
            ),
            "rad_interp_pix": None,
        }

    for extraction, row in zip(extractions, rows):
        # The profile stores R in arcsec; the probe sees the pixel value the
        # extractor was called with. Tolerance covers the CSV round trip only.
        if abs(extraction["sma_pix"] - row["sma_pix"]) > 1e-6 * max(1.0, row["sma_pix"]):
            return {
                "per_ring": unknown,
                "all_interpolated": None,
                "all_nearest": None,
                "attribution_ok": False,
                "note": (
                    "extraction at sma=%.6f px does not match profile row at %.6f px"
                    % (extraction["sma_pix"], row["sma_pix"])
                ),
                "rad_interp_pix": None,
            }

    modes = [{"interpolated": bool(e["interpolated"]), "rad_interp_pix": e["rad_interp_pix"]} for e in extractions]
    thresholds = {round(e["rad_interp_pix"], 9) for e in extractions}
    return {
        "per_ring": modes,
        "all_interpolated": all(m["interpolated"] for m in modes),
        "all_nearest": not any(m["interpolated"] for m in modes),
        "attribution_ok": True,
        "note": "",
        # One threshold for the whole image, since the PSF is measured once.
        "rad_interp_pix": thresholds.pop() if len(thresholds) == 1 else sorted(thresholds),
    }


def run_job(job):
    """Run one forced-profile job and return its rows, provenance, and timings."""
    output_dir = job["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    name = job["name"]
    rings = job["rings"]
    orders = job["orders"]

    image = np.asarray(job["image"], dtype=np.float64) if "image" in job else None
    image_path = job.get("image_path")
    fits_start = time.perf_counter()
    if image is not None:
        image_path = os.path.join(output_dir, f"{name}.fits")
        fits.PrimaryHDU(image).writeto(image_path, overwrite=True)
    fits_write_s = time.perf_counter() - fits_start

    forcing_start = time.perf_counter()
    profile_path, aux_path = _write_forcing_files(output_dir, name, rings, job["pixel_scale"])
    forcing_write_s = time.perf_counter() - forcing_start
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
        "ap_iso_measurecoefs": list(orders) if orders else None,
        "ap_isoclip": bool(job["isoclip"]),
        # Force line sampling unconditionally: see module docstring, point 3.
        "ap_isoband_fixed": True,
        "ap_isoband_width": 0.1,
        # A grid axis, not an inherited default: see the design spec's A3 and
        # module docstring point 5. Multiplied by results["psf fwhm"], so
        # the realized threshold is reported back rather than assumed.
        "ap_iso_interpolate_start": float(job.get("interpolate_start", 5.0)),
    }
    # The other half of the interpolation threshold. AutoProf's ``psf`` step
    # assumes 4.0 px and measures nothing, so this is the only way to move the
    # switch radius without touching ap_iso_interpolate_start -- which is what
    # makes the two independent knobs onto the same mechanism.
    if job.get("set_psf") is not None:
        options["ap_set_psf"] = float(job["set_psf"])
    options.update(job.get("extra_options", {}))

    pipeline_start = time.perf_counter()
    outcome = pipeline.Process_Image(options=options)
    pipeline_wall_s = time.perf_counter() - pipeline_start
    if outcome == 1:
        raise SystemExit(f"AutoProf reported failure for {name}")

    parse_start = time.perf_counter()
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
            # The *other* radial profile AutoProf publishes, and the wrong
            # one to build Track 2's denominator from. ``b0`` is the mean of
            # the exact vector that entered the FFT, so it is the estimator
            # consistent with the harmonic numerator; this column is a
            # *median*. Carried so the gap between them is measured rather
            # than asserted to be small.
            #
            # The column name depends on ``ap_fluxunits``: with "intensity"
            # AutoProf writes ``I``, and only in magnitude mode does it write
            # ``SB``. The design note calls it "the median SB profile"
            # because that is its name in AutoProf's default configuration;
            # it is the same estimator either way.
            "median_flux": (
                float(data["I"][index])
                if "I" in columns
                else float(data["SB"][index])
                if "SB" in columns
                else float("nan")
            ),
            "median_flux_column": "I" if "I" in columns else ("SB" if "SB" in columns else None),
            "pixels": int(data["pixels"][index]) if "pixels" in columns else -1,
        }
        for order in orders:
            for prefix in ("a", "b"):
                key = f"{prefix}{order}"
                entry[key] = float(data[key][index]) if key in columns else float("nan")
        rows.append(entry)

    interpolation = _attribute_extractions(events["extractions"], rows, job["pixel_scale"])
    for entry, ring_mode in zip(rows, interpolation["per_ring"]):
        entry["interpolated"] = ring_mode["interpolated"]
        entry["rad_interp_pix"] = ring_mode["rad_interp_pix"]

    import autoprof

    interpolate_start = float(options["ap_iso_interpolate_start"])
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
            "ap_iso_interpolate_start": interpolate_start,
            "ap_set_psf": job.get("set_psf"),
            "psf_fwhm_pix": _measured_psf_fwhm(output_dir, name),
            "rad_interp_pix": interpolation["rad_interp_pix"],
        },
        "sampling_mode": {
            "band_sampling_calls": events["band_sampling_calls"],
            "iso_between_total_calls": events["total_calls"],
            "all_rings_line_sampled": events["band_sampling_calls"] == 0,
            "interpolator_calls": events["interpolator_calls"],
            "iso_extract_calls": len(events["extractions"]),
            "per_ring_interpolated": [m["interpolated"] for m in interpolation["per_ring"]],
            "all_rings_interpolated": interpolation["all_interpolated"],
            "all_rings_nearest_pixel": interpolation["all_nearest"],
            "attribution_ok": interpolation["attribution_ok"],
            "attribution_note": interpolation["note"],
        },
        "timing": {
            "fits_write_s": fits_write_s,
            "forcing_write_s": forcing_write_s,
            "pipeline_wall_s": pipeline_wall_s,
            "pipeline_steps_s": {key: float(value) for key, value in outcome.items()},
            "profile_parse_s": time.perf_counter() - parse_start,
        },
    }
    return result


def main(job_path):
    with open(job_path) as handle:
        job = json.load(handle)
    result = run_job(job)
    with open(job["result_path"], "w") as handle:
        json.dump(result, handle)


if __name__ == "__main__":
    main(sys.argv[1])
