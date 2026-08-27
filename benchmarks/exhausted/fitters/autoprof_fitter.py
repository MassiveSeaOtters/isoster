"""AutoProf campaign fitter.

Runs AutoProf in an isolated venv via subprocess. When the venv is
missing or autoprof cannot import, the arm is reported as
``status="skipped"`` with a clear ``error_msg`` — never as a hard
failure. The campaign YAML's ``tools.autoprof.venv_python`` points at
the isolated python; we probe it lazily on the first arm of each
galaxy.

Output schema matches the other fitters: ``profile.fits`` / ``model.fits``
/ ``qa.png`` / ``run_record.json`` / ``config.yaml`` with an extra
``raw/`` subdirectory for AutoProf's native ``.prof`` / ``.aux`` /
``_genmodel.fits``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from astropy.io import fits

from benchmarks.autoprof_env import (
    DEFAULT_AUTOPROF_VENV_PYTHON,
    autoprof_install_hint,
    resolve_autoprof_python,
)
from isoster import build_isoster_model
from isoster.plotting import plot_qa_summary
from isoster.utils import isophote_results_to_fits

from ..adapters.base import GalaxyBundle
from ..analysis.inventory import INVENTORY_COLUMNS
from ..analysis.metrics import summarize_fit
from ..analysis.model_evaluation import evaluate_model_v11, profile_summary_for_inventory
from ..analysis.quality_flags import evaluate_flags

_VENV_PROBE_CACHE: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Small-image fallback — retry-on-failure knob pack
# ---------------------------------------------------------------------------
#
# Two AutoProf pipeline steps are known to crash on small cutouts (<= ~50 px
# per side) with the stock defaults:
#
#   * ``Center_HillClimb`` (``autoprof/pipeline_steps/Center.py``): builds
#     ``sampleradii = linspace(1, ap_centeringring, ap_centeringring) *
#     psf_fwhm``; the outermost ring is at ``10 * psf_fwhm`` = 40 px with
#     defaults. On a 33x33 image the ring is fully outside the frame,
#     ``_iso_extract`` returns an empty array, and ``np.quantile(..., 0.85)``
#     raises ``IndexError: index -1 is out of bounds for axis 0 with size 0``.
#
#   * ``Isophote_Extract._Generate_Profile`` (``pipeline_steps/
#     Isophote_Extract.py`` line ~168): when the last extracted ring falls
#     entirely outside the image, all samples are masked and
#     ``np.interp(theta, [], [], ...)`` raises
#     ``ValueError: array of sample points is empty``. Preceding log line:
#     ``WARNING: Entire Isophote is Masked!``.
#
# The fallback here is narrow on purpose: it fires only after the first
# attempt fails with one of the two signatures above, and it injects the
# smallest knob set that keeps AutoProf within the image:
#
#   * ``ap_centeringring`` = safe ring count from image half-extent / PSF,
#     so no centering ring extends beyond the frame.
#   * ``ap_truncate_evaluation = True`` — documented AutoProf stop condition
#     that terminates profile extraction once the ellipse escapes the image.
#   * ``ap_extractfull = False`` — belt-and-braces: never extract past the
#     fit limit.
#
# Anything else (the arm delta, isoclip knobs, center override) is left
# untouched, so we preserve the arm semantics as far as possible.


SMALL_IMAGE_FAILURE_SIGNATURES: tuple[tuple[str, ...], ...] = (
    ("Center_HillClimb", "index -1 is out of bounds"),
    ("_Generate_Profile", "array of sample points is empty"),
    ("Entire Isophote is Masked",),
)


def _read_autoprof_log(log_path: Path | None) -> str:
    """Return the autoprof.log contents if present, else an empty string."""
    if log_path is None or not log_path.is_file():
        return ""
    try:
        return log_path.read_text(errors="replace")
    except OSError:
        return ""


def _detect_small_image_failure(stderr: str, log_text: str) -> str | None:
    """Return a short tag describing which small-image signature matched, or None."""
    hay = (stderr or "") + "\n" + (log_text or "")
    for signature in SMALL_IMAGE_FAILURE_SIGNATURES:
        if all(needle in hay for needle in signature):
            return "::".join(signature)
    return None


def _small_image_fallback_delta(
    image_shape: tuple[int, ...],
    psf_fwhm_px: float = 4.0,
) -> dict[str, Any]:
    """Image-size-aware AutoProf knobs safe for small cutouts.

    AutoProf's hill-climb builds ``ap_centeringring`` rings at spacing
    ``psf_fwhm``, so the outermost sits at ``ring_count * psf_fwhm`` pixels
    from the seed center. With a default ``psf_fwhm=4`` and a 33x33 image
    (half-extent 16 px), the default ``ap_centeringring=10`` puts the last
    ring at 40 px — always outside the frame. We drop it to the largest
    value that keeps every ring inside the frame minus one-PSF safety
    margin.
    """
    half_extent = min(image_shape[:2]) // 2
    # Each ring is at r = i * psf_fwhm (i=1..N). Largest safe N satisfies
    # N * psf_fwhm <= half_extent - psf_fwhm. Floor to int, clamp >=2.
    ring_budget = int((half_extent - psf_fwhm_px) // max(psf_fwhm_px, 1.0))
    safe_ring_count = max(2, min(10, ring_budget))
    return {
        "ap_centeringring": safe_ring_count,
        "ap_truncate_evaluation": True,
        "ap_extractfull": False,
    }


def run_one_arm(
    bundle: GalaxyBundle,
    arm_id: str,
    arm_delta: dict[str, Any],
    output_dir: Path,
    *,
    write_qa: bool = True,
    write_model_fits: bool = True,
    sb_profile_scale: str = "log10",
    sb_asinh_softening: float | None = None,
    venv_python: str = DEFAULT_AUTOPROF_VENV_PYTHON,
    timeout: int = 300,
) -> dict[str, Any]:
    """Run one ``(galaxy, autoprof-arm)`` pair. Returns an inventory row."""
    total_start = time.perf_counter()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    row = _empty_inventory_row(bundle.metadata.galaxy_id, arm_id)

    # 1) Venv probe. Skip gracefully if unusable.
    probe_reason = _probe_venv(venv_python)
    if probe_reason is not None:
        row["status"] = "skipped"
        row["error_msg"] = probe_reason
        _write_run_record(output_dir, {"status": "skipped", "reason": probe_reason})
        return row

    # 2) Stage inputs — AutoProf needs a plain-HDU FITS on disk.
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    image = np.asarray(bundle.image, dtype=np.float64)
    galaxy_tag = bundle.metadata.galaxy_id.replace("/", "__")
    image_path = temp_dir / f"{galaxy_tag}_image.fits"
    fits.PrimaryHDU(data=image).writeto(image_path, overwrite=True)
    mask_path: Path | None = None
    if bundle.mask is not None:
        mask_path = temp_dir / f"{galaxy_tag}_mask.fits"
        # AutoProf convention: 0 = good, >0 = bad (matches isoster bool mask)
        fits.PrimaryHDU(data=np.asarray(bundle.mask, dtype=np.int16)).writeto(mask_path, overwrite=True)

    # 3) Build options JSON. A per-arm _fix_center_from sentinel can
    # pin AutoProf's center to an external reference; absent that, we
    # let AutoProf run its own centering pipeline (the recommended
    # mode for the benchmark baseline).
    geom = bundle.initial_geometry
    center_override, center_error = _resolve_center_override(
        arm_delta=arm_delta or {},
        arm_dir=output_dir,
    )
    if center_error is not None:
        row["status"] = "skipped"
        row["error_msg"] = center_error
        _write_run_record(output_dir, {"status": "skipped", "reason": center_error})
        return row
    options = _build_options(
        bundle=bundle,
        arm_delta=arm_delta,
        image_path=image_path,
        mask_path=mask_path,
        save_dir=str(raw_dir),
        galaxy_tag=galaxy_tag,
        center_override=center_override,
    )
    json_path = temp_dir / f"{galaxy_tag}_options.json"
    with json_path.open("w") as handle:
        json.dump(options, handle, indent=2)

    # 4) Subprocess invocation. On failure with a known small-image
    # signature we merge a narrow fallback-knob pack into options and
    # retry exactly once. This keeps every other scenario running with
    # AutoProf's stock behavior.
    worker_script = Path(__file__).with_name("autoprof_worker.py")
    autoprof_log_path = raw_dir / f"{galaxy_tag}_autoprof.log"
    status_path = raw_dir / f"{galaxy_tag}_status.json"

    venv_python_resolved = resolve_autoprof_python(venv_python)

    def _invoke_once() -> tuple[int, str, float]:
        fit_start = time.perf_counter()
        try:
            proc_ = subprocess.run(
                [venv_python_resolved, str(worker_script), str(json_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return proc_.returncode, proc_.stderr, time.perf_counter() - fit_start
        except subprocess.TimeoutExpired:
            return -1, f"timeout after {timeout}s", time.perf_counter() - fit_start

    rc, stderr, wall_fit = _invoke_once()
    small_image_fallback: dict[str, Any] | None = None
    small_image_signature: str | None = None
    if rc != 0:
        signature = _detect_small_image_failure(stderr, _read_autoprof_log(autoprof_log_path))
        if signature is not None:
            small_image_signature = signature
            small_image_fallback = _small_image_fallback_delta(image.shape)
            options.update(small_image_fallback)
            with json_path.open("w") as handle:
                json.dump(options, handle, indent=2)
            # Clear prior status / log so the retry's state is unambiguous;
            # keep the first-attempt log for audit under a suffix.
            if status_path.is_file():
                status_path.unlink()
            if autoprof_log_path.is_file():
                autoprof_log_path.rename(autoprof_log_path.with_name(autoprof_log_path.name + ".attempt1"))
            rc, stderr, extra_wall = _invoke_once()
            wall_fit += extra_wall
    fallback_record = {
        "small_image_fallback": small_image_fallback,
        "small_image_signature": small_image_signature,
    }
    if rc == -1:
        row["status"] = "failed"
        row["error_msg"] = stderr
        row["wall_time_fit_s"] = float(wall_fit)
        row["wall_time_total_s"] = float(time.perf_counter() - total_start)
        _write_run_record(
            output_dir,
            {"status": "failed", "error_msg": stderr, "wall_time_fit_s": float(wall_fit), **fallback_record},
        )
        return row
    if rc != 0 and not status_path.is_file():
        row["status"] = "failed"
        row["error_msg"] = (stderr or f"returncode={rc}")[:500]
        row["wall_time_fit_s"] = float(wall_fit)
        row["wall_time_total_s"] = float(time.perf_counter() - total_start)
        _write_run_record(
            output_dir,
            {"status": "failed", "error_msg": row["error_msg"], **fallback_record},
        )
        return row

    if status_path.is_file():
        with status_path.open() as handle:
            status_data = json.load(handle)
        if status_data.get("status") != "ok":
            row["status"] = "failed"
            row["error_msg"] = str(status_data.get("error_msg", "unknown autoprof error"))
            row["wall_time_fit_s"] = float(wall_fit)
            row["wall_time_total_s"] = float(time.perf_counter() - total_start)
            _write_run_record(output_dir, {"status": "failed", **status_data, **fallback_record})
            return row

    # 5) Parse AutoProf outputs.
    prof_path = raw_dir / f"{galaxy_tag}.prof"
    aux_path = raw_dir / f"{galaxy_tag}.aux"
    # The angle basis is chosen at runtime by the same flag that selects the
    # resampling: with a mask or ``ap_isoclip=True`` AutoProf re-interpolates
    # onto polar angle from the image x-axis, and otherwise works in the
    # eccentric anomaly. It decides whether a same-order rotation is even a
    # meaningful operation, so it is recorded per row rather than assumed.
    harmonic_basis = (
        "polar_from_image_x_axis"
        if bool(options.get("ap_isoclip", AUTOPROF_DEFAULTS["ap_isoclip"]))
        else "eccentric_anomaly"
    )
    isophotes, n_raw, n_filtered = _parse_prof_file(
        prof_path,
        pixel_scale_arcsec=bundle.metadata.pixel_scale_arcsec,
        sb_zeropoint=bundle.metadata.sb_zeropoint,
        harmonic_basis=harmonic_basis,
    )
    aux_info = _parse_aux_file(aux_path)

    if not isophotes:
        row["status"] = "failed"
        row["error_msg"] = f"empty profile (n_raw={n_raw}, filtered={n_filtered})"
        row["wall_time_fit_s"] = float(wall_fit)
        row["wall_time_total_s"] = float(time.perf_counter() - total_start)
        _write_run_record(
            output_dir,
            {"status": "failed", "error_msg": row["error_msg"]},
        )
        return row

    # Stamp centers onto every isophote. AutoProf writes a "center x:"
    # line only when it actually ran its centering pipeline. With
    # `ap_set_center` present, the pipeline is skipped and the .aux
    # file only carries the round-tripped `option ap_set_center` line.
    # Fall back to the value we passed in (initial_geometry) so the
    # downstream model / QA / drift metric all see the correct pixel.
    cx = float(aux_info.get("center_x", np.nan))
    cy = float(aux_info.get("center_y", np.nan))
    if not np.isfinite(cx) or not np.isfinite(cy):
        cx = float(geom["x0"])
        cy = float(geom["y0"])
    for iso in isophotes:
        if "x0" not in iso or not np.isfinite(iso.get("x0", np.nan)):
            iso["x0"] = cx
        if "y0" not in iso or not np.isfinite(iso.get("y0", np.nan)):
            iso["y0"] = cy

    results: dict[str, Any] = {
        "isophotes": isophotes,
        "tool": "autoprof",
        "config": dict(arm_delta or {}),
        "first_isophote_failure": False,
        "first_isophote_retry_log": [],
        "autoprof_aux": aux_info,
        "autoprof_n_raw": n_raw,
        "autoprof_n_filtered": n_filtered,
        "profile_schema_version": PROFILE_SCHEMA_VERSION,
        "harmonic_basis": harmonic_basis,
        "small_image_fallback": small_image_fallback,
        "small_image_signature": small_image_signature,
    }

    profile_path = output_dir / "profile.fits"
    isophote_results_to_fits(results, str(profile_path), overwrite=True)

    config_path = output_dir / "config.yaml"
    with config_path.open("w") as handle:
        yaml.safe_dump(
            {"tool": "autoprof", "arm_id": arm_id, **(arm_delta or {})},
            handle,
            sort_keys=False,
        )

    # 6) Model: prefer AutoProf's genmodel; fall back to our rebuild.
    genmodel_path = raw_dir / f"{galaxy_tag}_genmodel.fits"
    model: np.ndarray
    if genmodel_path.is_file():
        with fits.open(genmodel_path) as hdul:
            # AutoProf places model in HDU 1
            model_hdu = hdul[1] if len(hdul) > 1 else hdul[0]
            model = np.asarray(model_hdu.data, dtype=np.float64)
    else:
        model = build_isoster_model(image.shape, isophotes)

    model_path = None
    if write_model_fits:
        model_path = output_dir / "model.fits"
        residual = image - model
        primary = fits.PrimaryHDU(
            data=model.astype(np.float32),
            header=fits.Header({"EXTNAME": "MODEL", "ARM_ID": arm_id}),
        )
        resid_hdu = fits.ImageHDU(
            data=residual.astype(np.float32),
            header=fits.Header({"EXTNAME": "RESIDUAL", "ARM_ID": arm_id}),
        )
        fits.HDUList([primary, resid_hdu]).writeto(model_path, overwrite=True)

    qa_path = None
    if write_qa:
        qa_path = output_dir / "qa.png"
        try:
            plot_qa_summary(
                title=f"{bundle.metadata.galaxy_id}  |  autoprof::{arm_id}",
                image=image,
                isoster_model=model,
                isoster_res=isophotes,
                mask=bundle.mask,
                filename=str(qa_path),
                relative_residual=False,
                sb_zeropoint=bundle.metadata.sb_zeropoint,
                pixel_scale_arcsec=bundle.metadata.pixel_scale_arcsec,
                sb_profile_scale=sb_profile_scale,
                sb_asinh_softening=sb_asinh_softening,
            )
        except Exception as exc:  # noqa: BLE001
            qa_path.with_suffix(".png.err.txt").write_text(f"{type(exc).__name__}: {exc}\n")
            qa_path = None

    # 7) Metrics + v1.1 model evaluation + flags (shared pipeline).
    metrics = summarize_fit(results, sma0=float(geom.get("sma0", 1.0)))
    metrics.update(profile_summary_for_inventory(str(profile_path)))
    model_metrics = evaluate_model_v11(
        image=image,
        model=model,
        mask=bundle.mask,
        x0=float(geom["x0"]),
        y0=float(geom["y0"]),
        eps=float(geom.get("eps", 0.2)),
        pa_rad=float(geom.get("pa", 0.0)),
        R_ref_pix=bundle.metadata.effective_Re_pix,
        maxsma_pix=float(geom.get("maxsma", min(image.shape) // 2)),
        r_inner_floor_pix=float(metrics.get("min_sma_pix", 0.0) or 0.0),
    )

    row.update(
        {
            "status": "ok",
            "error_msg": "",
            "wall_time_fit_s": float(wall_fit),
            "wall_time_total_s": float(time.perf_counter() - total_start),
            **metrics,
            **model_metrics,
            "first_isophote_failure": False,
            "first_isophote_retry_attempts": 0,
            "first_isophote_retry_stop_codes": "",
            "qa_path": str(qa_path) if qa_path else "",
            "profile_path": str(profile_path),
            "model_path": str(model_path) if model_path else "",
            "config_path": str(config_path),
        }
    )
    row.update(evaluate_flags(row))
    _write_run_record(
        output_dir,
        {
            "status": "ok",
            "wall_time_fit_s": float(wall_fit),
            "wall_time_total_s": row["wall_time_total_s"],
            "metrics": {**metrics, **model_metrics},
            "flags": row.get("flags", ""),
            "flag_severity_max": row.get("flag_severity_max", 0.0),
            "arm_delta": arm_delta,
            "autoprof_aux": aux_info,
            "autoprof_n_raw": n_raw,
            "autoprof_n_filtered": n_filtered,
            "fix_center_override": center_override,
            "small_image_fallback": small_image_fallback,
            "small_image_signature": small_image_signature,
        },
    )
    return row


# ---------------------------------------------------------------------------
# Venv probe
# ---------------------------------------------------------------------------


def _probe_venv(venv_python: str) -> str | None:
    """Return ``None`` if the venv can import autoprof; otherwise a skip reason.

    Result cached per ``venv_python`` path so only the first arm on the
    first galaxy pays the probe cost. The path goes through
    :func:`resolve_autoprof_python`, so YAMLs and CLI args may use
    ``~/.venvs/...`` and an unset value falls back to ``AUTOPROF_PYTHON``
    and then to the canonical default.
    """
    venv_path = Path(resolve_autoprof_python(venv_python))
    cache_key = str(venv_path)
    cached = _VENV_PROBE_CACHE.get(cache_key)
    if cached == "OK":
        return None
    if cached is not None:
        return cached
    if not venv_path.is_file():
        reason = autoprof_install_hint(venv_path)
        _VENV_PROBE_CACHE[cache_key] = reason
        return reason
    try:
        proc = subprocess.run(
            [str(venv_path), "-c", "import autoprof; print('ok')"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as exc:  # noqa: BLE001
        reason = f"autoprof probe crashed: {exc}"
        _VENV_PROBE_CACHE[cache_key] = reason
        return reason
    if proc.returncode != 0:
        reason = f"autoprof import failed inside venv ({venv_path}): {proc.stderr.strip()[:200]}"
        _VENV_PROBE_CACHE[cache_key] = reason
        return reason
    _VENV_PROBE_CACHE[cache_key] = "OK"
    return None


# ---------------------------------------------------------------------------
# Options builder
# ---------------------------------------------------------------------------

AUTOPROF_DEFAULTS: dict[str, Any] = {
    "ap_isoclip": True,
    "ap_isoclip_nsigma": 5.0,
    "ap_isoaverage_method": "median",
    "ap_regularize_scale": 1.0,
    "ap_fit_limit": 2.0,
    "ap_samplegeometricscale": 0.1,
    "ap_iso_interpolate_start": 5,
    "ap_iso_measurecoefs": (3, 4),
}


def _build_options(
    *,
    bundle: GalaxyBundle,
    arm_delta: dict[str, Any],
    image_path: Path,
    mask_path: Path | None,
    save_dir: str,
    galaxy_tag: str,
    center_override: dict[str, float] | None = None,
) -> dict[str, Any]:
    cfg = dict(AUTOPROF_DEFAULTS)
    cfg.update(arm_delta or {})
    geom = bundle.initial_geometry
    md = bundle.metadata

    options: dict[str, Any] = {
        "ap_image_file": str(image_path),
        "ap_name": galaxy_tag,
        "ap_pixscale": md.pixel_scale_arcsec,
        "ap_zeropoint": md.sb_zeropoint,
        "ap_process_mode": "image",
        "ap_doplot": False,
        "ap_saveto": save_dir + "/",
        "ap_plotpath": save_dir + "/",
        "ap_iso_measurecoefs": list(cfg["ap_iso_measurecoefs"]),
        "ap_isoclip": bool(cfg["ap_isoclip"]),
        "ap_isoaverage_method": cfg["ap_isoaverage_method"],
        "ap_regularize_scale": float(cfg["ap_regularize_scale"]),
        "ap_fit_limit": float(cfg["ap_fit_limit"]),
        "ap_samplegeometricscale": float(cfg["ap_samplegeometricscale"]),
        "ap_iso_interpolate_start": int(cfg["ap_iso_interpolate_start"]),
    }
    if cfg["ap_isoclip"]:
        options["ap_isoclip_nsigma"] = float(cfg["ap_isoclip_nsigma"])
    if mask_path is not None:
        options["ap_mask_file"] = str(mask_path)
        options["ap_mask_hdu"] = 0
    # By default, let AutoProf run its own centering pipeline — the
    # fairest comparison is the tool used as recommended. The adapter's
    # (x0, y0) is passed as `ap_guess_center` (a seed, *not* a fix) so
    # the centering routine starts near the galaxy rather than at the
    # image-center fallback. Only when an arm supplies a
    # `_fix_center_from` sentinel do we hard-fix the center via
    # `ap_set_center` (see _resolve_center_override).
    options["ap_guess_center"] = {"x": float(geom["x0"]), "y": float(geom["y0"])}
    if center_override is not None:
        options["ap_set_center"] = {
            "x": float(center_override["x"]),
            "y": float(center_override["y"]),
        }
    # Optional PA / ellipticity seeds from the adapter.
    options["ap_isoinit_pa_set"] = float(np.degrees(geom.get("pa", 0.0)))
    options["ap_isoinit_ellip_set"] = float(geom.get("eps", 0.2))
    return options


def _resolve_center_override(
    *,
    arm_delta: dict[str, Any],
    arm_dir: Path,
) -> tuple[dict[str, float] | None, str | None]:
    """Resolve the optional ``_fix_center_from`` sentinel on an arm.

    Returns ``(center_dict, None)`` when the sentinel is honored,
    ``(None, None)`` when the arm uses AutoProf's own centering, and
    ``(None, error_msg)`` when the sentinel cannot be resolved (the
    caller should mark the arm as ``status="skipped"``).

    Supported sentinels:
      - ``isoster_weighted``: intensity-weighted (x0, y0) over the
        first 10 stop_code==0 isophotes of the companion
        ``isoster/arms/ref_default/profile.fits``. Requires isoster to
        have run first for the same galaxy.
    """
    source = arm_delta.get("_fix_center_from")
    if source is None:
        return None, None
    source = str(source)
    if source != "isoster_weighted":
        return None, f"unknown _fix_center_from sentinel: {source!r}"
    # arm_dir layout: <galaxy_dir>/autoprof/arms/<arm_id>/
    galaxy_dir = arm_dir.parents[2]
    isoster_profile = galaxy_dir / "isoster" / "arms" / "ref_default" / "profile.fits"
    if not isoster_profile.is_file():
        return None, (
            "fix_center requires isoster ref_default profile at "
            f"{isoster_profile} but it does not exist. Enable the isoster "
            "tool in the campaign and make sure it runs before autoprof."
        )
    try:
        with fits.open(isoster_profile) as hdul:
            table = hdul[1].data
            sma = np.asarray(table["sma"], dtype=float)
            x0 = np.asarray(table["x0"], dtype=float)
            y0 = np.asarray(table["y0"], dtype=float)
            intens = np.asarray(table["intens"], dtype=float)
            stop_code = np.asarray(table["stop_code"], dtype=int)
    except Exception as exc:  # noqa: BLE001
        return None, f"fix_center failed to read {isoster_profile}: {exc}"
    order = np.argsort(sma)
    x0 = x0[order]
    y0 = y0[order]
    intens = intens[order]
    stop_code = stop_code[order]
    valid = (stop_code == 0) & np.isfinite(x0) & np.isfinite(y0) & np.isfinite(intens) & (intens > 0.0)
    pool_x = x0[valid][:10]
    pool_y = y0[valid][:10]
    pool_i = intens[valid][:10]
    if pool_x.size == 0:
        return None, "fix_center: no stop_code==0 isophotes in isoster ref_default"
    weights = pool_i / pool_i.sum()
    cx = float(np.sum(weights * pool_x))
    cy = float(np.sum(weights * pool_y))
    if not (np.isfinite(cx) and np.isfinite(cy)):
        return None, "fix_center produced non-finite (cx, cy)"
    return {"x": cx, "y": cy, "n_used": int(pool_x.size)}, None


# ---------------------------------------------------------------------------
# .prof / .aux parsing
# ---------------------------------------------------------------------------

#: Orders AutoProf is asked for, and therefore the ones the schema carries.
HARMONIC_ORDERS = (3, 4)

#: Bumped because the meaning of an existing column changed. Before this,
#: ``a3`` in an AutoProf arm held AutoProf's *native* coefficient while ``a3``
#: in an isoster or photutils arm held a Bender-normalized major-axis one --
#: two quantities under one name, and no way to tell archived files apart.
#: Old files are not readable as new ones and must not be silently mixed with
#: them; see ``docs/09-exhausted-benchmark.md`` for the migration note.
PROFILE_SCHEMA_VERSION = 2


def _harmonic_schema_fields(iso: dict[str, Any], harmonic_basis: str) -> dict[str, Any]:
    """The A5 columns: native values preserved, converted values only if valid.

    The bare names ``a3``/``b3``/``a4``/``b4`` keep their established meaning
    across every tool --- Bender-normalized, major-axis frame, comparable with
    isoster and photutils. For an AutoProf arm they are **NaN**, together with
    a false validity flag and a stated reason, rather than filled with a
    plausible number.

    Two things block the conversion here, and only one of them is about the
    angle basis:

    * **No gradient.** Bender normalization divides by ``sma * |dI/da|`` and
      AutoProf reports no radial gradient. Part A's Track 2 would reconstruct
      one by finite-differencing AutoProf's own ``b0`` profile, but that is a
      measurement Part A has not yet licensed, so nothing here invents it.
    * **The angle basis, when it is the eccentric-anomaly one.** That is not a
      rotation of the polar basis but a different one, and changing between
      them mixes harmonic *orders* --- no same-order two-component transform
      can express it. Part A measured the cost of pretending otherwise: 12% at
      eps = 0.3 and 63% at eps = 0.6.

    A NaN here is correct on its own. The reason column is what makes it
    diagnostic a year later, when "the AutoProf harmonics are NaN" and "they
    are NaN because AutoProf reports no gradient" are different amounts of
    information.
    """
    if harmonic_basis == "eccentric_anomaly":
        reason = "eccentric_anomaly_basis_mixes_orders; and no radial gradient reported"
    else:
        reason = "no_radial_gradient_reported"

    fields: dict[str, Any] = {
        "harmonic_basis": harmonic_basis,
        "harmonic_conversion_valid": False,
        "harmonic_conversion_reason": reason,
        # AutoProf's own failure reason for a NaN row. It reports none, so the
        # column exists and says so rather than being absent for this tool and
        # present for the others.
        "harmonic_measurement_status": "not_reported_by_tool",
    }
    for order in HARMONIC_ORDERS:
        # Raw sine/cosine amplitudes are exact from the native pair and b0,
        # and they are the primary track precisely because they need no
        # gradient. They stay in the *sky* frame here: rotating them to the
        # major-axis frame needs a per-ring position angle, which a free fit
        # supplies but which the rotation's validity depends on the basis for.
        native_a = iso.get(f"autoprof_a{order}_native", float("nan"))
        native_b = iso.get(f"autoprof_b{order}_native", float("nan"))
        scale = 2.0 * abs(iso.get("autoprof_b0", float("nan")))
        fields[f"s{order}_raw_sky"] = -scale * native_a
        fields[f"c{order}_raw_sky"] = scale * native_b
        fields[f"a{order}"] = float("nan")
        fields[f"b{order}"] = float("nan")
    return fields


def _parse_prof_file(
    prof_path: Path,
    *,
    pixel_scale_arcsec: float,
    sb_zeropoint: float,
    harmonic_basis: str = "unknown",
) -> tuple[list[dict[str, Any]], int, int]:
    """Convert an AutoProf .prof into an isoster-compatible list of dicts."""
    if not prof_path.is_file():
        return [], 0, 0
    data = np.genfromtxt(prof_path, delimiter=",", names=True, skip_header=1)
    if data.ndim == 0:
        data = np.array([data])
    n_raw = len(data)
    valid = data["SB"] < 90.0
    n_filtered = int(np.sum(~valid))
    data = data[valid]
    if len(data) == 0:
        return [], n_raw, n_filtered

    # SMA in pixels
    sma = data["R"] / pixel_scale_arcsec
    sb = data["SB"]
    sb_err = data["SB_e"]
    intens_arcsec2 = 10.0 ** (-(sb - sb_zeropoint) / 2.5)
    intens = intens_arcsec2 * pixel_scale_arcsec**2
    intens_err = intens * np.log(10.0) / 2.5 * np.abs(sb_err)

    eps = data["ellip"]
    eps_err = data["ellip_e"]
    pa_rad = np.deg2rad(data["pa"] - 90.0) % np.pi
    pa_err = np.deg2rad(data["pa_e"])

    ndata = (
        np.asarray(data["pixels"], dtype=np.int32)
        if "pixels" in data.dtype.names
        else np.zeros(len(data), dtype=np.int32)
    )
    nflag = (
        np.asarray(data["maskedpixels"], dtype=np.int32)
        if "maskedpixels" in data.dtype.names
        else np.zeros(len(data), dtype=np.int32)
    )
    # A5: native AutoProf coefficients must never share a column name with
    # Bender-normalized ones. They previously did -- these values went
    # straight into ``a3``/``b3``/``a4``/``b4``, the same keys the isoster and
    # photutils fitters fill with Bender-normalized major-axis values. Two
    # different quantities under one name, indistinguishable once written.
    #
    # AutoProf's are ``a_n = -S_n / (2|b0|)``, ``b_n = +C_n / (2|b0|)``,
    # measured in whatever angle basis the run used. See
    # ``benchmarks/harmonic_scale/conventions.py``.
    harmonics: dict[str, np.ndarray] = {}
    for order in HARMONIC_ORDERS:
        for prefix in ("a", "b"):
            key = f"{prefix}{order}"
            if key in data.dtype.names:
                harmonics[f"autoprof_{key}_native"] = np.asarray(data[key], dtype=np.float64)
    # The DC term, which makes the raw reconstruction exact rather than an
    # estimate. Without it nothing downstream can undo AutoProf's
    # normalization.
    b0 = np.asarray(data["b0"], dtype=np.float64) if "b0" in data.dtype.names else np.full(len(data), np.nan)

    rows: list[dict[str, Any]] = []
    for i in range(len(data)):
        iso = {
            "sma": float(sma[i]),
            "intens": float(intens[i]),
            "intens_err": float(intens_err[i]),
            "eps": float(eps[i]),
            "eps_err": float(eps_err[i]),
            "pa": float(pa_rad[i]),
            "pa_err": float(pa_err[i]),
            "x0": float("nan"),
            "y0": float("nan"),
            "x0_err": 0.0,
            "y0_err": 0.0,
            "rms": float("nan"),
            "stop_code": 0,
            "ndata": int(ndata[i]),
            "nflag": int(nflag[i]),
            "niter": 0,
            "grad": float("nan"),
            "grad_error": float("nan"),
            "grad_r_error": float("nan"),
            "tflux_e": 0.0,
            "tflux_c": 0.0,
            "npix_e": 0,
            "npix_c": 0,
            "lsb_locked": False,
            "tool": "autoprof",
        }
        for key, arr in harmonics.items():
            iso[key] = float(arr[i])
        iso["autoprof_b0"] = float(b0[i])
        iso.update(_harmonic_schema_fields(iso, harmonic_basis))
        rows.append(iso)
    return rows, n_raw, n_filtered


def _parse_aux_file(aux_path: Path) -> dict[str, float]:
    info = {
        "center_x": float("nan"),
        "center_y": float("nan"),
        "background": float("nan"),
        "background_noise": float("nan"),
        "psf_fwhm": float("nan"),
    }
    if not aux_path.is_file():
        return info
    for raw_line in aux_path.read_text().splitlines():
        line = raw_line.strip()
        if line.startswith("center x:"):
            try:
                parts = line.split(",")
                info["center_x"] = float(parts[0].split(":")[1].strip().split()[0])
                info["center_y"] = float(parts[1].split(":")[1].strip().split()[0])
            except Exception:  # noqa: BLE001
                pass
        elif line.startswith("option ap_set_center:"):
            # Written by AutoProf when ap_set_center is supplied and
            # the fitting centering steps are skipped. Format example:
            #   option ap_set_center: {'x': 566.0, 'y': 566.0}
            try:
                import ast

                payload = line.split(":", 1)[1].strip()
                parsed = ast.literal_eval(payload)
                info["center_x"] = float(parsed["x"])
                info["center_y"] = float(parsed["y"])
            except Exception:  # noqa: BLE001
                pass
        elif line.startswith("background:"):
            try:
                after = line.split(":", 1)[1].strip()
                info["background"] = float(after.split()[0])
                if "noise:" in line:
                    info["background_noise"] = float(line.split("noise:")[1].strip().split()[0])
            except (ValueError, IndexError):
                pass
        elif line.startswith("psf fwhm:"):
            try:
                info["psf_fwhm"] = float(line.split(":")[1].strip().split()[0])
            except (ValueError, IndexError):
                pass
    return info


# ---------------------------------------------------------------------------
# Inventory row skeleton + run_record helper (mirrors photutils_fitter)
# ---------------------------------------------------------------------------


def _empty_inventory_row(galaxy_id: str, arm_id: str) -> dict[str, Any]:
    row: dict[str, Any] = {col: "" for col in INVENTORY_COLUMNS}
    row["galaxy_id"] = galaxy_id
    row["tool"] = "autoprof"
    row["arm_id"] = arm_id
    row["status"] = "pending"
    for int_col in (
        "n_iso",
        "n_stop_0",
        "n_stop_1",
        "n_stop_2",
        "n_stop_m1",
        "n_locked",
        "first_isophote_retry_attempts",
        "n_iso_ref_used",
    ):
        row[int_col] = 0
    row["first_isophote_failure"] = False
    for num_col in (
        "wall_time_fit_s",
        "wall_time_total_s",
        "frac_stop_nonzero",
        "combined_drift_pix",
        "spline_rms_center",
        "max_dpa_deg",
        "max_deps",
        "outer_gerr_median",
        "outward_drift_x",
        "outward_drift_y",
        "locked_drift_x",
        "locked_drift_y",
        "resid_rms_inner",
        "resid_rms_mid",
        "resid_rms_outer",
        "resid_median_inner",
        "resid_median_mid",
        "resid_median_outer",
        "frac_above_3sigma_outer",
        "image_sigma_adu",
        "flag_severity_max",
        "composite_score",
    ):
        row[num_col] = float("nan")
    return row


def _write_run_record(output_dir: Path, record: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "run_record.json").open("w") as handle:
        json.dump(record, handle, indent=2, default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


__all__ = ["run_one_arm"]
# keep shutil import used for future cleanup if needed
_ = shutil
