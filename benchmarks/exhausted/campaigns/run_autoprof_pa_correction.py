"""Two-phase corrective campaign with archived-reference and data-safety checks.

Usage: python -m benchmarks.exhausted.campaigns.run_autoprof_pa_correction CONFIG
An existing campaign directory is always rejected. Interrupted work requires
a separately named recovery, not an implicit overwrite or quality-based retry.
"""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml

from benchmarks.exhausted.adapters.base import safe_galaxy_id
from benchmarks.exhausted.analysis.publication_huang import digest, join_records, load_profile
from benchmarks.exhausted.fitters.autoprof_fitter import _resolve_center_override
from benchmarks.exhausted.orchestrator.config_loader import load_campaign
from benchmarks.exhausted.orchestrator.runner import run_campaign
from benchmarks.utils.autoprof_adapter import isoster_pa_to_autoprof_init

ARMS = {"baseline", "deep", "high_regularization", "fix_center"}


def phase_plan(plan, tool):
    """Reuse the runner, retaining truthful snapshots for both execution phases."""
    phase = copy.deepcopy(plan)
    for name, entry in phase.tools.items():
        entry.enabled = name == tool
        phase.raw["tools"][name]["enabled"] = name == tool
    return phase


def archived_sources(root, datasets):
    """Return accepted original records, including the explicit Huang recoveries."""
    sources = {}
    if "huang2013" in datasets:
        rows, _ = join_records(root)
        for row in rows:
            if row["tool"] in {"isoster", "autoprof"} and (row["tool"] == "autoprof" or row["arm"] == "ref_default"):
                sources["huang2013", row["galaxy"] + "/" + row["scenario"], row["tool"], row["arm"]] = Path(
                    row["record_path"]
                )
    if "s4g" in datasets:
        base = root / "fits/publication_single_band_s4g_2026_09_08/s4g"
        for record in sorted(base.glob("*/*/arms/*/run_record.json")):
            tool, arm = record.parents[2].name, record.parent.name
            if tool == "autoprof" or (tool == "isoster" and arm == "ref_default"):
                sources["s4g", record.parents[3].name.replace("__", "/"), tool, arm] = record
    return sources


def center_for(galaxy_dir):
    center, error = _resolve_center_override(
        arm_delta={"_fix_center_from": "isoster_weighted"}, arm_dir=galaxy_dir / "autoprof/arms/fix_center"
    )
    if error or center is None:
        raise RuntimeError(error or "Missing reference center")
    return np.array([center["x"], center["y"]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    plan = load_campaign(args.config)
    root = plan.output_root.parent
    target = plan.output_root / plan.campaign_name
    if not plan.campaign_name.startswith("publication_autoprof_pa_") or plan.output_root.name != "fits":
        raise ValueError("Only separately named publication PA campaigns are accepted")
    if set(plan.tools["autoprof"].arms) != ARMS or set(plan.tools["isoster"].arms) != {"ref_default"}:
        raise ValueError("Unexpected corrective arm roster")
    if plan.tools["photutils"].enabled or not 1 <= plan.execution["max_parallel_galaxies"] <= 8:
        raise ValueError("Unexpected tools or concurrency")
    if target.exists():
        raise FileExistsError(f"Refusing existing campaign: {target}")
    selected = []
    for dataset in plan.datasets.values():
        if not dataset.enabled:
            continue
        ids = dataset.adapter.list_galaxies()
        if dataset.select:
            if not set(dataset.select) <= set(ids):
                raise ValueError("Missing requested validation input")
            ids = [g for g in ids if g in dataset.select]
        selected.extend((dataset.name, galaxy) for galaxy in ids)
    if not selected:
        raise ValueError("Empty campaign")
    sources = archived_sources(root, {d for d, _ in selected})
    files, old_options = {args.config.resolve()}, []
    for dataset, galaxy in selected:
        for tool, arms in (("isoster", ["ref_default"]), ("autoprof", sorted(ARMS))):
            for arm in arms:
                path = sources[dataset, galaxy, tool, arm]
                files.add(path)
                manifest = path.parents[3] / "MANIFEST.json"
                files.add(manifest)
                files.add(Path(json.loads(manifest.read_text())["extra"]["fits_path"]))
                if (path.parent / "profile.fits").is_file():
                    files.add(path.parent / "profile.fits")
                if tool == "autoprof":
                    options_path = next((path.parent / "tmp").glob("*_options.json"))
                    files.add(options_path)
                    options = json.loads(options_path.read_text())
                    intended = json.loads(manifest.read_text())["initial_geometry"]["pa"]
                    difference = (options["ap_isoinit_pa_set"] - isoster_pa_to_autoprof_init(intended) + 90) % 180 - 90
                    old_options.append(dict(dataset=dataset, galaxy=galaxy, arm=arm, pa_error_deg=difference))
    target.mkdir(parents=True, exist_ok=False)
    control = target / "correction_audit"
    control.mkdir()
    before = {str(path): digest(path) for path in sorted(files)}
    (control / "source_hashes_before.json").write_text(json.dumps(before, indent=2) + "\n")
    (control / "old_options_audit.json").write_text(json.dumps(old_options, indent=2) + "\n")
    (control / "requested_config.yaml").write_text(yaml.safe_dump(plan.raw, sort_keys=False))
    metadata = dict(
        started=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        python=sys.version,
        platform=platform.platform(),
        workers=plan.execution["max_parallel_galaxies"],
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        packages={d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        requested_images=len(selected),
        requested_autoprof=len(selected) * 4,
    )
    autoprof_python = str(Path(plan.tools["autoprof"].extra["venv_python"]).expanduser())
    metadata["autoprof_environment"] = json.loads(
        subprocess.check_output(
            [
                autoprof_python,
                "-c",
                "import sys,json,importlib.metadata as m; print(json.dumps({'python':sys.version,'packages':{d.metadata['Name']:d.version for d in m.distributions()}}))",
            ],
            text=True,
        )
    )
    (control / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    started = time.monotonic()
    print(f"[pa-correction] {len(selected)} images; generating reference dependencies", flush=True)
    references = run_campaign(phase_plan(plan, "isoster"))
    (control / "reference_summary.json").write_text(json.dumps(asdict(references), indent=2) + "\n")
    if references.total_ok != len(selected):
        raise RuntimeError("Reference dependency failed; AutoProf not started")
    center_checks = []
    for dataset, galaxy in selected:
        new_dir = target / dataset / safe_galaxy_id(galaxy)
        old_dir = sources[dataset, galaxy, "isoster", "ref_default"].parents[3]
        difference = float(np.max(abs(center_for(new_dir) - center_for(old_dir))))
        center_checks.append(dict(dataset=dataset, galaxy=galaxy, max_abs_difference_pix=difference))
        if difference > 1e-8:
            raise RuntimeError(f"Reference center changed: {dataset}/{galaxy}: {difference}")
    (control / "reference_centers.json").write_text(json.dumps(center_checks, indent=2) + "\n")
    print("[pa-correction] reference centers verified; starting AutoProf", flush=True)
    results = run_campaign(phase_plan(plan, "autoprof"))
    audit = []
    for dataset, galaxy in selected:
        galaxy_dir = target / dataset / safe_galaxy_id(galaxy)
        geometry = json.loads((galaxy_dir / "MANIFEST.json").read_text())["initial_geometry"]
        for arm in sorted(ARMS):
            folder = galaxy_dir / "autoprof/arms" / arm
            record = json.loads((folder / "run_record.json").read_text())
            options = json.loads(next((folder / "tmp").glob("*_options.json")).read_text())
            error = (options["ap_isoinit_pa_set"] - isoster_pa_to_autoprof_init(geometry["pa"]) + 90) % 180 - 90
            if not abs(error) < 1e-10:
                raise RuntimeError(f"Incorrect saved PA: {folder}")
            if arm == "fix_center":
                fixed = np.array([options["ap_set_center"][k] for k in ("x", "y")])
                if not np.allclose(fixed, center_for(galaxy_dir), rtol=0, atol=1e-8):
                    raise RuntimeError(f"Fixed center mismatch: {folder}")
            count = 0
            if record["status"] == "ok":
                table = load_profile(folder / "profile.fits")
                valid = np.asarray(table["sma"]) > 0
                for column in ("sma", "intens", "pa", "eps", "x0", "y0"):
                    valid &= np.isfinite(table[column])
                count = int(valid.sum())
                if not count:
                    raise RuntimeError(f"Successful profile has no finite rows: {folder}")
            audit.append(
                dict(
                    dataset=dataset,
                    galaxy=galaxy,
                    arm=arm,
                    status=record["status"],
                    pa_error_deg=error,
                    finite_rows=count,
                    error_msg=record.get("error_msg", ""),
                    record_path=str(folder / "run_record.json"),
                )
            )
    after = {path: digest(Path(path)) for path in before}
    (control / "source_hashes_after.json").write_text(json.dumps(after, indent=2) + "\n")
    if before != after:
        raise RuntimeError("Source bytes changed during campaign")
    (control / "accepted_records.json").write_text(json.dumps(audit, indent=2) + "\n")
    completion = dict(
        elapsed_seconds=time.monotonic() - started,
        summary=asdict(results),
        outcomes=dict(Counter(row["status"] for row in audit)),
        all_source_hashes_unchanged=True,
        all_saved_pa_correct=True,
    )
    (control / "completion.json").write_text(json.dumps(completion, indent=2) + "\n")
    print(f"[pa-correction] complete {completion}", flush=True)


if __name__ == "__main__":
    main()
