"""Part B, Stage 1: the six fixtures and their frozen radial grids.

Part A's two galaxies are reused verbatim from ``FIXTURES`` so their archived
accuracy instrumentation applies unchanged. The four new ones are defined here
because Part B needs them and Part A never did --- an earlier draft named them
in prose without definitions or radii, so no accuracy bar could be computed for
four of six fixtures and Stage 2 could not have checked most of its arms.

The size ladder scales ``R_e`` and the radial grid with the canvas, so ring
count and samples-per-ring grow together and the extraction workload actually
changes. ``wide_canvas_961`` deliberately does not: it holds the galaxy fixed
and grows only the image, which is why it is confined to the end-to-end scope
where whole-image overhead is legitimately part of the task.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.harmonic_scale.run_harmonic_scale import FIXTURES as PART_A_FIXTURES  # noqa: E402
from benchmarks.harmonic_scale.run_harmonic_scale import PLANTED_HARMONICS  # noqa: E402

#: Radii as fractions of R_e, shared by every Part B fixture so the ladder
#: measures size rather than a change of sampling strategy. Part A's own radii
#: are kept for its two galaxies, which is why they are not generated here.
RADIUS_FRACTIONS = (0.5, 0.75, 1.0, 1.4, 1.8)

REFERENCE_SNR = 100.0
ENSEMBLE_REALIZATIONS = 25
SEED_BLOCKS = {"calibration": 100_000, "campaign": 60_260_822}
NOISE_ARMS = {
    "noiseless": {"distribution": "none", "snr_at_r_e": None, "realizations": 1},
    "gaussian_reference": {
        "distribution": "independent_gaussian",
        "generator": "numpy.random.Generator(PCG64).normal",
        "mean": 0.0,
        "snr_at_r_e": REFERENCE_SNR,
        "sigma": "I_e / snr_at_r_e",
        "realizations": ENSEMBLE_REALIZATIONS,
    },
}


def _ladder(name: str, r_e: float, shape: int, scope: str) -> Dict[str, object]:
    center = (shape - 1) / 2.0
    return {
        "label": name.replace("_", " "),
        "galaxy": {"n": 2.0, "R_e": r_e, "I_e": 100.0, "shape": (shape, shape), "center": (center, center)},
        "reference_eps": 0.3,
        "reference_pa": 0.0,
        "reference_harmonics": dict(PLANTED_HARMONICS),
        "radii": tuple(round(r_e * f, 3) for f in RADIUS_FRACTIONS),
        "scope": scope,
        "scientific_identity": name,
        "independent_scientific_fixture": True,
    }


def stage1_fixtures() -> Dict[str, Dict[str, object]]:
    """Every Part B fixture, Part A's two first."""
    fixtures: Dict[str, Dict[str, object]] = {}
    for name in ("sersic_n2_compact", "sersic_n4_extended"):
        spec = dict(PART_A_FIXTURES[name])
        spec["scope"] = "both"
        spec["reference_pa"] = 0.0
        spec["reference_harmonics"] = dict(PLANTED_HARMONICS)
        spec["scientific_identity"] = name
        spec["independent_scientific_fixture"] = True
        fixtures[name] = spec
    fixtures["size_ladder_481"] = _ladder("size_ladder_481", 50.0, 481, "both")
    fixtures["size_ladder_961"] = _ladder("size_ladder_961", 100.0, 961, "both")
    fixtures["size_ladder_1921"] = _ladder("size_ladder_1921", 200.0, 1921, "both")
    # Galaxy identical to sersic_n2_compact; only the canvas grows.
    wide = _ladder("wide_canvas_961", 25.0, 961, "end_to_end")
    wide["radii"] = PART_A_FIXTURES["sersic_n2_compact"]["radii"]
    wide["scientific_identity"] = "sersic_n2_compact"
    wide["independent_scientific_fixture"] = False
    fixtures["wide_canvas_961"] = wide
    return fixtures


def fixed_aperture_radii(name: str) -> List[float]:
    return [float(r) for r in stage1_fixtures()[name]["radii"]]


def stage1_seed(stage: str, realization_index: int) -> int:
    """Derive one seed from the frozen, disjoint calibration/campaign blocks."""
    if stage not in SEED_BLOCKS:
        raise ValueError(f"stage must be one of {sorted(SEED_BLOCKS)}, got {stage!r}")
    if not isinstance(realization_index, int) or not 0 <= realization_index < ENSEMBLE_REALIZATIONS:
        raise ValueError(f"realization_index must lie in [0, {ENSEMBLE_REALIZATIONS}), got {realization_index!r}")
    return SEED_BLOCKS[stage] + realization_index


def render_stage1_fixture(
    fixture: str,
    noise_arm: str,
    *,
    stage: str = "campaign",
    realization_index: int = 0,
):
    """Render exactly the scientific image Stage 2 and Stage 4 must use.

    Returns ``(image, variance, metadata)``. The noiseless arm has no variance
    map; the Gaussian arm has independent, constant-variance pixels generated
    by NumPy's explicitly recorded PCG64 generator.
    """
    from benchmarks.utils.sersic_model import create_sersic_image_with_harmonics

    try:
        spec = stage1_fixtures()[fixture]
    except KeyError as error:
        raise ValueError(f"unknown Stage 1 fixture {fixture!r}") from error
    if noise_arm not in NOISE_ARMS:
        raise ValueError(f"noise_arm must be one of {sorted(NOISE_ARMS)}, got {noise_arm!r}")
    galaxy = spec["galaxy"]
    image, truth = create_sersic_image_with_harmonics(
        n=galaxy["n"],
        R_e=galaxy["R_e"],
        I_e=galaxy["I_e"],
        eps=spec["reference_eps"],
        pa=spec["reference_pa"],
        shape=galaxy["shape"],
        center=galaxy["center"],
        harmonics=spec["reference_harmonics"],
    )
    if noise_arm == "noiseless":
        return image, None, {"noise_arm": noise_arm, "seed": None, "noise_sigma": 0.0, "truth": truth}

    seed = stage1_seed(stage, realization_index)
    sigma = float(galaxy["I_e"]) / float(NOISE_ARMS[noise_arm]["snr_at_r_e"])
    generator = np.random.Generator(np.random.PCG64(seed))
    noisy = image + generator.normal(0.0, sigma, size=image.shape)
    variance = np.full(image.shape, sigma**2, dtype=np.float64)
    return noisy, variance, {"noise_arm": noise_arm, "seed": seed, "noise_sigma": sigma, "truth": truth}
