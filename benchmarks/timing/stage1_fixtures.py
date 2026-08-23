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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.harmonic_scale.run_harmonic_scale import FIXTURES as PART_A_FIXTURES  # noqa: E402

#: Radii as fractions of R_e, shared by every Part B fixture so the ladder
#: measures size rather than a change of sampling strategy. Part A's own radii
#: are kept for its two galaxies, which is why they are not generated here.
RADIUS_FRACTIONS = (0.5, 0.75, 1.0, 1.4, 1.8)


def _ladder(name: str, r_e: float, shape: int, scope: str) -> Dict[str, object]:
    center = (shape - 1) / 2.0
    return {
        "label": name.replace("_", " "),
        "galaxy": {"n": 2.0, "R_e": r_e, "I_e": 100.0, "shape": (shape, shape), "center": (center, center)},
        "reference_eps": 0.3,
        "radii": tuple(round(r_e * f, 3) for f in RADIUS_FRACTIONS),
        "scope": scope,
    }


def stage1_fixtures() -> Dict[str, Dict[str, object]]:
    """Every Part B fixture, Part A's two first."""
    fixtures: Dict[str, Dict[str, object]] = {}
    for name in ("sersic_n2_compact", "sersic_n4_extended"):
        spec = dict(PART_A_FIXTURES[name])
        spec["scope"] = "both"
        fixtures[name] = spec
    fixtures["size_ladder_481"] = _ladder("size_ladder_481", 50.0, 481, "both")
    fixtures["size_ladder_961"] = _ladder("size_ladder_961", 100.0, 961, "both")
    fixtures["size_ladder_1921"] = _ladder("size_ladder_1921", 200.0, 1921, "both")
    # Galaxy identical to sersic_n2_compact; only the canvas grows.
    wide = _ladder("wide_canvas_961", 25.0, 961, "end_to_end")
    wide["radii"] = PART_A_FIXTURES["sersic_n2_compact"]["radii"]
    fixtures["wide_canvas_961"] = wide
    return fixtures


def fixed_aperture_radii(name: str) -> List[float]:
    return [float(r) for r in stage1_fixtures()[name]["radii"]]
