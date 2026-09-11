"""Pin the campaign's input convention independently of its output parser."""

from pathlib import Path

import numpy as np
import pytest

from benchmarks.exhausted.adapters.base import GalaxyBundle, GalaxyMetadata
from benchmarks.exhausted.fitters.autoprof_fitter import _build_options, _parse_prof_file


@pytest.mark.parametrize("degrees", [-270, -180, -95.197957, -90, -5.197957, 0, 30, 90, 179.9, 180, 270])
def test_campaign_pa_round_trip_and_unchanged_options(degrees, tmp_path):
    bundle = GalaxyBundle(
        metadata=GalaxyMetadata("test", "test", 0.168, 27.0),
        image=np.ones((21, 21)),
        variance=None,
        mask=None,
        initial_geometry=dict(x0=10.0, y0=10.0, eps=0.4, pa=np.deg2rad(degrees)),
    )
    options = _build_options(
        bundle=bundle,
        arm_delta={"ap_fit_limit": 1.0},
        image_path=Path("input.fits"),
        mask_path=None,
        save_dir=str(tmp_path),
        galaxy_tag="test",
        center_override={"x": 9.8, "y": 10.2},
    )
    # Installed AutoProf's PA_shift_convention: (input - 90 degrees) modulo 180.
    internal = (options["ap_isoinit_pa_set"] - 90) % 180
    difference = (internal - degrees + 90) % 180 - 90
    assert difference == pytest.approx(0, abs=1e-12)
    assert options["ap_isoinit_ellip_set"] == 0.4
    assert options["ap_guess_center"] == {"x": 10.0, "y": 10.0}
    assert options["ap_set_center"] == {"x": 9.8, "y": 10.2}
    assert options["ap_fit_limit"] == 1.0
    assert options["ap_isoclip"] is True


def test_native_output_pa_is_still_converted_once(tmp_path):
    path = tmp_path / "test.prof"
    path.write_text("# units\nR,SB,SB_e,ellip,ellip_e,pa,pa_e\n1,20,0.1,0.4,0.01,120,1\n")
    rows, count, filtered = _parse_prof_file(path, pixel_scale_arcsec=0.168, sb_zeropoint=27.0)
    assert count == 1 and filtered == 0
    assert rows[0]["pa"] == pytest.approx(np.deg2rad(30))


def test_reference_phase_does_not_mutate_full_campaign():
    from benchmarks.exhausted.campaigns.run_autoprof_pa_correction import phase_plan
    from benchmarks.exhausted.orchestrator.config_loader import load_campaign

    plan = load_campaign("benchmarks/exhausted/configs/campaign.publication_autoprof_pa_gate_2026_09_11.yaml")
    reference = phase_plan(plan, "isoster")
    assert reference.tools["isoster"].enabled
    assert not reference.tools["autoprof"].enabled
    assert not reference.raw["tools"]["autoprof"]["enabled"]
    assert plan.tools["autoprof"].enabled and plan.raw["tools"]["autoprof"]["enabled"]
