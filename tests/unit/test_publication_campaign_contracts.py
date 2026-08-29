from pathlib import Path

import pytest
import yaml
from astropy.io import fits

from benchmarks.exhausted.adapters.huang2013_scenarios import (
    Huang2013ScenariosAdapter,
)
from benchmarks.exhausted.orchestrator.config_loader import load_campaign


def test_noiseless_scenario_is_discovered_and_loaded(tmp_path: Path) -> None:
    galaxy_dir = tmp_path / "TEST001"
    galaxy_dir.mkdir()
    fits.PrimaryHDU(data=[[1.0, 2.0], [3.0, 4.0]]).writeto(galaxy_dir / "TEST001_noiseless_z005.fits")

    adapter = Huang2013ScenariosAdapter(tmp_path, depths=["noiseless"], redshift_tags=["005"])

    assert adapter.list_galaxies() == ["TEST001/noiseless_z005"]
    bundle = adapter.load_galaxy("TEST001/noiseless_z005")
    assert bundle.metadata.extra["depth"] == "noiseless"


def test_campaign_can_select_existing_arms(tmp_path: Path) -> None:
    campaign_path = tmp_path / "campaign.yaml"
    campaign_path.write_text(
        yaml.safe_dump(
            {
                "campaign_name": "selected",
                "output_root": str(tmp_path / "output"),
                "tools": {
                    "isoster": {
                        "enabled": True,
                        "arms_file": "benchmarks/exhausted/configs/isoster_arms.yaml",
                        "select_arms": ["reg_outer_damp", "ref_default"],
                    }
                },
                "isoster_harmonic_sweeps": [],
                "datasets": {},
            }
        )
    )

    plan = load_campaign(campaign_path)

    assert list(plan.tools["isoster"].arms) == ["reg_outer_damp", "ref_default"]
    assert "select_arms" not in plan.tools["isoster"].extra


def test_campaign_rejects_unknown_selected_arm(tmp_path: Path) -> None:
    campaign_path = tmp_path / "campaign.yaml"
    campaign_path.write_text(
        yaml.safe_dump(
            {
                "campaign_name": "bad",
                "output_root": str(tmp_path / "output"),
                "tools": {
                    "isoster": {
                        "enabled": True,
                        "arms_file": "benchmarks/exhausted/configs/isoster_arms.yaml",
                        "select_arms": ["missing"],
                    }
                },
                "datasets": {},
            }
        )
    )

    with pytest.raises(ValueError, match="unknown arm"):
        load_campaign(campaign_path)
