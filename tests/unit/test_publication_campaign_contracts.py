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


@pytest.mark.parametrize(
    ("config_name", "campaign_name", "dataset_name", "expected_fits"),
    [
        (
            "campaign.publication_huang_recovery_ngc1209.yaml",
            "publication_single_band_huang_recovery_ngc1209_2026_09_08",
            "huang2013",
            5,
        ),
        (
            "campaign.publication_huang_recovery_ngc3585.yaml",
            "publication_single_band_huang_recovery_ngc3585_2026_09_08",
            "huang2013",
            7,
        ),
        (
            "campaign.publication_s4g_2026_09_08.yaml",
            "publication_single_band_s4g_2026_09_08",
            "s4g",
            18_000,
        ),
    ],
)
def test_recovery_campaigns_use_new_directories_and_exact_fit_counts(
    config_name: str,
    campaign_name: str,
    dataset_name: str,
    expected_fits: int,
) -> None:
    config_path = Path("benchmarks/exhausted/configs") / config_name
    plan = load_campaign(config_path)
    dataset = plan.datasets[dataset_name]
    enabled_arm_count = sum(len(tool.arms) for tool in plan.tools.values() if tool.enabled)
    galaxy_ids = dataset.adapter.list_galaxies()
    if dataset.select:
        galaxy_ids = [galaxy_id for galaxy_id in galaxy_ids if galaxy_id in set(dataset.select)]

    assert plan.campaign_name == campaign_name
    assert plan.campaign_name != "publication_single_band_round1"
    assert plan.execution["skip_existing"] is True
    assert len(galaxy_ids) * enabled_arm_count == expected_fits
