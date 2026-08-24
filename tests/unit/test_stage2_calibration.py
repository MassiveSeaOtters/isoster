"""Small contract check for the Stage 2 calibration runner."""

from benchmarks.timing.accuracy_thresholds import accuracy_family_members
from benchmarks.timing.run_stage2_calibration import _base_record, _set_accuracy_outcomes
from benchmarks.timing.stage1_fixtures import fixed_aperture_radii


def test_noisy_arm_uses_all_realizations_and_derives_headline_eligibility():
    fixture = "sersic_n2_compact"
    members = accuracy_family_members(fixed_aperture_radii(fixture), harmonics_enabled=False, geometry_free=False)[
        "intensity"
    ]
    records = []
    for realization_index in range(25):
        record = _base_record(
            "fixed_aperture",
            "isoster",
            fixture,
            "gaussian_reference",
            False,
            0,
            realization_index,
        )
        record.update(
            {
                "execution_status": "ok",
                "coverage_status": "complete",
                "accuracy_residuals": {"intensity": {member: [0.0] for member in members}},
            }
        )
        records.append(record)

    assert _set_accuracy_outcomes(records) == []
    assert all(record["intensity_accuracy_status"] == "pass" for record in records)
    assert all(record["harmonic_accuracy_status"] == "not_applicable" for record in records)
    assert all(record["geometry_accuracy_status"] == "not_applicable" for record in records)
    assert all(record["headline_eligible"] is True for record in records)
