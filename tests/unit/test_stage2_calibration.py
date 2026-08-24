"""Small contract check for the Stage 2 calibration runner."""

from benchmarks.timing.accuracy_thresholds import accuracy_family_members
from benchmarks.timing.run_stage2_calibration import (
    _assess_sample,
    _base_record,
    _set_accuracy_outcomes,
    _wait_for_clean_retry,
)
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


def test_load_must_exceed_the_limit_twice_but_external_signals_abort_immediately():
    high_load = {"load_limit_exceeded": True, "contaminated": False}
    count, contaminated = _assess_sample(high_load, 0)
    assert (count, contaminated) == (1, False)
    assert _assess_sample(high_load, count) == (2, True)
    assert _assess_sample({"load_limit_exceeded": False, "contaminated": True}, count) == (0, True)


def test_retry_waits_for_the_lagged_load_average_to_recover(monkeypatch):
    samples = iter(
        [
            {"load": 5.0, "load_limit_exceeded": True, "contaminated": False},
            {"load": 4.0, "load_limit_exceeded": False, "contaminated": False},
        ]
    )
    sleeps = []
    monkeypatch.setattr("benchmarks.timing.run_stage2_calibration._indicator_sample", lambda limit: next(samples))
    monkeypatch.setattr("benchmarks.timing.run_stage2_calibration.time.sleep", sleeps.append)
    trace = [{"load": 5.0, "load_limit_exceeded": True, "contaminated": False}]

    _wait_for_clean_retry(trace, 4.745)

    assert [sample["load"] for sample in trace] == [5.0, 5.0, 4.0]
    assert sleeps == [10.0, 10.0]
