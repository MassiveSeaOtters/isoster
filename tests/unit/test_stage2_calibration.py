"""Small contract check for the Stage 2 calibration runner."""

from benchmarks.timing.accuracy_thresholds import accuracy_family_members
from benchmarks.timing.recover_stage2_autoprof import _merge_records, _select_failed_autoprof_records
from benchmarks.timing.run_stage2_calibration import (
    THREAD_LIMITS,
    _base_record,
    _indicator_sample,
    _request_id,
    _session_environment,
    _set_accuracy_outcomes,
    _wait_for_clean_retry,
)
from benchmarks.timing.run_stage4_campaign import _set_timing_eligibility, _timing_summaries
from benchmarks.timing.stage1_fixtures import fixed_aperture_radii
from benchmarks.timing.stage3_parameters import load_stage3_parameters


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


def test_in_session_load_is_recorded_but_does_not_abort(monkeypatch):
    monkeypatch.setattr("benchmarks.timing.run_stage2_calibration.load_average", lambda: 7.33)
    monkeypatch.setattr(
        "benchmarks.timing.run_stage2_calibration.thermal_warnings",
        lambda: {"warnings_recorded": False},
    )
    monkeypatch.setattr("benchmarks.timing.run_stage2_calibration.competing_processes", lambda: [])
    sample = _indicator_sample()
    assert sample["load"] == 7.33
    assert sample["contaminated"] is False


def test_retry_waits_for_thermal_or_process_contamination_to_clear(monkeypatch):
    samples = iter(
        [
            {"load": 5.0, "contaminated": True},
            {"load": 4.0, "contaminated": False},
        ]
    )
    sleeps = []
    monkeypatch.setattr("benchmarks.timing.run_stage2_calibration._indicator_sample", lambda: next(samples))
    monkeypatch.setattr("benchmarks.timing.run_stage2_calibration.time.sleep", sleeps.append)
    trace = [{"load": 5.0, "contaminated": True}]

    _wait_for_clean_retry(trace)

    assert [sample["load"] for sample in trace] == [5.0, 5.0, 4.0]
    assert sleeps == [10.0, 10.0]


def test_session_environment_overrides_numerical_library_thread_defaults(monkeypatch):
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "20")
    environment = _session_environment()
    assert {name: environment[name] for name in THREAD_LIMITS} == THREAD_LIMITS


def test_recovery_selects_and_replaces_only_failed_autoprof_records(tmp_path):
    failed = _base_record("end_to_end", "autoprof", "fixture", "noise", True, 0, 2)
    failed["error"] = "BrokenPipeError: worker exited"
    successful = _base_record("end_to_end", "isoster", "fixture", "noise", True, 0, 2)
    successful["execution_status"] = "ok"
    selected = _select_failed_autoprof_records([failed, successful])
    replacement = {**failed, "execution_status": "ok", "fit_only_s": 1.0}

    merged = _merge_records([failed, successful], [replacement], selected, tmp_path / "source.json")

    assert set(selected) == {_request_id(failed)}
    assert merged[0]["execution_status"] == "ok"
    assert merged[0]["recovery"]["original_error"] == "BrokenPipeError: worker exited"
    assert merged[1] is successful


def test_frozen_stage3_parameters_cover_every_arm():
    parameters = load_stage3_parameters()

    assert len(parameters["calls_per_batch_by_arm"]) == 132
    assert parameters["sessions"] == 3
    assert parameters["repetitions_per_session"] == 25


def test_unavailable_autoprof_mean_is_descriptive_not_an_evaluation_error():
    record = _base_record("fixed_aperture", "autoprof", "fixture", "noiseless", False, 0, 0)
    record.update(
        {
            "execution_status": "ok",
            "coverage_status": "complete",
            "intensity_accuracy_availability": "unavailable",
        }
    )

    assert _set_accuracy_outcomes([record]) == []
    assert record["intensity_accuracy_status"] == "fail"
    assert record["headline_eligible"] is False


def test_stage4_timing_eligibility_and_session_interval_are_separate_from_accuracy():
    records = []
    for session_index, elapsed in enumerate((1.0, 1.2, 0.9)):
        record = _base_record("end_to_end", "isoster", "fixture", "noiseless", False, session_index, 0)
        record.update(
            {
                "execution_status": "ok",
                "coverage_status": "complete",
                "fit_only_s": elapsed,
                "fit_plus_harness_s": elapsed,
                "calls_per_batch": 1,
                "ring_count": 10,
                "coverage_fraction": 1.0,
            }
        )
        records.append(record)

    _set_timing_eligibility(records)
    summary = next(iter(_timing_summaries(records).values()))

    assert all(record["timing_eligible"] for record in records)
    assert summary["session_median_interval_s"] == [0.9, 1.2]
