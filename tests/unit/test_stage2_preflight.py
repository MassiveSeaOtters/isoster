"""Stage 2 preflight: the gate that refuses to time a contaminated machine.

The decision logic is tested rather than the sampling, so it runs in CI where
there is no quiet machine to sample.
"""

from __future__ import annotations

from benchmarks.timing.accuracy_thresholds import CONTAMINATION
from benchmarks.timing.preflight import evaluate

QUIET = {"median": 1.2, "min": 1.0, "max": 1.5}
NO_THERMAL = {"warnings_recorded": False, "detail": []}


def test_a_quiet_idle_machine_is_fit():
    assert evaluate(QUIET, [], NO_THERMAL) == []


def test_load_above_the_frozen_bound_refuses():
    loud = {**QUIET, "median": float(CONTAMINATION["baseline_median_max"]) + 0.1}
    problems = evaluate(loud, [], NO_THERMAL)
    assert any("exceeds the frozen bound" in p for p in problems)


def test_a_competing_agent_session_refuses_even_when_load_looks_quiet():
    """A quiet average does not make an agent session absent; it makes it
    briefly idle. The clause excludes the work, not merely its instantaneous
    cost."""
    problems = evaluate(QUIET, ["claude (34.4%)"], NO_THERMAL)
    assert any("disqualifies a baseline" in p for p in problems)


def test_a_thermal_warning_refuses():
    thermal = {"warnings_recorded": True, "detail": ["CPU_Speed_Limit = 80"]}
    assert any("thermal" in p for p in evaluate(QUIET, [], thermal))


def test_every_reason_is_reported_not_just_the_first():
    loud = {**QUIET, "median": 99.0}
    thermal = {"warnings_recorded": True, "detail": ["x"]}
    assert len(evaluate(loud, ["claude (34.4%)"], thermal)) == 3


def test_the_bound_comes_from_the_frozen_contract():
    """The preflight must not carry its own copy of the limit."""
    at_limit = {**QUIET, "median": float(CONTAMINATION["baseline_median_max"])}
    assert evaluate(at_limit, [], NO_THERMAL) == []
