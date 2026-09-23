"""Keep three-tool figure comparisons on identical finite primary samples."""

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from benchmarks.exhausted.plotting.publication_huang import finite_summary, matched_primary, select_cases


@pytest.mark.parametrize("failure_status", ["failed", "error"])
def test_matched_primary_does_not_replace_failed_or_nonfinite_values(failure_status):
    rows = []
    for galaxy in ("complete", "failed", "nonfinite"):
        for tool in ("isoster", "photutils", "autoprof"):
            rows.append(
                dict(
                    galaxy=galaxy,
                    scenario="wide_z005",
                    tool=tool,
                    primary=True,
                    status=failure_status if galaxy == "failed" and tool == "autoprof" else "ok",
                    error=np.inf if galaxy == "nonfinite" and tool == "photutils" else 1.0,
                )
            )
    rows.append(dict(galaxy="failed", scenario="wide_z005", tool="autoprof", primary=False, status="ok", error=0.0))
    matched = matched_primary(pd.DataFrame(rows), "error")
    assert list(matched.index) == [("complete", "wide_z005")]
    assert matched.shape == (1, 3)
    assert finite_summary([1, np.nan, np.inf, 3])[0:3:2] == (2, 2)


def test_atlas_retains_a_failure_for_each_tool():
    rows = [
        dict(
            galaxy="A",
            scenario="wide_z005",
            tool=tool,
            arm=arm,
            status=status,
            primary=True,
            truth_relative_rms_all=1.0,
            initial_eps=0.2,
            reference_psf=2.0,
            common_support_fraction=0.8,
        )
        for tool, arm, status in (
            ("isoster", "ref_default", "ok"),
            ("photutils", "baseline_median", "failed"),
            ("autoprof", "baseline", "error"),
        )
    ]
    pairs = pd.DataFrame(columns=["galaxy", "scenario", "arm", "reference", "metric", "delta"]).astype({"delta": float})
    cases = select_cases(pd.DataFrame(rows), pairs)
    assert set(cases[cases.reason.str.contains("failure")].reason) == {
        "first retained autoprof primary failure",
        "first retained photutils primary failure",
    }
