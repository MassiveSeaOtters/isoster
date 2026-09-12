"""Keep three-tool figure comparisons on identical finite primary samples."""

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from benchmarks.exhausted.plotting.publication_huang import finite_summary, matched_primary


def test_matched_primary_does_not_replace_failed_or_nonfinite_values():
    rows = []
    for galaxy in ("complete", "failed", "nonfinite"):
        for tool in ("isoster", "photutils", "autoprof"):
            rows.append(
                dict(
                    galaxy=galaxy,
                    scenario="wide_z005",
                    tool=tool,
                    primary=True,
                    status="failed" if galaxy == "failed" and tool == "autoprof" else "ok",
                    error=np.inf if galaxy == "nonfinite" and tool == "photutils" else 1.0,
                )
            )
    rows.append(dict(galaxy="failed", scenario="wide_z005", tool="autoprof", primary=False, status="ok", error=0.0))
    matched = matched_primary(pd.DataFrame(rows), "error")
    assert list(matched.index) == [("complete", "wide_z005")]
    assert matched.shape == (1, 3)
    assert finite_summary([1, np.nan, np.inf, 3])[0:3:2] == (2, 2)
