"""Retention and actual saved EA reconstruction gates for the population runner."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.table import Table

pytest.importorskip("pandas")
from benchmarks.exhausted.analysis.huang_final import harmonic_metadata, native_model
from isoster.model import build_isoster_model


def test_native_rejects_missing_coefficients_and_preserves_ea_basis():
    table = Table(
        rows=[
            dict(
                sma=float(s),
                intens=100 - s,
                eps=0.45,
                pa=0.3,
                x0=30.0,
                y0=30.0,
                a3=0.4 / s,
                b3=0.0,
                a4=0.0,
                b4=0.8 / s,
                use_eccentric_anomaly=True,
            )
            for s in range(1, 25)
        ]
    )
    row = SimpleNamespace(tool="isoster")
    model = native_model(table, row, (61, 61), Path)
    assert harmonic_metadata(table, "isoster") == ("ea", "3,4")
    records = [{k: r[k] for k in table.colnames} for r in table]
    expected = build_isoster_model((61, 61), records, fill=np.nan, harmonic_orders=[3, 4], use_eccentric_anomaly=True)
    polar = build_isoster_model((61, 61), records, fill=np.nan, harmonic_orders=[3, 4], use_eccentric_anomaly=False)
    np.testing.assert_allclose(model, expected, equal_nan=True)
    assert np.nanmax(abs(model - polar)) > 0.1
    table["a3"][3] = np.nan
    with pytest.raises(ValueError, match="missing_native_coefficients"):
        native_model(table, row, (61, 61), Path)


def test_primary_wins_require_complete_triples_and_preserve_ties():
    import pandas as pd

    from benchmarks.exhausted.analysis.huang_final_summary import matched_wins

    rows = []
    for galaxy, values in [("tie", [1.0, 1.0, 2.0]), ("missing", [1.0, np.nan, 2.0])]:
        for tool, value in zip(("isoster", "photutils", "autoprof"), values):
            rows.append(
                dict(
                    galaxy=galaxy,
                    scenario="wide",
                    primary=True,
                    tool=tool,
                    mode="native",
                    zone="all",
                    truth_relative_rms=value,
                    flux_bias_abs=value,
                    data_sigma_rms=value,
                )
            )
    summary = matched_wins(pd.DataFrame(rows))
    subset = summary[summary.scenario.eq("ALL") & summary.metric.eq("truth_relative_rms")].set_index("tool")
    assert subset.denominator.eq(1).all()
    assert subset.eligible_inputs.eq(2).all()
    assert subset.sole_wins.eq(0).all()
    assert subset.loc["isoster", "tied_best"] == subset.loc["photutils", "tied_best"] == 1
