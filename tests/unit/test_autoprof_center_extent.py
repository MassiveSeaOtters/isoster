"""Check diagnostic definitions without running either fitter."""

import numpy as np
from astropy.table import Table

from benchmarks.exhausted.analysis.autoprof_center_extent import profile_measurements


def test_failed_outer_row_is_retained_and_inner_cut_is_explicit():
    table = Table(
        dict(
            sma=[0.0, 1.0, 2.0, 4.0, 8.0],
            x0=[99.0, 99.0, 3.0, 0.0, 6.0],
            y0=[0.0, 0.0, 4.0, 0.0, 8.0],
            stop_code=[0, 0, 0, 2, -1],
        )
    )
    result = profile_measurements(table, dict(x0=0.0, y0=0.0))
    assert result["last_profile_radius"] == 8
    assert result["last_converged_radius"] == 2
    assert result["last_propagation_accepted_radius"] == 4
    assert result["center_median_pix"] == 5
    assert result["center_last_pix"] == 10
    table["stop_code"] = -1
    result = profile_measurements(table, dict(x0=0.0, y0=0.0))
    assert np.isnan(result["last_converged_radius"])
    assert np.isnan(result["last_propagation_accepted_radius"])
