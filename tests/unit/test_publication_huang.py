"""Small deterministic checks for publication measurement conventions."""

import numpy as np
import pytest
from astropy.table import Table

from benchmarks.exhausted.analysis.publication_huang import pixel_metrics, replace_autoprof_records, ring_truth


def test_missing_support_and_noise_are_not_success_scores():
    truth = np.ones((3, 3))
    missing = pixel_metrics(truth, truth, truth, np.zeros_like(truth, dtype=bool), np.nan)
    assert missing["npix"] == 0 and np.isnan(missing["truth_relative_rms"])
    result = pixel_metrics(2 * truth, truth, truth, np.ones_like(truth, dtype=bool), np.nan)
    assert result["truth_relative_rms"] == 1 and result["flux_bias"] == 1
    assert np.isnan(result["data_sigma_rms"])


def test_ring_coordinate_bases_and_statistics():
    y, x = np.mgrid[:101, :101]
    truth = (x - 50) ** 2 + (y - 50) ** 2
    table = Table(dict(sma=[20.0], eps=[0.5], pa=[0.0], x0=[50.0], y0=[50.0], intens=[250.0]))
    ea = ring_truth(table, truth, True, False, 1, 2048)
    phi = ring_truth(table, truth, False, False, 1, 2048)
    assert ea["ring_n"] == 1 and ea["ring_relative_rms"] < 0.002
    assert phi["ring_relative_rms"] > 0.2
    outside = ring_truth(table, truth, True, True, 30)
    assert outside["ring_n"] == 0 and np.isnan(outside["ring_relative_rms"])


def test_correction_retains_failures_and_requires_complete_unique_roster():
    key = ("galaxy", "wide_z005", "autoprof", "baseline")
    original = dict(zip(("galaxy", "scenario", "tool", "arm"), key)) | {"status": "ok"}
    corrected = original | {"status": "failed"}
    reference_key = ("galaxy", "wide_z005", "isoster", "ref_default")
    selected = {key: original, reference_key: {"status": "ok"}}
    result, excluded = replace_autoprof_records(selected, [corrected])
    assert result[key]["status"] == "failed"
    assert selected[key]["status"] == "ok"
    assert len(excluded) == 1 and result[reference_key] == {"status": "ok"}
    with pytest.raises(ValueError, match="incomplete"):
        replace_autoprof_records(selected, [])
    with pytest.raises(ValueError, match="Duplicate"):
        replace_autoprof_records(selected, [corrected, corrected])
