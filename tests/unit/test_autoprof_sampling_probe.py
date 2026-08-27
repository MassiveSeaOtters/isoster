"""The sampling probe must be reusable by the persistent AutoProf worker."""

from types import SimpleNamespace

from benchmarks.harmonic_scale import autoprof_worker


def test_reinstalling_the_probe_replaces_instead_of_nesting(monkeypatch):
    extract_module = SimpleNamespace(
        _iso_between=lambda image, low, high: None,
        _iso_extract=lambda image, sma: None,
    )
    shared_module = SimpleNamespace(
        interpolate_Lanczos=lambda *args: None,
        interpolate_bicubic=lambda *args: None,
    )
    modules = {
        "autoprof.pipeline_steps.Isophote_Extract": extract_module,
        "autoprof.autoprofutils.SharedFunctions": shared_module,
    }
    monkeypatch.setattr(autoprof_worker.importlib, "import_module", modules.__getitem__)
    monkeypatch.setattr(autoprof_worker, "_ORIGINAL_PROBE_TARGETS", {})

    old_events = autoprof_worker._install_sampling_mode_probe()
    new_events = autoprof_worker._install_sampling_mode_probe()
    extract_module._iso_between(None, 0.0, 1.0)
    extract_module._iso_extract(None, 2.0)
    shared_module.interpolate_Lanczos()

    assert old_events == {"band_sampling_calls": 0, "total_calls": 0, "interpolator_calls": 0, "extractions": []}
    assert new_events["total_calls"] == 1
    assert new_events["interpolator_calls"] == 1
    assert len(new_events["extractions"]) == 1
