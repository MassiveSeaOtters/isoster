"""A5: native and converted harmonics must never share a column name.

This is the whole failure the schema section exists to prevent, and it was
live in the code before this: the AutoProf fitter wrote AutoProf's native
coefficients straight into ``a3``/``b3``/``a4``/``b4``, the same keys the
isoster and photutils fitters fill with Bender-normalized major-axis values.
Two different quantities under one name, indistinguishable once written.

Pure arithmetic on a dict, so it runs in CI without AutoProf.
"""

from __future__ import annotations

import math

import pytest

from benchmarks.exhausted.fitters.autoprof_fitter import (
    HARMONIC_ORDERS,
    PROFILE_SCHEMA_VERSION,
    _harmonic_schema_fields,
)


@pytest.fixture
def native_row():
    """One AutoProf row, with a deliberately asymmetric set of values.

    Distinct magnitudes and mixed signs, so a transposed order or a dropped
    negation shows up as a wrong number rather than as agreement.
    """
    return {
        "autoprof_a3_native": 0.010,
        "autoprof_b3_native": -0.020,
        "autoprof_a4_native": -0.030,
        "autoprof_b4_native": 0.040,
        "autoprof_b0": 50.0,
    }


class TestColumnSeparation:
    def test_the_bare_names_are_never_the_native_values(self, native_row):
        fields = _harmonic_schema_fields(native_row, "polar_from_image_x_axis")
        for order in HARMONIC_ORDERS:
            assert math.isnan(fields[f"a{order}"])
            assert math.isnan(fields[f"b{order}"])

    def test_native_values_survive_untouched_under_their_own_names(self, native_row):
        # The fitter copies these in verbatim; this pins that the schema
        # helper does not overwrite or rescale them.
        fields = _harmonic_schema_fields(native_row, "polar_from_image_x_axis")
        assert "autoprof_a3_native" not in fields, "the helper must not restate native values"
        assert native_row["autoprof_a3_native"] == 0.010

    def test_a_nan_bender_value_carries_a_reason(self, native_row):
        fields = _harmonic_schema_fields(native_row, "polar_from_image_x_axis")
        assert fields["harmonic_conversion_valid"] is False
        assert fields["harmonic_conversion_reason"]
        assert "gradient" in fields["harmonic_conversion_reason"]

    def test_every_row_carries_a_measurement_status(self, native_row):
        fields = _harmonic_schema_fields(native_row, "polar_from_image_x_axis")
        assert fields["harmonic_measurement_status"] == "not_reported_by_tool"


class TestRawReconstruction:
    """``S_n = -2|b0|a_n``, ``C_n = +2|b0|b_n`` --- exact, and sign-sensitive."""

    def test_the_sine_amplitude_is_negated_and_the_cosine_is_not(self, native_row):
        fields = _harmonic_schema_fields(native_row, "polar_from_image_x_axis")
        scale = 2.0 * abs(native_row["autoprof_b0"])
        assert fields["s3_raw_sky"] == pytest.approx(-scale * native_row["autoprof_a3_native"])
        assert fields["c3_raw_sky"] == pytest.approx(scale * native_row["autoprof_b3_native"])
        assert fields["s4_raw_sky"] == pytest.approx(-scale * native_row["autoprof_a4_native"])
        assert fields["c4_raw_sky"] == pytest.approx(scale * native_row["autoprof_b4_native"])

    def test_a_negative_b0_does_not_flip_the_reconstructed_sign(self, native_row):
        # AutoProf divides by |fft(I)[0]|, so the absolute value is the
        # convention. A noisy outer ring can have a non-positive mean, and
        # using the signed value would flip the sign exactly where the data
        # are worst.
        positive = _harmonic_schema_fields(native_row, "polar_from_image_x_axis")
        native_row["autoprof_b0"] = -native_row["autoprof_b0"]
        negative = _harmonic_schema_fields(native_row, "polar_from_image_x_axis")
        assert positive["s3_raw_sky"] == pytest.approx(negative["s3_raw_sky"])

    def test_the_raw_columns_are_labelled_sky_frame(self, native_row):
        # Rotating to the major-axis frame needs a per-ring position angle
        # and is only valid on the polar-resampled path; naming them for the
        # frame they are actually in is what keeps that from being assumed.
        fields = _harmonic_schema_fields(native_row, "polar_from_image_x_axis")
        assert "s3_raw_sky" in fields
        assert "s3_raw_major" not in fields


class TestAngleBasis:
    def test_the_basis_is_recorded_on_every_row(self, native_row):
        for basis in ("polar_from_image_x_axis", "eccentric_anomaly"):
            assert _harmonic_schema_fields(native_row, basis)["harmonic_basis"] == basis

    def test_the_eccentric_anomaly_path_states_the_order_mixing_as_well(self, native_row):
        reason = _harmonic_schema_fields(native_row, "eccentric_anomaly")["harmonic_conversion_reason"]
        assert "mixes_orders" in reason
        # The missing gradient still applies; one reason must not mask the other.
        assert "gradient" in reason

    def test_neither_basis_is_ever_marked_valid(self, native_row):
        # Both blockers are unresolved today. If Track 2 is ever licensed,
        # this test is the one that should be updated deliberately.
        for basis in ("polar_from_image_x_axis", "eccentric_anomaly"):
            assert _harmonic_schema_fields(native_row, basis)["harmonic_conversion_valid"] is False


class TestSchemaVersion:
    def test_the_version_was_bumped_past_the_ambiguous_layout(self):
        # Version 1 files carry an `a3` whose meaning depends on which tool
        # wrote them. Without the bump, old and new files are
        # indistinguishable while meaning different things.
        assert PROFILE_SCHEMA_VERSION >= 2

    def test_the_orders_the_schema_covers_are_declared(self):
        assert tuple(HARMONIC_ORDERS) == (3, 4)
