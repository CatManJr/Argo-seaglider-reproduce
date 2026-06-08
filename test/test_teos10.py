"""
test/test_teos10.py — Pytest tests for utils/teos10.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from utils.teos10 import (
    conservative_temperature,
    absolute_salinity,
    potential_density,
    spice,
    brunt_vaisala_frequency,
    log_brunt_vaisala,
    oxygen_saturation,
    compute_all_teos10,
)


class TestConversions:
    """Unit tests for individual TEOS-10 conversion functions."""

    def test_conservative_temperature_range(self):
        """CT should be close to in-situ T for typical ocean ranges."""
        t, sp, p = 5.0, 34.5, 200.0
        ct = conservative_temperature(t, sp, p)
        assert abs(ct - t) < 1.0, f"CT={ct} too far from T={t}"

    def test_conservative_temperature_cold(self):
        """CT for cold Southern Ocean water."""
        ct = conservative_temperature(0.0, 34.0, 500.0)
        assert -2.0 < ct < 3.0, f"CT={ct} out of expected Southern Ocean range"

    def test_absolute_salinity_near_practical(self):
        """SA ~ SP for open ocean."""
        sa = absolute_salinity(34.5, 100.0, 40.0, -55.0)
        assert 34.0 < sa < 35.5, f"SA={sa} out of range"

    def test_potential_density_positive(self):
        """σ₀ must be positive and realistic for seawater."""
        ct, sa = 2.0, 34.5
        s0 = potential_density(sa, ct)
        assert 25.0 < s0 < 29.0, f"sigma0={s0} out of Southern Ocean range"

    def test_spice_finite(self):
        """Spice should be finite."""
        tau = spice(34.5, 2.0)
        assert np.isfinite(tau)
        assert -5.0 < tau < 5.0

    def test_oxygen_saturation_percent(self):
        """O₂ saturation should be 0-150% range."""
        o2sat = oxygen_saturation(34.5, 2.0, 100.0, 300.0)
        assert 0 < o2sat < 200, f"O2sat={o2sat}%"

    def test_brunt_vaisala_stable_stratification(self):
        """N² > 0 for stable stratification (needs smooth high-res profile)."""
        p = np.linspace(10, 500, 50)
        sa = 34.0 + 0.002 * p
        ct = 5.0 - 0.005 * p
        n2 = brunt_vaisala_frequency(sa, ct, p)
        valid = n2[~np.isnan(n2)]
        if len(valid) > 0:
            positive = valid[valid > 0]
            assert len(positive) / len(valid) > 0.7, (
                f"Only {len(positive)}/{len(valid)} positive"
            )

    def test_log_n2_positive_only(self):
        """log₁₀(N²) only defined for N² > 0."""
        n2 = np.array([1e-5, 1e-4, 0, -1e-6, np.nan, 1e-3])
        logn2 = log_brunt_vaisala(n2)
        assert not np.isnan(logn2[0])
        assert not np.isnan(logn2[1])
        assert np.isnan(logn2[2])
        assert np.isnan(logn2[3])
        assert np.isnan(logn2[4])
        assert not np.isnan(logn2[5])

    def test_log_n2_values(self):
        """log₁₀(1e-4) = -4, log₁₀(1e-3) = -3."""
        logn2 = log_brunt_vaisala(np.array([1e-4, 1e-3]))
        assert abs(logn2[0] - (-4.0)) < 0.01
        assert abs(logn2[1] - (-3.0)) < 0.01


class TestComputeAll:
    """Integration test for compute_all_teos10()."""

    @pytest.fixture
    def sample_df(self):
        import pandas as pd
        n = 30
        p = np.linspace(10, 500, n)
        return pd.DataFrame({
            "temp": 2.0 - 0.003 * p,
            "psal": 34.0 + 0.001 * p,
            "pressure": p,
            "oxygen": 300 - 0.05 * p,
            "lon": np.full(n, 40.0),
            "lat": np.full(n, -55.0),
        })

    def test_output_columns(self, sample_df):
        result = compute_all_teos10(sample_df)
        expected = ["CT", "SA", "sigma0", "spice", "N2", "logN2"]
        for col in expected:
            assert col in result.columns, f"Missing {col}"

    def test_no_nan_in_computed(self, sample_df):
        result = compute_all_teos10(sample_df)
        for col in ["CT", "SA", "sigma0", "spice"]:
            assert not result[col].isna().any(), f"{col} has NaN"

    def test_idempotent(self, sample_df):
        """Running twice should not duplicate columns."""
        r1 = compute_all_teos10(sample_df)
        r2 = compute_all_teos10(r1)  # should skip
        assert len(r1.columns) == len(r2.columns)

    def test_sigma0_monotonic(self, sample_df):
        """σ₀ should increase with depth (stable stratification)."""
        result = compute_all_teos10(sample_df)
        diffs = np.diff(result["sigma0"].values)
        assert np.all(diffs > -0.5), "σ₀ not increasing with depth"
