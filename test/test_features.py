"""
test/test_features.py — Pytest tests for utils/features.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest
from utils.features import (
    RFR_FEATURES,
    RFR_TARGET,
    seasonal_encode,
    datetime_to_yearday,
    select_rfr_features,
    get_feature_info,
)


class TestSeasonalEncoding:
    """Cyclical encoding sz₁=cos(2π·yd/365), sz₂=sin(2π·yd/365)."""

    def test_jan1(self):
        """Jan 1 (yd=0): sz₁=1, sz₂=0."""
        sz1, sz2 = seasonal_encode(0)
        assert abs(sz1 - 1.0) < 1e-10
        assert abs(sz2 - 0.0) < 1e-10

    def test_jul2(self):
        """Jul 2 (yd=182): sz₁≈-1, sz₂=sin(364π/365)≈0.0086."""
        sz1, sz2 = seasonal_encode(182)
        assert abs(sz1 - (-1.0)) < 0.002
        # sin(2π·182/365) = sin(π − π/365) ≈ π/365 ≈ 0.0086
        assert abs(sz2 - 0.0086) < 0.001

    def test_apr1(self):
        """Apr 1 (yd=90): sz₁≈0, sz₂≈1."""
        sz1, sz2 = seasonal_encode(90)
        assert abs(sz1) < 0.03
        assert abs(sz2 - 1.0) < 0.03

    def test_oct1(self):
        """Oct 1 (yd=273): sz₁≈0, sz₂≈-1."""
        sz1, sz2 = seasonal_encode(273)
        assert abs(sz1) < 0.03
        assert abs(sz2 + 1.0) < 0.03

    def test_periodicity(self):
        """yd and yd+365 should give same encoding."""
        sz1a, sz2a = seasonal_encode(50)
        sz1b, sz2b = seasonal_encode(415)
        assert abs(sz1a - sz1b) < 1e-10
        assert abs(sz2a - sz2b) < 1e-10

    def test_array_input(self):
        """Should accept array inputs."""
        yd = np.array([0, 91, 182, 273])
        sz1, sz2 = seasonal_encode(yd)
        assert len(sz1) == 4
        assert len(sz2) == 4

    def test_range(self):
        """sz₁ and sz₂ must be in [-1, 1]."""
        yd = np.linspace(0, 1000, 500)
        sz1, sz2 = seasonal_encode(yd)
        assert np.all(sz1 >= -1.0001) and np.all(sz1 <= 1.0001)
        assert np.all(sz2 >= -1.0001) and np.all(sz2 <= 1.0001)

    def test_sum_of_squares(self):
        """sz₁² + sz₂² ≈ 1 (unit circle)."""
        yd = np.linspace(0, 365, 100)
        sz1, sz2 = seasonal_encode(yd)
        r2 = sz1 ** 2 + sz2 ** 2
        assert np.allclose(r2, 1.0, atol=1e-10)


class TestDatetimeConversion:
    """Yearday from datetime."""

    def test_jan1_noon(self):
        """Jan 1 12:00 → yd ≈ 0.5."""
        s = pd.Series(["2019-01-01 12:00:00"])
        yd = datetime_to_yearday(s, ref_year=2019)
        assert abs(yd.iloc[0] - 0.5) < 0.01

    def test_jul1(self):
        """Jul 1 → yd ≈ 181."""
        s = pd.Series(["2019-07-01"])
        yd = datetime_to_yearday(s, ref_year=2019)
        assert 180 < yd.iloc[0] < 182

    def test_may31(self):
        """May 31 → yd ≈ 150 (end of paper's deployment)."""
        s = pd.Series(["2019-05-31"])
        yd = datetime_to_yearday(s, ref_year=2019)
        assert 149 < yd.iloc[0] < 151


class TestFeatureSelection:
    """RFR 9-feature selection."""

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "CT": [2.0, 1.8, 1.5],
            "SA": [34.5, 34.6, 34.7],
            "pressure": [10, 50, 100],
            "oxygen": [310, 300, 280],
            "lat": [-55, -55, -55],
            "lon": [40, 40, 40],
            "yearday": [150, 150, 150],
            "nitrate": [26.0, 26.5, 27.0],
            "sigma0": [27.1, 27.2, 27.3],  # screened out
            "spice": [0.1, 0.1, 0.1],       # screened out
        })

    def test_output_shape(self, sample_df):
        X, y = select_rfr_features(sample_df)
        assert X.shape == (3, 9), f"X shape {X.shape} != (3, 9)"
        assert len(y) == 3
        assert y.name == RFR_TARGET

    def test_only_nine_features(self, sample_df):
        X, _ = select_rfr_features(sample_df)
        assert list(X.columns) == RFR_FEATURES

    def test_no_screened_out(self, sample_df):
        X, _ = select_rfr_features(sample_df)
        for col in ["sigma0", "spice", "N2", "O2sat"]:
            assert col not in X.columns, f"{col} should be excluded"

    def test_auto_seasonal_encode(self, sample_df):
        """ydcos/ydsin should be auto-computed from yearday."""
        df = sample_df.drop(columns=["sigma0", "spice"])
        X, _ = select_rfr_features(df)  # No ydcos/ydsin in input
        assert "ydcos" in X.columns
        assert "ydsin" in X.columns

    def test_missing_feature_raises(self, sample_df):
        df = sample_df.drop(columns=["CT"])
        with pytest.raises(KeyError, match="CT"):
            select_rfr_features(df)

    def test_no_target_ok(self, sample_df):
        df = sample_df.drop(columns=["nitrate"])
        X, y = select_rfr_features(df, target_col="nitrate")
        assert X.shape == (3, 9)
        assert y is None


class TestFeatureInfo:
    """Feature roster table."""

    def test_has_all_features(self):
        info = get_feature_info()
        all_features = set(info["feature"])
        assert "CT" in all_features
        assert "sigma0" in all_features  # screened out
        assert "logN2" in all_features   # screened out
        assert len(info) >= 14  # 9 selected + 5 excluded

    def test_selected_count(self):
        info = get_feature_info()
        selected = info[info["role"] == "selected"]
        assert len(selected) == 9
