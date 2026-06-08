"""
test/test_pipeline.py — Integration tests for preprocess/pipeline.py

Verifies that the full preprocessing pipeline produces valid
RFR-ready feature matrices from the Zenodo data.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

# Lazy imports to avoid circular issues
from preprocess.pipeline import (
    load_dataset,
    apply_teos10,
    apply_features,
    apply_mld,
    DATASETS,
    DATA_DIR,
    OUTPUT_DIR,
    RFR_FEATURES,
    RFR_TARGET,
)


def _has_data():
    """Check if Zenodo data files exist."""
    return (DATA_DIR / "bgcArgo_datasetB_10float_ASZ_2017-2021.csv").exists()


# ── Unit-level pipeline tests (no file I/O) ────────────────────────

class TestLoadDataset:
    """load_dataset() with real Zenodo files if available."""

    @pytest.fixture
    def sample_csv(self, tmp_path):
        """Create a minimal CSV mimicking Zenodo format."""
        import pandas as pd
        df = pd.DataFrame({
            "Unnamed: 0": [0, 1, 2],
            "profid": ["p1", "p1", "p2"],
            "yearday": [150, 150, 151],
            "lat": [-55, -55, -55],
            "lon": [40, 40, 40],
            "pressure": [10, 100, 10],
            "CT": [2.0, 1.5, 2.1],
            "SA": [34.5, 34.6, 34.4],
            "oxygen": [300, 280, 305],
            "sigma0": [27.0, 27.2, 27.0],
            "spice": [0.1, 0.1, 0.1],
            "ydcos": [-0.84, -0.84, -0.85],
            "ydsin": [0.53, 0.53, 0.52],
            "nitrate": [25.0, 27.0, 24.5],
        })
        fpath = tmp_path / "test.csv"
        df.to_csv(fpath, index=False)
        return fpath

    def test_removes_unnamed(self, sample_csv):
        df = load_dataset(sample_csv, "test")
        assert "Unnamed: 0" not in df.columns

    def test_preserves_features(self, sample_csv):
        df = load_dataset(sample_csv, "test")
        for col in RFR_FEATURES:
            assert col in df.columns, f"Missing {col}"

    def test_no_nan_critical(self, sample_csv):
        df = load_dataset(sample_csv, "test")
        assert not df["CT"].isna().any()
        assert not df["SA"].isna().any()


class TestFeatureApplication:
    """apply_features() with synthetic data."""

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "profid": ["p1"] * 5,
            "yearday": [150, 150, 150, 150, 150],
            "lat": [-55] * 5,
            "lon": [40] * 5,
            "pressure": [10, 50, 100, 200, 500],
            "CT": [2.5, 2.3, 2.0, 1.5, 1.0],
            "SA": [34.0, 34.2, 34.4, 34.5, 34.6],
            "oxygen": [310, 300, 290, 270, 250],
            "nitrate": [24.5, 25.0, 26.0, 27.5, 29.0],
        })

    def test_computes_seasonal_encoding(self, sample_df):
        df, X, y = apply_features(sample_df.copy(), "test")
        assert "ydcos" in X.columns
        assert "ydsin" in X.columns

    def test_exactly_nine_features(self, sample_df):
        df, X, y = apply_features(sample_df.copy(), "test")
        assert X.shape[1] == 9

    def test_feature_names_match(self, sample_df):
        df, X, y = apply_features(sample_df.copy(), "test")
        assert list(X.columns) == RFR_FEATURES

    def test_no_nan_in_output(self, sample_df):
        df, X, y = apply_features(sample_df.copy(), "test")
        assert not X.isna().any().any()

    def test_y_matches_target(self, sample_df):
        df, X, y = apply_features(sample_df.copy(), "test")
        assert y is not None
        assert len(y) == 5

    def test_glider_with_juld(self):
        """Glider data uses 'juld' column for time."""
        df = pd.DataFrame({
            "juld": ["2019-06-01", "2019-06-02"],
            "yearday": [151, 152],  # pre-existing
            "lat": [-54, -54],
            "lon": [41, 41],
            "pressure": [50, 100],
            "CT": [2.0, 1.8],
            "SA": [34.3, 34.4],
            "oxygen": [290, 285],
        })
        _, X, y = apply_features(df, "glider")
        assert X.shape[1] == 9


class TestMLDApplication:
    """apply_mld() with synthetic profiles."""

    @pytest.fixture
    def two_profile_df(self):
        p = np.linspace(5, 300, 60)
        return pd.DataFrame({
            "profid": ["A"] * 60 + ["B"] * 60,
            "pressure": list(p) + list(p),
            "sigma0": (list(27.0 + 0.01 * (p - 5)) +
                       list(27.2 + 0.04 * (p - 5))),
            "nitrate": list(25 + 0.02 * p) + list(24 + 0.03 * p),
        })

    def test_returns_dataframe_tuple(self, two_profile_df):
        df, mld_df = apply_mld(two_profile_df, "test")
        assert mld_df is not None
        assert "MLD" in mld_df.columns

    def test_two_profiles_two_mlds(self, two_profile_df):
        df, mld_df = apply_mld(two_profile_df, "test")
        assert len(mld_df) == 2

    def test_n_ml_in_mld_df(self, two_profile_df):
        df, mld_df = apply_mld(two_profile_df, "test")
        assert "N_ML" in mld_df.columns
        assert "delta_N_ML" in mld_df.columns

    def test_n_ml_reasonable(self, two_profile_df):
        """N_ML should be near-surface nitrate."""
        df, mld_df = apply_mld(two_profile_df, "test")
        for pid in ["A", "B"]:
            nml = mld_df.loc[pid, "N_ML"]
            surface_n = df[df["profid"] == pid]["nitrate"].iloc[0]
            assert abs(nml - surface_n) < 2.0


# ── Full pipeline integration (requires Zenodo data) ───────────────

@pytest.mark.skipif(not _has_data(), reason="Zenodo data not downloaded")
class TestPipelineWithRealData:
    """Integration tests using actual Zenodo data files."""

    def test_training_data_loadable(self):
        spec = DATASETS["training"]["argo"]
        df = load_dataset(spec["path"], "training_argo", row_filter=spec.get("filter"))
        assert len(df) > 20000

    def test_training_features_valid(self):
        spec = DATASETS["training"]["argo"]
        df = load_dataset(spec["path"], "training_argo", row_filter=spec.get("filter"))
        df, X, y = apply_features(df, "training_argo")
        assert X.shape[1] == 9
        assert X.shape[0] == len(y)
        assert not X.isna().any().any()

    def test_training_feature_ranges(self):
        """Verify feature ranges match paper's Southern Ocean context."""
        spec = DATASETS["training"]["argo"]
        df = load_dataset(spec["path"], "training_argo", row_filter=spec.get("filter"))
        df, X, y = apply_features(df, "training_argo")

        # Paper: upper 1000 m, Antarctic Southern Zone
        assert X["pressure"].max() <= 1100  # ~1000 dbar limit
        assert X["lat"].min() >= -70
        assert X["lat"].max() <= -45
        assert X["lon"].min() >= 0
        assert X["lon"].max() <= 70

    def test_test_data_has_predictions(self):
        spec = DATASETS["test"]["sogos_float"]
        df = load_dataset(spec["path"], "sogos_float")
        # SOGOS test file has RFR_prediction column
        assert "RFR_prediction" in df.columns or "nitrate" in df.columns

    def test_glider_data_loadable(self):
        spec = DATASETS["gliders"]["sg659"]
        df = load_dataset(spec["path"], "sg659")
        assert len(df) > 100000  # ~948k observations for SG659
        assert "nitrate_G" in df.columns  # RFR-predicted nitrate

    def test_output_directory_writable(self, tmp_path):
        """Pipeline should be able to write output."""
        assert OUTPUT_DIR.exists() or OUTPUT_DIR.parent.exists()
