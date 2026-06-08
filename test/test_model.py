"""
test/test_model.py — Tests for model.py RFR training & evaluation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

import model as m


class TestProfileSplitting:
    """Profile-aware splitting functions."""

    @pytest.fixture
    def sample_data(self):
        X = np.random.randn(100, 9)
        y = np.random.randn(100)
        groups = np.array(["A"] * 30 + ["B"] * 40 + ["C"] * 30)
        return X, y, groups

    def test_holdout_profiles_intact(self, sample_data):
        X, y, groups = sample_data
        X_tr, X_va, y_tr, y_va, g_tr, g_va = m.profile_holdout_split(
            X, y, groups, test_size=0.2, random_state=42
        )
        # No group should appear in both train and val
        tr_set = set(g_tr)
        va_set = set(g_va)
        assert tr_set.isdisjoint(va_set)

    def test_holdout_test_size(self, sample_data):
        X, y, groups = sample_data
        _, X_va, _, _, _, g_va = m.profile_holdout_split(
            X, y, groups, test_size=0.3, random_state=42
        )
        n_test_groups = len(set(g_va))
        assert n_test_groups >= 1  # at least one group in test

    def test_kfold_all_profiles_covered(self, sample_data):
        X, y, groups = sample_data
        splits = m.profile_kfold_cv(X, y, groups, n_splits=3)
        all_val = set()
        for tr_i, val_i in splits:
            all_val.update(groups[val_i])
        assert all_val == set(groups)

    def test_spatial_loo_one_per_group(self, sample_data):
        X, y, groups = sample_data
        splits = m.spatial_leave_one_out(X, y, groups)
        assert len(splits) == len(np.unique(groups))


class TestEvaluation:
    """Evaluation metrics."""

    def test_perfect_prediction(self):
        y = np.array([25.0, 26.0, 27.0])
        result = m.evaluate(y, y, "perfect")
        assert result["MAE"] == 0.0
        assert result["pct_within"] == 100.0
        assert result["mean_bias"] == 0.0

    def test_constant_offset(self):
        y_true = np.array([25.0, 26.0, 27.0])
        y_pred = y_true + 0.3
        result = m.evaluate(y_true, y_pred, "offset")
        assert abs(result["MAE"] - 0.3) < 0.01
        assert abs(result["mean_bias"] - 0.3) < 0.01

    def test_within_uncertainty(self):
        y_true = np.array([25.0] * 100)
        y_pred = np.concatenate([
            y_true[:80] + 0.2,
            y_true[80:] + 1.0,
        ])
        result = m.evaluate(y_true, y_pred, "partial")
        assert 75 < result["pct_within"] < 85  # ~80%

    def test_error_bounds_symmetric(self):
        y_true = np.array([25.0] * 1000)
        y_pred = y_true + np.random.randn(1000) * 0.3
        result = m.evaluate(y_true, y_pred, "symmetric")
        assert result["err_lo"] < 0 < result["err_hi"]


class TestRFRParams:
    """Paper hyperparameters."""

    def test_paper_params(self):
        assert m.RFR_PARAMS["n_estimators"] == 1000
        assert m.RFR_PARAMS["max_features"] == 1 / 3
        assert m.RFR_PARAMS["min_samples_leaf"] == 5
        assert m.RFR_PARAMS["oob_score"] is True

    def test_feature_count(self):
        assert len(m.FEATURE_COLS) == 9

    def test_uncertainty(self):
        assert m.UNCERTAINTY == 0.5


class TestTrainRFR:
    """RFR training function."""

    def test_train_returns_rf(self):
        X = np.random.randn(100, 9)
        y = np.random.randn(100)
        rf = m.train_rfr(X, y, verbose=False)
        assert isinstance(rf, RandomForestRegressor)
        assert rf.n_estimators == 1000

    def test_oob_score(self):
        X = np.random.randn(200, 9)
        y = np.random.randn(200)
        rf = m.train_rfr(X, y, verbose=False)
        assert hasattr(rf, 'oob_score_')
        assert -1.0 < rf.oob_score_ < 1.0


@pytest.mark.skipif(
    not (Path(__file__).parent.parent / "data" / "processed" / "training_X.csv").exists(),
    reason="Processed data not available"
)
class TestModelPipeline:
    """Integration tests with real data."""

    def test_load_processed(self):
        train_X, train_y, test_X, test_y, groups = m.load_processed_data()
        assert train_X.shape[0] == len(train_y)
        assert test_X.shape[0] == len(test_y)
        assert train_X.shape[1] == 9
        assert len(np.unique(groups)) == 10  # 9 floats + 1 GO-SHIP

    def test_training_count(self):
        train_X, train_y, _, _, _ = m.load_processed_data()
        assert train_X.shape[0] == 22800  # paper exact

    def test_model_runs(self):
        train_X, train_y, test_X, test_y, groups = m.load_processed_data()
        rf = m.train_rfr(train_X.values, train_y.values, verbose=False)
        y_pred = rf.predict(test_X.values)
        mae = np.median(np.abs(y_pred - test_y.values))
        # Should be better than random (MAE < 10 for nitrate ~25)
        assert mae < 5.0

    def test_glider_data_exists(self):
        assert (m.ZENODO / "sogos_glider659_RFRpred_setB_modG.csv").exists()
        assert (m.ZENODO / "sogos_glider660_RFRpred_setB_modG.csv").exists()
