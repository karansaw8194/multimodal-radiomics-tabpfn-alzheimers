"""Two-Stage Hierarchical 3D Vision Mamba + TabPFN Training and Evaluation Pipeline.

Architecture:
  - Input: 3D Vision Mamba (Med-Mamba) deep spatial embeddings + Granular Clinical Staging (CDRSB, sub-scores, APOE4)
  - Preprocessing: Train-only SimpleImputer(strategy='median') & StandardScaler
  - Feature Selection: Grey Wolf Optimization (GWO) with GroupKFold(subject_id) constrained to 10-18 features
  - Stage 1 (Screening): TabPFN distinguishing Cognitively Normal (CN) vs. Impaired (MCI + AD)
  - Stage 2 (Staging): TabPFN distinguishing Mild Cognitive Impairment (MCI) vs. Alzheimer's Disease (AD)
  - Calibration: Validation-guided Platt Scaling (Sigmoid Logistic Calibration) at standard threshold P(AD) >= 0.50
  - Invariants: Strict zero-leakage subject splitting, 1,000-sample bootstrap 95% CIs
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Callable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    recall_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from tabpfn import TabPFNClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DATA_DIR = PROJECT_ROOT / "data" / "clean_v3"
DATASET_CSV = DATA_DIR / "mamba_multimodal_dataset.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "v4_mamba_tabpfn"
REPORTS_DIR = PROJECT_ROOT / "reports" / "v4_mamba_tabpfn"

META_COLS = {
    "nifti_path",
    "image_id",
    "subject_id",
    "date_acquired",
    "xml_path",
    "series_id",
    "diagnosis",
    "target_label",
    "split",
}


class ConstrainedBinaryGWO:
    """Grey Wolf Optimizer for feature selection constrained strictly to [min_features, max_features]."""

    def __init__(
        self,
        n_wolves: int = 10,
        max_iter: int = 8,
        min_features: int = 10,
        max_features: int = 18,
        cv_folds: int = 5,
        random_state: int = 42,
        cost_sensitive: bool = False,
    ) -> None:
        self.n_wolves = n_wolves
        self.max_iter = max_iter
        self.min_features = min_features
        self.max_features = max_features
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.cost_sensitive = cost_sensitive

    def _evaluate_feature_subset(
        self,
        mask: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
    ) -> float:
        num_selected = int(np.sum(mask))
        if num_selected == 0:
            return 0.0

        X_sub = X[:, mask]
        gkf = GroupKFold(n_splits=self.cv_folds)
        scores = []

        for train_idx, val_idx in gkf.split(X_sub, y, groups=groups):
            X_tr, y_tr = X_sub[train_idx], y[train_idx]
            X_va, y_va = X_sub[val_idx], y[val_idx]

            clf = LGBMClassifier(
                n_estimators=40,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1,
            )
            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_va)

            if self.cost_sensitive:
                rec_ad = recall_score(y_va, preds, pos_label=1, zero_division=0)
                rec_mci = recall_score(y_va, preds, pos_label=0, zero_division=0)
                score = 0.55 * rec_ad + 0.45 * rec_mci
            else:
                score = balanced_accuracy_score(y_va, preds)
            scores.append(score)

        mean_score = float(np.mean(scores))
        # Penalty for size
        feature_ratio = num_selected / len(mask)
        fitness = mean_score - 0.001 * feature_ratio
        return fitness

    def _enforce_constraints(self, sigmas: np.ndarray) -> np.ndarray:
        """Enforce strict [min_features, max_features] bounds based on selection probabilities."""
        mask = sigmas > 0.5
        k = int(np.sum(mask))
        if k < self.min_features:
            top_indices = np.argsort(sigmas)[-self.min_features :]
            mask = np.zeros_like(mask, dtype=bool)
            mask[top_indices] = True
        elif k > self.max_features:
            top_indices = np.argsort(sigmas)[-self.max_features :]
            mask = np.zeros_like(mask, dtype=bool)
            mask[top_indices] = True
        return mask

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        feature_names: List[str],
        stage_name: str = "Stage 1",
    ) -> List[str]:
        rng = np.random.RandomState(self.random_state)
        n_features = X.shape[1]

        # Initialize wolves uniformly in [-2.0, 2.0]
        positions = rng.uniform(-2.0, 2.0, size=(self.n_wolves, n_features))

        alpha_pos = np.zeros(n_features)
        alpha_score = -float("inf")
        beta_pos = np.zeros(n_features)
        beta_score = -float("inf")
        delta_pos = np.zeros(n_features)
        delta_score = -float("inf")

        print(
            f"\n[{stage_name} GWO] Starting Swarm Optimization ({self.n_wolves} wolves, {self.max_iter} iter, "
            f"constrained {self.min_features}-{self.max_features} features)..."
        )

        for it in range(self.max_iter):
            t0 = time.time()
            a = 2.0 - it * (2.0 / self.max_iter)

            for i in range(self.n_wolves):
                sig = 1.0 / (1.0 + np.exp(-np.clip(positions[i], -30, 30)))
                mask = self._enforce_constraints(sig)
                fitness = self._evaluate_feature_subset(mask, X, y, groups)

                if fitness > alpha_score:
                    delta_score, delta_pos = beta_score, beta_pos.copy()
                    beta_score, beta_pos = alpha_score, alpha_pos.copy()
                    alpha_score, alpha_pos = fitness, positions[i].copy()
                elif fitness > beta_score:
                    delta_score, delta_pos = beta_score, beta_pos.copy()
                    beta_score, beta_pos = fitness, positions[i].copy()
                elif fitness > delta_score:
                    delta_score, delta_pos = fitness, positions[i].copy()

            # Update wolf positions
            for i in range(self.n_wolves):
                for j in range(n_features):
                    r1, r2 = rng.rand(), rng.rand()
                    A1 = 2.0 * a * r1 - a
                    C1 = 2.0 * r2
                    D_alpha = abs(C1 * alpha_pos[j] - positions[i, j])
                    X1 = alpha_pos[j] - A1 * D_alpha

                    r1, r2 = rng.rand(), rng.rand()
                    A2 = 2.0 * a * r1 - a
                    C2 = 2.0 * r2
                    D_beta = abs(C2 * beta_pos[j] - positions[i, j])
                    X2 = beta_pos[j] - A2 * D_beta

                    r1, r2 = rng.rand(), rng.rand()
                    A3 = 2.0 * a * r1 - a
                    C3 = 2.0 * r2
                    D_delta = abs(C3 * delta_pos[j] - positions[i, j])
                    X3 = delta_pos[j] - A3 * D_delta

                    positions[i, j] = (X1 + X2 + X3) / 3.0

            final_sig = 1.0 / (1.0 + np.exp(-np.clip(alpha_pos, -30, 30)))
            best_mask = self._enforce_constraints(final_sig)
            elapsed = time.time() - t0
            print(
                f"[{stage_name} GWO] Iter {it+1:02d}/{self.max_iter:02d} | "
                f"Best Fitness: {alpha_score:.4f} | Features: {int(np.sum(best_mask))} | "
                f"Elapsed: {elapsed:.1f}s"
            )

        final_sig = 1.0 / (1.0 + np.exp(-np.clip(alpha_pos, -30, 30)))
        final_mask = self._enforce_constraints(final_sig)
        selected_features = [feature_names[i] for i in range(n_features) if final_mask[i]]

        print(
            f"[{stage_name} GWO] Final Selection: {len(selected_features)} features "
            f"(Fitness: {alpha_score:.4f}):\n  {selected_features}"
        )
        return selected_features


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 1000,
    seed: int = 42,
) -> Tuple[float, float]:
    """Calculate empirical 95% bootstrap confidence interval."""
    rng = np.random.RandomState(seed)
    scores = []
    n = len(y_true)
    for _ in range(n_boot):
        indices = rng.choice(n, size=n, replace=True)
        scores.append(metric_fn(y_true[indices], y_pred[indices]))
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def main() -> None:
    print("=" * 70)
    print("3D VISION MAMBA + TABPFN HIERARCHICAL DIAGNOSTIC PIPELINE (V4)")
    print("=" * 70)

    if not DATASET_CSV.exists():
        print(f"Dataset CSV missing: {DATASET_CSV}")
        sys.exit(1)

    df_all = pd.read_csv(DATASET_CSV)
    print(f"Loaded cohort dataset: {len(df_all)} scans, {len(df_all['subject_id'].unique())} subjects.")

    # Drop zero-information columns (all-NaN columns from XML)
    drop_cols = ["clin_apoe_a1", "clin_apoe_a2", "clin_gds"]

    feature_cols = [
        c
        for c in df_all.columns
        if c not in META_COLS
        and c not in drop_cols
        and pd.api.types.is_numeric_dtype(df_all[c])
    ]
    print(f"Total candidate multimodal features: {len(feature_cols)}")

    # Split partitions
    df_train = df_all[df_all["split"] == "train"].copy()
    df_val = df_all[df_all["split"] == "validation"].copy()
    df_test = df_all[df_all["split"] == "test"].copy()

    # 1. Train-Only Imputation & Standard Scaling (Strict Zero-Leakage)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler(with_mean=True, with_std=True)

    X_train_raw = df_train[feature_cols].values
    X_val_raw = df_val[feature_cols].values
    X_test_raw = df_test[feature_cols].values

    imputer.fit(X_train_raw)
    X_train_imp = imputer.transform(X_train_raw)
    scaler.fit(X_train_imp)

    X_train_scaled = scaler.transform(X_train_imp)
    X_val_scaled = scaler.transform(imputer.transform(X_val_raw))
    X_test_scaled = scaler.transform(imputer.transform(X_test_raw))

    # Save preprocessing transformers and feature catalog
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(imputer, ARTIFACTS_DIR / "imputer.joblib")
    joblib.dump(scaler, ARTIFACTS_DIR / "scaler.joblib")
    with open(ARTIFACTS_DIR / "feature_names.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    # -------------------------------------------------------------------------
    # STAGE 1: SCREENING CLASSIFIER (CN vs IMPAIRED [MCI + AD])
    # -------------------------------------------------------------------------
    y_train_orig = df_train["target_label"].values
    y_train_s1 = (y_train_orig != 0).astype(int)  # 0: CN, 1: Impaired
    groups_train = df_train["subject_id"].values

    print("\n" + "-" * 70)
    print("STAGE 1 TRAINING: SCREENING CLASSIFIER (CN vs IMPAIRED)")
    print(f"Class distribution: CN = {np.sum(y_train_s1 == 0)}, Impaired = {np.sum(y_train_s1 == 1)}")
    print("-" * 70)

    gwo_s1 = ConstrainedBinaryGWO(
        n_wolves=10,
        max_iter=8,
        min_features=10,
        max_features=18,
        cv_folds=5,
        random_state=42,
        cost_sensitive=False,
    )
    s1_features = gwo_s1.fit(
        X_train_scaled, y_train_s1, groups_train, feature_cols, stage_name="Stage 1"
    )

    with open(ARTIFACTS_DIR / "stage1_selected_features.json", "w") as f:
        json.dump(s1_features, f, indent=2)

    s1_feat_indices = [feature_cols.index(f) for f in s1_features]
    X_tr_s1 = X_train_scaled[:, s1_feat_indices]

    print("Fitting TabPFN Classifier for Stage 1 (n_estimators=10)...")
    model_stage1 = TabPFNClassifier(device="cpu", n_estimators=10, show_progress_bar=False)
    model_stage1.fit(X_tr_s1, y_train_s1)
    joblib.dump(model_stage1, ARTIFACTS_DIR / "model_stage1.joblib")
    print("Stage 1 model fitted and saved.")

    # -------------------------------------------------------------------------
    # STAGE 2: SEVERITY / STAGING CLASSIFIER (MCI vs AD)
    # -------------------------------------------------------------------------
    # Train strictly on impaired subjects: 0 = MCI, 1 = AD
    impaired_mask = y_train_orig != 0
    df_train_s2 = df_train[impaired_mask].copy()
    X_tr_s2_all = X_train_scaled[impaired_mask]
    y_train_s2 = (df_train_s2["target_label"].values == 2).astype(int)
    groups_s2 = df_train_s2["subject_id"].values

    print("\n" + "-" * 70)
    print("STAGE 2 TRAINING: SEVERITY CLASSIFIER (MCI vs AD)")
    print(f"Class distribution: MCI = {np.sum(y_train_s2 == 0)}, AD = {np.sum(y_train_s2 == 1)}")
    print("-" * 70)

    gwo_s2 = ConstrainedBinaryGWO(
        n_wolves=10,
        max_iter=8,
        min_features=10,
        max_features=18,
        cv_folds=5,
        random_state=101,
        cost_sensitive=True,
    )
    s2_features = gwo_s2.fit(
        X_tr_s2_all, y_train_s2, groups_s2, feature_cols, stage_name="Stage 2"
    )

    with open(ARTIFACTS_DIR / "stage2_selected_features.json", "w") as f:
        json.dump(s2_features, f, indent=2)

    s2_feat_indices = [feature_cols.index(f) for f in s2_features]
    X_tr_s2 = X_tr_s2_all[:, s2_feat_indices]

    print("Fitting TabPFN Classifier for Stage 2 (n_estimators=12)...")
    model_stage2 = TabPFNClassifier(device="cpu", n_estimators=12, show_progress_bar=False)
    model_stage2.fit(X_tr_s2, y_train_s2)
    joblib.dump(model_stage2, ARTIFACTS_DIR / "model_stage2.joblib")

    print("Fitting Constrained LightGBM Auxiliary Classifier for Stage 2...")
    model_stage2_tree = LGBMClassifier(
        n_estimators=80,
        learning_rate=0.03,
        max_depth=3,
        num_leaves=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    model_stage2_tree.fit(X_tr_s2, y_train_s2)
    joblib.dump(model_stage2_tree, ARTIFACTS_DIR / "model_stage2_tree.joblib")
    print("Stage 2 models fitted and saved.")

    # -------------------------------------------------------------------------
    # VALIDATION-GUIDED PLATT SCALING (PROBABILITY CALIBRATION)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("STAGE 2 PLATT SCALING & ENSEMBLE CALIBRATION ON VALIDATION SPLIT")
    print("-" * 70)
    X_va_s1 = X_val_scaled[:, s1_feat_indices]
    X_va_s2 = X_val_scaled[:, s2_feat_indices]

    y_val_orig = df_val["target_label"].values
    val_imp = y_val_orig != 0
    va_p2_tab = model_stage2.predict_proba(X_va_s2[val_imp])[:, 1]
    va_p2_tree = model_stage2_tree.predict_proba(X_va_s2[val_imp])[:, 1]
    y_val_s2 = (y_val_orig[val_imp] == 2).astype(int)

    best_w = 0.50
    best_val_score = -1.0
    best_platt = None

    for w_cand in [0.0, 0.15, 0.30, 0.50, 0.70, 0.85, 1.0]:
        va_blend = w_cand * va_p2_tab + (1.0 - w_cand) * va_p2_tree
        platt = LogisticRegression(C=1.0, solver="lbfgs")
        platt.fit(va_blend.reshape(-1, 1), y_val_s2)
        cal_val_p = platt.predict_proba(va_blend.reshape(-1, 1))[:, 1]
        val_preds = (cal_val_p >= 0.50).astype(int)
        v_acc = accuracy_score(y_val_s2, val_preds)
        v_bal = balanced_accuracy_score(y_val_s2, val_preds)
        score = 0.5 * v_acc + 0.5 * v_bal
        if score > best_val_score:
            best_val_score = score
            best_w = w_cand
            best_platt = platt

    print(f"Optimal Stage 2 Blend Weight: TabPFN={best_w:.2f}, LightGBM={1-best_w:.2f}")
    print(f"Platt Scaler Fitted (Validation Stage 2 Score: {best_val_score*100:.2f}%)")
    joblib.dump(best_platt, ARTIFACTS_DIR / "stage2_platt_scaler.joblib")
    with open(ARTIFACTS_DIR / "stage2_optimal_threshold.json", "w") as f:
        json.dump(
            {
                "calibrated_threshold": 0.50,
                "best_weight_tabpfn": best_w,
                "val_score": best_val_score,
            },
            f,
            indent=2,
        )

    # -------------------------------------------------------------------------
    # EVALUATION ON HELD-OUT TEST SET (85 Scans across 39 Unique Subjects)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EVALUATING ON HELD-OUT TEST SET (85 Scans across 39 Unique Subjects)")
    print("=" * 70)

    y_test_orig = df_test["target_label"].values  # 0: CN, 1: MCI, 2: AD
    y_test_s1_true = (y_test_orig != 0).astype(int)

    # Stage 1 Test Evaluation
    X_te_s1 = X_test_scaled[:, s1_feat_indices]
    s1_probs = model_stage1.predict_proba(X_te_s1)
    p_impaired = s1_probs[:, 1]
    y_test_s1_pred = (p_impaired >= 0.50).astype(int)

    s1_acc = accuracy_score(y_test_s1_true, y_test_s1_pred)
    s1_bal_acc = balanced_accuracy_score(y_test_s1_true, y_test_s1_pred)
    s1_f1 = f1_score(y_test_s1_true, y_test_s1_pred)
    s1_acc_ci = compute_bootstrap_ci(y_test_s1_true, y_test_s1_pred, accuracy_score)

    print("\n[Stage 1: Screening (CN vs Impaired)]")
    print(f"  Test Accuracy:     {s1_acc*100:.2f}% (95% CI: [{s1_acc_ci[0]*100:.1f}%, {s1_acc_ci[1]*100:.1f}%])")
    print(f"  Balanced Accuracy: {s1_bal_acc*100:.2f}%")
    print(f"  F1-Score:          {s1_f1:.4f}")
    print(f"  Confusion Matrix:\n{confusion_matrix(y_test_s1_true, y_test_s1_pred)}")

    # Stage 2 Direct Test Evaluation (on ground-truth impaired cases)
    te_impaired_mask = y_test_orig != 0
    X_te_s2_impaired = X_test_scaled[te_impaired_mask][:, s2_feat_indices]
    y_test_s2_true = (y_test_orig[te_impaired_mask] == 2).astype(int)

    s2_tab_probs_imp = model_stage2.predict_proba(X_te_s2_impaired)[:, 1]
    s2_tree_probs_imp = model_stage2_tree.predict_proba(X_te_s2_impaired)[:, 1]
    s2_blend_imp = best_w * s2_tab_probs_imp + (1.0 - best_w) * s2_tree_probs_imp
    s2_calibrated_imp = best_platt.predict_proba(s2_blend_imp.reshape(-1, 1))[:, 1]
    y_test_s2_pred = (s2_calibrated_imp >= 0.50).astype(int)

    s2_acc = accuracy_score(y_test_s2_true, y_test_s2_pred)
    s2_bal_acc = balanced_accuracy_score(y_test_s2_true, y_test_s2_pred)
    s2_f1 = f1_score(y_test_s2_true, y_test_s2_pred)
    s2_acc_ci = compute_bootstrap_ci(y_test_s2_true, y_test_s2_pred, accuracy_score)

    print("\n[Stage 2: Staging (MCI vs AD) Direct Evaluation]")
    print(f"  Test Accuracy:     {s2_acc*100:.2f}% (95% CI: [{s2_acc_ci[0]*100:.1f}%, {s2_acc_ci[1]*100:.1f}%])")
    print(f"  Balanced Accuracy: {s2_bal_acc*100:.2f}%")
    print(f"  F1-Score:          {s2_f1:.4f}")
    print(f"  Confusion Matrix:\n{confusion_matrix(y_test_s2_true, y_test_s2_pred)}")

    # End-to-End Hierarchical System (All 3 Classes: CN, MCI, AD)
    X_te_s2_all = X_test_scaled[:, s2_feat_indices]
    s2_tab_all = model_stage2.predict_proba(X_te_s2_all)[:, 1]
    s2_tree_all = model_stage2_tree.predict_proba(X_te_s2_all)[:, 1]
    s2_raw_all = best_w * s2_tab_all + (1.0 - best_w) * s2_tree_all
    s2_cal_all = best_platt.predict_proba(s2_raw_all.reshape(-1, 1))

    final_preds = []
    final_probs = []

    for i in range(len(df_test)):
        p_imp = p_impaired[i]
        p_norm = 1.0 - p_imp

        if p_imp < 0.50:
            final_preds.append(0)  # CN
            final_probs.append([p_norm, 0.5 * p_imp, 0.5 * p_imp])
        else:
            p_mci_s2 = s2_cal_all[i, 0]
            p_ad_s2 = s2_cal_all[i, 1]
            final_preds.append(2 if p_ad_s2 >= 0.50 else 1)
            final_probs.append([p_norm, p_imp * p_mci_s2, p_imp * p_ad_s2])

    final_preds = np.array(final_preds)
    final_probs = np.array(final_probs)

    overall_acc = accuracy_score(y_test_orig, final_preds)
    overall_bal_acc = balanced_accuracy_score(y_test_orig, final_preds)
    overall_macro_f1 = f1_score(y_test_orig, final_preds, average="macro")
    overall_weighted_f1 = f1_score(y_test_orig, final_preds, average="weighted")
    overall_mcc = matthews_corrcoef(y_test_orig, final_preds)
    overall_acc_ci = compute_bootstrap_ci(y_test_orig, final_preds, accuracy_score)
    overall_bal_ci = compute_bootstrap_ci(y_test_orig, final_preds, balanced_accuracy_score)

    print("\n" + "=" * 70)
    print("OVERALL END-TO-END 3-CLASS HIERARCHICAL DIAGNOSTIC RESULTS")
    print("(Platt-Calibrated Decision Cutoff: P(AD) >= 0.50)")
    print("=" * 70)
    print(f"  Test Accuracy:          {overall_acc*100:.2f}% (95% CI: [{overall_acc_ci[0]*100:.1f}%, {overall_acc_ci[1]*100:.1f}%])")
    print(f"  Balanced Accuracy:      {overall_bal_acc*100:.2f}% (95% CI: [{overall_bal_ci[0]*100:.1f}%, {overall_bal_ci[1]*100:.1f}%])")
    print(f"  Macro F1-Score:         {overall_macro_f1:.4f}")
    print(f"  Weighted F1-Score:      {overall_weighted_f1:.4f}")
    print(f"  Matthews Corr (MCC):    {overall_mcc:.4f}")
    cm = confusion_matrix(y_test_orig, final_preds)
    print(f"\n  Confusion Matrix [Rows: True (CN, MCI, AD), Cols: Pred (CN, MCI, AD)]:\n{cm}")

    # Subject-Level Patient Aggregation (Majority / Calibrated Probability Vote)
    df_test["pred_label"] = final_preds
    for c_i, c_name in enumerate(["prob_CN", "prob_MCI", "prob_AD"]):
        df_test[c_name] = final_probs[:, c_i]

    subj_preds = []
    subj_trues = []

    for subj_id, s_group in df_test.groupby("subject_id"):
        true_lbl = s_group["target_label"].iloc[0]
        mean_p_cn = s_group["prob_CN"].mean()
        mean_p_imp = 1.0 - mean_p_cn
        if mean_p_imp < 0.50:
            pred_lbl = 0
        else:
            mean_p_ad = s_group["prob_AD"].mean()
            mean_p_mci = s_group["prob_MCI"].mean()
            cond_p_ad = mean_p_ad / (mean_p_ad + mean_p_mci + 1e-12)
            pred_lbl = 2 if cond_p_ad >= 0.50 else 1
        subj_trues.append(true_lbl)
        subj_preds.append(pred_lbl)

    subj_trues = np.array(subj_trues)
    subj_preds = np.array(subj_preds)

    subj_acc = accuracy_score(subj_trues, subj_preds)
    subj_bal_acc = balanced_accuracy_score(subj_trues, subj_preds)
    subj_f1 = f1_score(subj_trues, subj_preds, average="macro")
    subj_acc_ci = compute_bootstrap_ci(subj_trues, subj_preds, accuracy_score)

    print("\n" + "=" * 70)
    print(f"SUBJECT-LEVEL AGGREGATED EVALUATION (N = {len(subj_trues)} Independent Patients)")
    print("=" * 70)
    print(f"  Subject Accuracy:       {subj_acc*100:.2f}% (95% CI: [{subj_acc_ci[0]*100:.1f}%, {subj_acc_ci[1]*100:.1f}%])")
    print(f"  Subject Balanced Acc:   {subj_bal_acc*100:.2f}%")
    print(f"  Subject Macro F1:       {subj_f1:.4f}")
    print(f"  Patients Correct:       {int(np.sum(subj_preds == subj_trues))} / {len(subj_trues)}")
    cm_subj = confusion_matrix(subj_trues, subj_preds)
    print(f"\n  Subject Confusion Matrix:\n{cm_subj}")

    # Save summary metrics table
    summary_df = pd.DataFrame(
        [
            {
                "Pipeline": "Stage 1 (CN vs Impaired)",
                "Features": len(s1_features),
                "Test_Accuracy": s1_acc,
                "Balanced_Accuracy": s1_bal_acc,
                "Macro_F1": s1_f1,
                "CI_Low": s1_acc_ci[0],
                "CI_High": s1_acc_ci[1],
            },
            {
                "Pipeline": "Stage 2 (MCI vs AD)",
                "Features": len(s2_features),
                "Test_Accuracy": s2_acc,
                "Balanced_Accuracy": s2_bal_acc,
                "Macro_F1": s2_f1,
                "CI_Low": s2_acc_ci[0],
                "CI_High": s2_acc_ci[1],
            },
            {
                "Pipeline": "Overall 3-Class (Scan)",
                "Features": f"{len(s1_features)} + {len(s2_features)}",
                "Test_Accuracy": overall_acc,
                "Balanced_Accuracy": overall_bal_acc,
                "Macro_F1": overall_macro_f1,
                "CI_Low": overall_acc_ci[0],
                "CI_High": overall_acc_ci[1],
            },
            {
                "Pipeline": "Overall 3-Class (Subject)",
                "Features": f"{len(s1_features)} + {len(s2_features)}",
                "Test_Accuracy": subj_acc,
                "Balanced_Accuracy": subj_bal_acc,
                "Macro_F1": subj_f1,
                "CI_Low": subj_acc_ci[0],
                "CI_High": subj_acc_ci[1],
            },
        ]
    )
    summary_df.to_csv(REPORTS_DIR / "mamba_tabpfn_model_comparison.csv", index=False)
    print(f"\nSaved comparison table to: {REPORTS_DIR / 'mamba_tabpfn_model_comparison.csv'}")


if __name__ == "__main__":
    main()
