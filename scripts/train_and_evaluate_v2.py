from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from tabpfn import TabPFNClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from adni_pipeline.calibration_uncertainty import (
    compute_brier_score,
    compute_multiclass_ece,
    compute_predictive_entropy,
)
from adni_pipeline.tabpfn_runner import align_probabilities

CLASS_NAMES = ["CN", "MCI", "AD"]
CLASS_LABELS = [0, 1, 2]
META_COLUMNS = {"subject_id", "acquisition_group_id", "split", "diagnosis_xml", "target_label"}


def compute_all_metrics(
    y_true: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    acc = float(accuracy_score(y_true, predictions))
    bacc = float(balanced_accuracy_score(y_true, predictions))
    macro_f1 = float(f1_score(y_true, predictions, labels=CLASS_LABELS, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, predictions, labels=CLASS_LABELS, average="weighted", zero_division=0))
    prec_macro = float(precision_score(y_true, predictions, labels=CLASS_LABELS, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true, predictions, labels=CLASS_LABELS, average="macro", zero_division=0))
    mcc = float(matthews_corrcoef(y_true, predictions))
    kappa = float(cohen_kappa_score(y_true, predictions))
    cm = confusion_matrix(y_true, predictions, labels=CLASS_LABELS).tolist()
    brier = float(compute_brier_score(y_true, probabilities))
    ece_val, mce_val, _ = compute_multiclass_ece(y_true, probabilities)
    ece = float(ece_val)
    mce = float(mce_val)
    entropies = compute_predictive_entropy(probabilities)
    mean_entropy = float(np.mean(entropies))
    mean_conf = float(np.mean(np.max(probabilities, axis=1)))

    per_class_f1 = f1_score(y_true, predictions, labels=CLASS_LABELS, average=None, zero_division=0)
    per_class_f1_dict = {name: float(val) for name, val in zip(CLASS_NAMES, per_class_f1)}

    report = classification_report(
        y_true,
        predictions,
        labels=CLASS_LABELS,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "mcc": mcc,
        "cohen_kappa": kappa,
        "brier_score": brier,
        "ece": ece,
        "mce": mce,
        "mean_predictive_entropy": mean_entropy,
        "mean_confidence": mean_conf,
        "per_class_f1": per_class_f1_dict,
        "confusion_matrix": cm,
        "classification_report": report,
    }


def compute_subject_level_metrics(
    df: pd.DataFrame,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    temp = df[["subject_id", "target_label", "diagnosis_xml"]].copy().reset_index(drop=True)
    for i, name in enumerate(CLASS_NAMES):
        temp[f"prob_{name}"] = probabilities[:, i]

    subject_rows = []
    for subject_id, group in temp.groupby("subject_id"):
        true_label = int(group["target_label"].iloc[0])
        diagnosis = str(group["diagnosis_xml"].iloc[0])
        mean_probs = np.array([group[f"prob_{name}"].mean() for name in CLASS_NAMES])
        pred_label = int(np.argmax(mean_probs))
        subject_rows.append({
            "subject_id": subject_id,
            "true_label": true_label,
            "diagnosis_xml": diagnosis,
            "predicted_label": pred_label,
            "predicted_class": CLASS_NAMES[pred_label],
            "prob_CN": float(mean_probs[0]),
            "prob_MCI": float(mean_probs[1]),
            "prob_AD": float(mean_probs[2]),
            "confidence": float(np.max(mean_probs)),
        })

    subj_df = pd.DataFrame(subject_rows)
    y_true = subj_df["true_label"].to_numpy()
    y_pred = subj_df["predicted_label"].to_numpy()

    acc = float(accuracy_score(y_true, y_pred))
    bacc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, labels=CLASS_LABELS, average="macro", zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS).tolist()

    return {
        "subject_count": len(subj_df),
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
        "subject_predictions": subj_df.to_dict(orient="records"),
    }


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_iterations: int = 1000,
    random_seed: int = 42,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(random_seed)
    n = len(y_true)
    accs, baccs, f1s = [], [], []

    for _ in range(n_iterations):
        indices = rng.choice(n, size=n, replace=True)
        sample_true = y_true[indices]
        sample_pred = y_pred[indices]

        accs.append(accuracy_score(sample_true, sample_pred))
        baccs.append(balanced_accuracy_score(sample_true, sample_pred))
        f1s.append(f1_score(sample_true, sample_pred, labels=CLASS_LABELS, average="macro", zero_division=0))

    return {
        "accuracy": {
            "lower": float(np.percentile(accs, 2.5)),
            "upper": float(np.percentile(accs, 97.5)),
        },
        "balanced_accuracy": {
            "lower": float(np.percentile(baccs, 2.5)),
            "upper": float(np.percentile(baccs, 97.5)),
        },
        "macro_f1": {
            "lower": float(np.percentile(f1s, 2.5)),
            "upper": float(np.percentile(f1s, 97.5)),
        },
    }


def run_training_and_evaluation() -> None:
    project_root = PROJECT_ROOT
    clean_data_dir = project_root / "data" / "clean_v2"
    model_inputs_dir = clean_data_dir / "model_inputs"
    gwo_dir = clean_data_dir / "gwo_selection"
    experiments_dir = project_root / "experiments" / "v2_tabpfn"
    reports_dir = project_root / "reports" / "v2_gwo_tabpfn"
    artifact_pkg_dir = project_root / "artifacts" / "v2_gwo_tabpfn"

    experiments_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    artifact_pkg_dir.mkdir(parents=True, exist_ok=True)

    experiments = [
        {
            "name": "clinical_baseline",
            "modality": "clinical",
            "input_dir": model_inputs_dir / "clinical",
            "features_json": model_inputs_dir / "clinical" / "feature_names.json",
            "is_gwo": False,
        },
        {
            "name": "radiomics_baseline",
            "modality": "radiomics",
            "input_dir": model_inputs_dir / "radiomics",
            "features_json": model_inputs_dir / "radiomics" / "feature_names.json",
            "is_gwo": False,
        },
        {
            "name": "multimodal_baseline",
            "modality": "multimodal",
            "input_dir": model_inputs_dir / "multimodal",
            "features_json": model_inputs_dir / "multimodal" / "feature_names.json",
            "is_gwo": False,
        },
        {
            "name": "multimodal_gwo",
            "modality": "multimodal",
            "input_dir": gwo_dir / "multimodal",
            "features_json": gwo_dir / "multimodal" / "selected_features.json",
            "is_gwo": True,
        },
    ]

    all_results = {}
    comparison_rows = []
    subject_comparison_rows = []

    print("==================================================")
    print("      STARTING V2 TABPFN EXPERIMENTS EVALUATION")
    print("==================================================")

    for exp in experiments:
        exp_name = exp["name"]
        print(f"\n--- Running Experiment: {exp_name} ---")

        train_path = exp["input_dir"] / "train.csv"
        val_path = exp["input_dir"] / "validation.csv"
        test_path = exp["input_dir"] / "test.csv"

        with open(exp["features_json"], encoding="utf-8") as f:
            features_list = json.load(f)

        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)

        X_train = train_df[features_list].to_numpy(dtype=np.float32)
        y_train = train_df["target_label"].to_numpy(dtype=int)

        X_val = val_df[features_list].to_numpy(dtype=np.float32)
        y_val = val_df["target_label"].to_numpy(dtype=int)

        X_test = test_df[features_list].to_numpy(dtype=np.float32)
        y_test = test_df["target_label"].to_numpy(dtype=int)

        # Train TabPFN
        clf = TabPFNClassifier(device="cpu", n_estimators=4, show_progress_bar=False)
        t0 = time.perf_counter()
        clf.fit(X_train, y_train)
        fit_time = time.perf_counter() - t0

        # Predictions
        val_probs_raw = clf.predict_proba(X_val)
        val_preds = clf.predict(X_val)
        classes = getattr(clf, "classes_", np.array(CLASS_LABELS))
        val_probs = align_probabilities(val_probs_raw, classes)

        test_probs_raw = clf.predict_proba(X_test)
        test_preds = clf.predict(X_test)
        test_probs = align_probabilities(test_probs_raw, classes)

        # Save experiment dir
        exp_out = experiments_dir / exp_name
        exp_out.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, exp_out / "model.joblib")

        # Save test predictions CSV
        test_pred_df = test_df[["subject_id", "acquisition_group_id", "split", "diagnosis_xml", "target_label"]].copy()
        test_pred_df["prediction"] = test_preds
        test_pred_df["confidence"] = np.max(test_probs, axis=1)
        for i, cname in enumerate(CLASS_NAMES):
            test_pred_df[f"probability_{cname}"] = test_probs[:, i]
        test_pred_df.to_csv(exp_out / "test_predictions.csv", index=False)

        # Metrics
        val_metrics = compute_all_metrics(y_val, val_preds, val_probs)
        test_metrics = compute_all_metrics(y_test, test_preds, test_probs)
        subj_test_metrics = compute_subject_level_metrics(test_df, test_probs)

        # Bootstrap CIs for test
        scan_ci = bootstrap_ci(y_test, test_preds)
        test_metrics["bootstrap_ci_95"] = scan_ci

        subj_true = np.array([r["true_label"] for r in subj_test_metrics["subject_predictions"]])
        subj_pred = np.array([r["predicted_label"] for r in subj_test_metrics["subject_predictions"]])
        subj_ci = bootstrap_ci(subj_true, subj_pred)
        subj_test_metrics["bootstrap_ci_95"] = subj_ci

        exp_summary = {
            "experiment_name": exp_name,
            "modality": exp["modality"],
            "feature_count": len(features_list),
            "features": features_list,
            "training_samples": len(train_df),
            "validation_samples": len(val_df),
            "test_samples": len(test_df),
            "training_time_sec": fit_time,
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics,
            "subject_level_test_metrics": subj_test_metrics,
        }

        with open(exp_out / "experiment_summary.json", "w", encoding="utf-8") as f:
            json.dump(exp_summary, f, indent=2)

        all_results[exp_name] = exp_summary

        print(f"  Feature Count: {len(features_list)}")
        print(f"  Validation Acc: {val_metrics['accuracy']:.4f} | Bal Acc: {val_metrics['balanced_accuracy']:.4f}")
        print(f"  Test Scan Acc:  {test_metrics['accuracy']:.4f} (95% CI: [{scan_ci['accuracy']['lower']:.3f}, {scan_ci['accuracy']['upper']:.3f}])")
        print(f"  Test Scan BalAcc: {test_metrics['balanced_accuracy']:.4f} | Macro F1: {test_metrics['macro_f1']:.4f}")
        print(f"  Test Subj Acc:  {subj_test_metrics['accuracy']:.4f} (95% CI: [{subj_ci['accuracy']['lower']:.3f}, {subj_ci['accuracy']['upper']:.3f}])")
        print(f"  Test Subj BalAcc: {subj_test_metrics['balanced_accuracy']:.4f} | Subj Macro F1: {subj_test_metrics['macro_f1']:.4f}")

        comparison_rows.append({
            "Experiment": exp_name,
            "Modality": exp["modality"],
            "Features": len(features_list),
            "Val_Accuracy": val_metrics["accuracy"],
            "Val_Balanced_Accuracy": val_metrics["balanced_accuracy"],
            "Test_Accuracy": test_metrics["accuracy"],
            "Test_Balanced_Accuracy": test_metrics["balanced_accuracy"],
            "Test_Macro_F1": test_metrics["macro_f1"],
            "Test_Weighted_F1": test_metrics["weighted_f1"],
            "Test_Precision_Macro": test_metrics["precision_macro"],
            "Test_Recall_Macro": test_metrics["recall_macro"],
            "Test_MCC": test_metrics["mcc"],
            "Test_Brier_Score": test_metrics["brier_score"],
            "Test_ECE": test_metrics["ece"],
            "Test_Entropy": test_metrics["mean_predictive_entropy"],
            "CI_Accuracy_Low": scan_ci["accuracy"]["lower"],
            "CI_Accuracy_High": scan_ci["accuracy"]["upper"],
            "CI_BalAcc_Low": scan_ci["balanced_accuracy"]["lower"],
            "CI_BalAcc_High": scan_ci["balanced_accuracy"]["upper"],
        })

        subject_comparison_rows.append({
            "Experiment": exp_name,
            "Modality": exp["modality"],
            "Features": len(features_list),
            "Subject_Accuracy": subj_test_metrics["accuracy"],
            "Subject_Balanced_Accuracy": subj_test_metrics["balanced_accuracy"],
            "Subject_Macro_F1": subj_test_metrics["macro_f1"],
            "CI_Subj_Acc_Low": subj_ci["accuracy"]["lower"],
            "CI_Subj_Acc_High": subj_ci["accuracy"]["upper"],
            "CI_Subj_BalAcc_Low": subj_ci["balanced_accuracy"]["lower"],
            "CI_Subj_BalAcc_High": subj_ci["balanced_accuracy"]["upper"],
        })

    # Save Comparison CSVs
    comp_df = pd.DataFrame(comparison_rows)
    comp_csv = reports_dir / "model_comparison.csv"
    comp_df.to_csv(comp_csv, index=False)

    subj_comp_df = pd.DataFrame(subject_comparison_rows)
    subj_comp_csv = reports_dir / "subject_level_comparison.csv"
    subj_comp_df.to_csv(subj_comp_csv, index=False)

    with open(reports_dir / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    # -------------------------------------------------------------------------
    # Package Active Final Model into artifacts/v2_gwo_tabpfn/
    # -------------------------------------------------------------------------
    print("\n==================================================")
    print("      PACKAGING ACTIVE MODEL -> artifacts/v2_gwo_tabpfn/")
    print("==================================================")

    active_exp = all_results["multimodal_gwo"]
    active_clf_path = experiments_dir / "multimodal_gwo" / "model.joblib"
    imputer_src = model_inputs_dir / "multimodal" / "imputer.joblib"
    scaler_src = model_inputs_dir / "multimodal" / "scaler.joblib"
    all_features_src = model_inputs_dir / "multimodal" / "feature_names.json"
    clean_manifest_src = project_root / "reports" / "data_audit" / "v2_clean_manifest.csv"

    # Copy files
    shutil.copy2(active_clf_path, artifact_pkg_dir / "model.joblib")
    shutil.copy2(imputer_src, artifact_pkg_dir / "imputer.joblib")
    shutil.copy2(scaler_src, artifact_pkg_dir / "scaler.joblib")
    shutil.copy2(all_features_src, artifact_pkg_dir / "feature_names.json")
    shutil.copy2(clean_manifest_src, artifact_pkg_dir / "manifest.csv")

    with open(artifact_pkg_dir / "selected_features.json", "w", encoding="utf-8") as f:
        json.dump(active_exp["features"], f, indent=2)

    with open(artifact_pkg_dir / "class_mapping.json", "w", encoding="utf-8") as f:
        json.dump({"CN": 0, "MCI": 1, "AD": 2}, f, indent=2)

    pipeline_config = {
        "active_model": "TabPFNClassifier",
        "active_feature_selection": "GreyWolfOptimizer",
        "classifier": {
            "backend": "tabpfn",
            "device": "cpu",
            "n_estimators": 4,
        },
        "feature_selection": {
            "method": "GWO",
            "population_size": 10,
            "iterations": 12,
            "fitness_evaluator": "TabPFNClassifier",
            "cv_strategy": "GroupKFold (train-only, by subject_id)",
        },
        "preprocessing": {
            "imputation": "SimpleImputer(strategy='median', keep_empty_features=True)",
            "scaling": "StandardScaler()",
            "fit_split": "train_only",
        },
        "classes": CLASS_NAMES,
        "input_features_count": 52,
        "selected_features_count": len(active_exp["features"]),
    }
    with open(artifact_pkg_dir / "pipeline_config.json", "w", encoding="utf-8") as f:
        json.dump(pipeline_config, f, indent=2)

    metadata = {
        "package_name": "v2_gwo_tabpfn",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "description": "Clean, deduplicated, zero-leakage multimodal Alzheimer's classification package",
        "total_subjects": 24,
        "train_subjects": 14,
        "val_subjects": 5,
        "test_subjects": 5,
        "train_scans": 35,
        "val_scans": 7,
        "test_scans": 23,
        "metrics": {
            "validation_accuracy": active_exp["validation_metrics"]["accuracy"],
            "validation_balanced_accuracy": active_exp["validation_metrics"]["balanced_accuracy"],
            "test_accuracy": active_exp["test_metrics"]["accuracy"],
            "test_balanced_accuracy": active_exp["test_metrics"]["balanced_accuracy"],
            "test_macro_f1": active_exp["test_metrics"]["macro_f1"],
            "test_brier_score": active_exp["test_metrics"]["brier_score"],
            "test_ece": active_exp["test_metrics"]["ece"],
            "subject_level_test_accuracy": active_exp["subject_level_test_metrics"]["accuracy"],
            "subject_level_test_balanced_accuracy": active_exp["subject_level_test_metrics"]["balanced_accuracy"],
            "subject_level_test_macro_f1": active_exp["subject_level_test_metrics"]["macro_f1"],
        },
        "bootstrap_ci_95": {
            "scan_level": active_exp["test_metrics"]["bootstrap_ci_95"],
            "subject_level": active_exp["subject_level_test_metrics"]["bootstrap_ci_95"],
        },
    }
    with open(artifact_pkg_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Active Artifact Package Contents:")
    for p in sorted(artifact_pkg_dir.iterdir()):
        print(f"  {p.name} ({p.stat().st_size} bytes)")

    print(f"\nModel Comparison Saved to: {comp_csv}")
    print(f"Subject-Level Comparison Saved to: {subj_comp_csv}")
    print("\nFinished successfully!")


if __name__ == "__main__":
    run_training_and_evaluation()
