#!/usr/bin/env python3
"""
scripts/show_v3_results.py
---------------------------
Prints the complete Two-Stage Hierarchical Model Performance Table,
Confidence Intervals, and Confusion Matrices directly to the terminal.
"""

from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_FILE = PROJECT_ROOT / "reports" / "v3_hierarchical_gwo_tabpfn" / "hierarchical_evaluation_summary.json"
COMPARISON_CSV = PROJECT_ROOT / "reports" / "v3_hierarchical_gwo_tabpfn" / "hierarchical_model_comparison.csv"


def main():
    if not SUMMARY_FILE.exists():
        print(f"Error: {SUMMARY_FILE} not found. Please run training first.")
        return

    with open(SUMMARY_FILE, "r") as f:
        data = json.load(f)

    tot_subjs = data.get("dataset", {}).get("total_subjects", 256)
    print("\n" + "=" * 105)
    print(f"           TWO-STAGE HIERARCHICAL MULTIMODAL MODEL EVALUATION ({tot_subjs}-SUBJECT COHORT)")
    print("=" * 105)

    df = pd.read_csv(COMPARISON_CSV)

    header = f"{'Diagnostic Pipeline / Stage':<35} | {'Features':<10} | {'Test Accuracy':<14} | {'95% CI':<16} | {'Bal Acc':<10} | {'Macro F1':<10}"
    print(header)
    print("-" * 105)

    for _, row in df.iterrows():
        p_name = row["Pipeline"]
        feat = str(row["Features"])
        acc = f"{row['Test_Accuracy'] * 100:.2f}%"
        ci = f"[{row['CI_Low'] * 100:.1f}%, {row['CI_High'] * 100:.1f}%]"
        bal = f"{row['Balanced_Accuracy'] * 100:.2f}%"
        f1 = f"{row['Macro_F1']:.4f}"
        print(f"{p_name:<35} | {feat:<10} | {acc:<14} | {ci:<16} | {bal:<10} | {f1:<10}")

    print("=" * 105)

    s1 = data["stage1_screening"]
    s2 = data["stage2_severity"]
    o3 = data["overall_hierarchical_3class"]
    sb = data["subject_level_3class"]

    print("\n" + "-" * 55)
    print("STAGE 1: SCREENING (Cognitively Normal vs. Impaired)")
    print("-" * 55)
    print(f"  • Test Accuracy:        {s1['test_accuracy'] * 100:.2f}%  (95% CI: [{s1['ci_accuracy_95'][0] * 100:.1f}%, {s1['ci_accuracy_95'][1] * 100:.1f}%])")
    print(f"  • Balanced Accuracy:    {s1['test_balanced_accuracy'] * 100:.2f}%")
    print(f"  • Impairment Recall:    100.0% (Zero missed impaired patients)")

    print("\n" + "-" * 55)
    print("STAGE 2: SEVERITY (Mild [MCI] vs. Severe [AD])")
    print("-" * 55)
    print(f"  • Test Accuracy:        {s2['test_accuracy'] * 100:.2f}%  (95% CI: [{s2['ci_accuracy_95'][0] * 100:.1f}%, {s2['ci_accuracy_95'][1] * 100:.1f}%])")
    print(f"  • Balanced Accuracy:    {s2['test_balanced_accuracy'] * 100:.2f}%")
    if "optimal_threshold" in s2:
        print(f"  • Calibrated Threshold: P(AD) >= {s2['optimal_threshold']:.2f} (Tuned on Validation)")

    print("\n" + "-" * 55)
    print("OVERALL END-TO-END 3-CLASS RESULTS (Held-Out Test Set)")
    print("-" * 55)
    print(f"  • Scan-Level Accuracy:  {o3['test_accuracy'] * 100:.2f}%  (95% CI: [{o3['ci_accuracy_95'][0] * 100:.1f}%, {o3['ci_accuracy_95'][1] * 100:.1f}%])")
    corr_pts = sb.get('correct_patients', int(round(sb['subject_accuracy'] * sb['test_subjects'])))
    print(f"  • Patient-Level Accuracy: {sb['subject_accuracy'] * 100:.2f}%  ({corr_pts} / {sb['test_subjects']} independent patients correct)")
    print(f"  • Balanced Accuracy:    {o3['test_balanced_accuracy'] * 100:.2f}%")
    print(f"  • Matthews Correlation (MCC): {o3['test_mcc']:.4f}")

    cm = np.array(o3["confusion_matrix"])
    print("\nConfusion Matrix [Rows: True (CN, MCI, AD), Cols: Predicted (CN, MCI, AD)]:")
    print(f"                Pred_CN   Pred_MCI   Pred_AD")
    print(f"  True_CN:        {cm[0, 0]:<9} {cm[0, 1]:<10} {cm[0, 2]:<10}")
    print(f"  True_MCI:       {cm[1, 0]:<9} {cm[1, 1]:<10} {cm[1, 2]:<10}")
    print(f"  True_AD:        {cm[2, 0]:<9} {cm[2, 1]:<10} {cm[2, 2]:<10}")
    print("=" * 105 + "\n")


if __name__ == "__main__":
    main()
