from pathlib import Path
import html
import json
import sys
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]

CF_DIR = PROJECT / "reports/v2_gwo_tabpfn/counterfactuals"

REPORT_ROOT = PROJECT / "reports/live_inference"


def main():
    reports = sorted(
        REPORT_ROOT.glob("*/live_inference_report.html"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not reports:
        print("No live inference HTML report found.")
        return

    report = Path(sys.argv[1]) if len(sys.argv) > 1 else reports[0]

    summary_path = CF_DIR / "counterfactual_summary.json"
    csv_path = CF_DIR / "counterfactuals.csv"
    selected_path = CF_DIR / "feature_names.json"

    if not report.is_file():
        print(f"Report not found: {report}")
        return

    if not summary_path.is_file() or not csv_path.is_file():
        print("Counterfactual output is incomplete.")
        return

    content = report.read_text(encoding="utf-8")

    if "counterfactual-explainability-section" in content:
        print("Counterfactual section already exists in this report.")
        return

    with summary_path.open(encoding="utf-8") as f:
        summary = json.load(f)

    df = pd.read_csv(csv_path)
    successful = df["found"].astype(str).str.lower().eq("true")
    success_count = int(successful.sum())
    request_count = int(len(df))
    rate = 100.0 * success_count / request_count if request_count else 0.0

    target_rows = []
    for target in ["CN", "MCI", "AD"]:
        subset = df[
            df["requested_target_class_name"].astype(str).eq(target)
        ]
        found = subset["found"].astype(str).str.lower().eq("true").sum()
        target_rows.append(
            "<tr><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                html.escape(target), int(found), len(subset)
            )
        )

    target_table = "".join(target_rows)

    clinical_features = [
        "clin_apoe_a1",
        "clin_apoe_a2",
        "clin_cdr",
    ]

    validation_rows = []
    for feature in clinical_features:
        column = f"counterfactual_{feature}"
        if column not in df.columns:
            validation_rows.append(
                f"<tr><td>{html.escape(feature)}</td><td>Unavailable</td></tr>"
            )
            continue

        observed = set(
            pd.to_numeric(
                pd.read_csv(
                    PROJECT / "data/clean_v2/model_inputs/multimodal/train.csv"
                )[feature],
                errors="coerce",
            ).dropna().astype(float)
        )
        generated = set(
            pd.to_numeric(df.loc[successful, column], errors="coerce")
            .dropna().astype(float)
        )
        invalid_count = len(generated - observed)
        status = "Pass" if invalid_count == 0 else f"{invalid_count} unexpected value(s)"
        validation_rows.append(
            "<tr><td>{}</td><td>{}</td></tr>".format(
                html.escape(feature), html.escape(status)
            )
        )

    validation_table = "".join(validation_rows)

    section = f"""
<section id="counterfactual-explainability-section"
         style="margin:30px 0;padding:20px;border:1px solid #ccc;
                border-radius:10px;font-family:Arial,sans-serif">
  <h2>Counterfactual Explanation</h2>
  <p>
    This section summarizes a separate global test-set counterfactual
    experiment using the multimodal TabPFN model. It does not describe
    the current MRI patient individually.
  </p>
  <h3>Global search results</h3>
  <p><strong>Successful requests:</strong> {success_count}/{request_count}
     ({rate:.1f}%)</p>
  <table style="border-collapse:collapse;width:100%">
    <thead><tr><th>Requested target class</th>
    <th>Successful</th><th>Total requests</th></tr></thead>
    <tbody>{target_table}</tbody>
  </table>
  <h3>Clinical feature value validation</h3>
  <p>
    Generated values for the listed features were checked against observed
    values in the training set used by this experiment.
  </p>
  <table style="border-collapse:collapse;width:100%">
    <thead><tr><th>Feature</th><th>Validation</th></tr></thead>
    <tbody>{validation_table}</tbody>
  </table>
  <p style="font-size:small">
    Counterfactuals are model-based examples, not causal explanations,
    treatment recommendations, or clinical advice. Matching observed
    training values alone does not establish clinical validity. These
    global results must not be interpreted as patient-specific findings.
  </p>
</section>
"""

    if "</body>" in content:
        content = content.replace("</body>", section + "\n</body>", 1)
    else:
        content += section

    report.write_text(content, encoding="utf-8")
    print(f"Counterfactual section added to: {report}")


if __name__ == "__main__":
    main()
