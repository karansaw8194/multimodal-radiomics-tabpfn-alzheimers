from pathlib import Path
import base64
import html
import pandas as pd
import sys

PROJECT = Path(__file__).resolve().parents[1]

SHAP_DIR = PROJECT / "reports/v2_gwo_tabpfn/shap"

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

    graph = SHAP_DIR / "global_shap_importance.png"
    csv_file = SHAP_DIR / "global_shap_importance.csv"

    if not report.exists():
        print(f"Report not found: {report}")
        return

    if not graph.exists() or not csv_file.exists():
        print("SHAP output not ready yet. Run this script again after SHAP finishes.")
        print(f"Expected graph: {graph}")
        return

    content = report.read_text(encoding="utf-8")

    if "shap-explainability-section" in content:
        print("SHAP section already exists in this report.")
        return

    df = pd.read_csv(csv_file).head(10)
    rows = "".join(
        "<tr><td>{}</td><td>{:.6f}</td></tr>".format(
            html.escape(str(row["feature"])),
            float(row["mean_abs_shap"]),
        )
        for _, row in df.iterrows()
    )

    image_data = base64.b64encode(graph.read_bytes()).decode("ascii")

    section = f"""
    <section id="shap-explainability-section"
             style="margin:30px 0;padding:20px;border:1px solid #ccc;
                    border-radius:10px;font-family:Arial,sans-serif">
      <h2>Explainable AI (SHAP)</h2>
      <p>
        Global feature importance for the multimodal TabPFN model.
        Larger mean absolute SHAP values indicate greater average
        contribution magnitude to the model output.
      </p>
      <img src="data:image/png;base64,{image_data}"
           alt="Global SHAP feature importance"
           style="max-width:100%;height:auto">
      <h3>Top 10 influential features</h3>
      <table style="border-collapse:collapse;width:100%">
        <thead><tr><th>Feature</th><th>Mean absolute SHAP</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
      <p style="font-size:small">
        This is a global explanation from the test samples used by the
        SHAP script. It is not a patient-specific explanation for the
        live demo prediction.
      </p>
    </section>
    """

    if "</body>" in content:
        content = content.replace("</body>", section + "\n</body>", 1)
    else:
        content += section

    report.write_text(content, encoding="utf-8")
    print(f"SHAP section added to: {report}")


if __name__ == "__main__":
    main()
