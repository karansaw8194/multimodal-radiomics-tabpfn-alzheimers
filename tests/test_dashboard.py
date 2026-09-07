from pathlib import Path
from adni_pipeline.dashboard import compile_research_dashboard

def test_compile_research_dashboard(tmp_path: Path) -> None:
    output_dir = tmp_path / "reports" / "dashboard"

    result = compile_research_dashboard(
        project_root=tmp_path,
        output_dir=output_dir,
    )

    assert Path(result["dashboard_json"]).exists()
    assert Path(result["dashboard_plot"]).exists()
    assert Path(result["dashboard_html"]).exists()
