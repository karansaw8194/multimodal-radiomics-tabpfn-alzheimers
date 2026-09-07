from pathlib import Path

import pytest

from scripts.select_processing_versions import (
    choose_record,
    select_processing_versions,
)

def test_preferred_version_is_selected() -> None:
    records = [
        {
            "processing_version": "MPR__GradWarp",
            "nifti_path": "/data/a.nii",
        },
        {
            "processing_version": "MPR__GradWarp__B1_Correction__N3",
            "nifti_path": "/data/b.nii",
        },
    ]

    selected, reason = choose_record(
        records=records,
        preferred_versions=[
            "MPR__GradWarp__B1_Correction__N3",
            "MPR__GradWarp",
        ],
        allow_singleton=False,
    )

    assert selected is not None
    assert selected["nifti_path"] == "/data/b.nii"
    assert reason == "preferred_version"

def test_unresolved_group_is_not_silently_selected() -> None:
    records = [
        {
            "processing_version": "MT1__N3m",
            "nifti_path": "/data/a.nii",
        },
        {
            "processing_version": "MT1__GradWarp__N3m",
            "nifti_path": "/data/b.nii",
        },
    ]

    selected, reason = choose_record(
        records=records,
        preferred_versions=["MPR__GradWarp"],
        allow_singleton=False,
    )

    assert selected is None
    assert reason.startswith("no_preferred_version_available")

def test_singleton_requires_explicit_permission() -> None:
    records = [
        {
            "processing_version": "MT1__N3m",
            "nifti_path": "/data/a.nii",
        }
    ]

    selected_without_permission, _ = choose_record(
        records=records,
        preferred_versions=[],
        allow_singleton=False,
    )

    selected_with_permission, reason = choose_record(
        records=records,
        preferred_versions=[],
        allow_singleton=True,
    )

    assert selected_without_permission is None
    assert selected_with_permission is not None
    assert reason == "explicit_singleton"

def test_selection_writes_reports(tmp_path: Path) -> None:
    groups_path = tmp_path / "groups.csv"
    manifest_path = tmp_path / "manifest.csv"
    output_dir = tmp_path / "selected"

    groups_path.write_text(
        (
            "group_id,subject_id,split,record_count,"
            "processing_versions,nifti_paths\n"
            "G1,S001,train,2,"
            "MPR__GradWarp|MPR__GradWarp__B1_Correction,"
            "/data/a.nii|/data/b.nii\n"
        ),
        encoding="utf-8",
    )

    manifest_path.write_text(
        (
            "nifti_path,subject_id,diagnosis_xml,split,"
            "processing_version\n"
            "/data/a.nii,S001,AD,train,MPR__GradWarp\n"
            "/data/b.nii,S001,AD,train,"
            "MPR__GradWarp__B1_Correction\n"
        ),
        encoding="utf-8",
    )

    summary = select_processing_versions(
        acquisition_groups_path=groups_path,
        manifest_path=manifest_path,
        output_dir=output_dir,
        preferred_versions=[
            "MPR__GradWarp__B1_Correction"
        ],
        allow_singleton=False,
    )

    assert summary["selection_completed"] is True
    assert summary["selected_record_count"] == 1
    assert (output_dir / "selected_manifest.csv").exists()
    assert (
        output_dir / "processing_version_selection.csv"
    ).exists()
    assert (
        output_dir / "processing_version_selection_summary.json"
    ).exists()
