from __future__ import annotations

import base64
import io
import json

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

def load_nifti_image(
    nifti_path: Path,
) -> tuple[np.ndarray, Any]:
    """Load a NIfTI image as a finite 3D NumPy array."""
    nifti_path = nifti_path.expanduser().resolve()

    if not nifti_path.exists():
        raise FileNotFoundError(
            f"NIfTI file does not exist: {nifti_path}"
        )

    image = nib.load(str(nifti_path))
    data = image.get_fdata(dtype=np.float32)

    if data.ndim > 3:
        data = data[..., 0]

    if data.ndim != 3:
        raise ValueError(
            f"Expected a 3D NIfTI image, got shape={data.shape}"
        )

    data = np.asarray(data, dtype=np.float32)

    if not np.isfinite(data).all():
        raise ValueError(
            "NIfTI image contains NaN or infinite values"
        )

    return data, image

def load_nifti_mask(
    mask_path: Path,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    """Load and validate a 3D mask."""
    mask_path = mask_path.expanduser().resolve()

    if not mask_path.exists():
        raise FileNotFoundError(
            f"Mask file does not exist: {mask_path}"
        )

    mask_image = nib.load(str(mask_path))
    mask = mask_image.get_fdata()

    if mask.ndim > 3:
        mask = mask[..., 0]

    if mask.shape != expected_shape:
        raise ValueError(
            "Mask shape does not match image shape: "
            f"{mask.shape} versus {expected_shape}"
        )

    return mask > 0

def normalize_slice(
    slice_data: np.ndarray,
) -> np.ndarray:
    """
    Normalize an image slice to [0, 1] using robust percentiles.

    The source image is not changed.
    """
    slice_data = np.asarray(
        slice_data,
        dtype=np.float32,
    )

    finite_values = slice_data[
        np.isfinite(slice_data)
    ]

    if finite_values.size == 0:
        return np.zeros_like(slice_data)

    lower = float(
        np.percentile(finite_values, 1)
    )

    upper = float(
        np.percentile(finite_values, 99)
    )

    if upper <= lower:
        lower = float(np.min(finite_values))
        upper = float(np.max(finite_values))

    if upper <= lower:
        return np.zeros_like(slice_data)

    normalized = (
        slice_data - lower
    ) / (upper - lower)

    return np.clip(
        normalized,
        0.0,
        1.0,
    )

def extract_center_slices(
    data: np.ndarray,
) -> dict[str, np.ndarray]:
    """Extract center axial, coronal, and sagittal slices."""
    x_center = data.shape[0] // 2
    y_center = data.shape[1] // 2
    z_center = data.shape[2] // 2

    return {
        "axial": np.rot90(data[:, :, z_center]),
        "coronal": np.rot90(data[:, y_center, :]),
        "sagittal": np.rot90(data[x_center, :, :]),
    }

def extract_mask_slices(
    mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Extract center mask slices using the same orientation as images."""
    x_center = mask.shape[0] // 2
    y_center = mask.shape[1] // 2
    z_center = mask.shape[2] // 2

    return {
        "axial": np.rot90(mask[:, :, z_center]),
        "coronal": np.rot90(mask[:, y_center, :]),
        "sagittal": np.rot90(mask[x_center, :, :]),
    }

def image_to_base64_png(
    image_slice: np.ndarray,
    mask_slice: np.ndarray | None = None,
    title: str = "",
) -> str:
    """Render one MRI slice and return a base64 PNG string."""
    normalized = normalize_slice(image_slice)

    figure, axis = plt.subplots(
        figsize=(5, 5)
    )

    axis.imshow(
        normalized,
        cmap="gray",
        interpolation="nearest",
    )

    if mask_slice is not None and np.any(mask_slice):
        overlay = np.ma.masked_where(
            ~mask_slice,
            mask_slice,
        )

        axis.imshow(
            overlay,
            cmap="autumn",
            alpha=0.35,
            interpolation="nearest",
        )

    axis.set_title(title)
    axis.axis("off")

    buffer = io.BytesIO()

    figure.tight_layout()
    figure.savefig(
        buffer,
        format="png",
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(figure)

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("ascii")

    return encoded

def create_mri_visualization(
    nifti_path: Path,
    output_dir: Path,
    mask_path: Path | None = None,
    title_prefix: str = "MRI",
) -> dict[str, Any]:
    """
    Create MRI slice images and a visualization summary.

    Images are embedded as base64 strings in the returned dictionary and are
    also saved as PNG files in output_dir.
    """
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    data, image = load_nifti_image(nifti_path)

    mask = None

    if mask_path is not None:
        mask = load_nifti_mask(
            mask_path=mask_path,
            expected_shape=data.shape,
        )

    image_slices = extract_center_slices(data)

    mask_slices = (
        extract_mask_slices(mask)
        if mask is not None
        else {
            "axial": None,
            "coronal": None,
            "sagittal": None,
        }
    )

    png_paths = {}
    base64_images = {}

    for plane in (
        "axial",
        "coronal",
        "sagittal",
    ):
        png_base64 = image_to_base64_png(
            image_slice=image_slices[plane],
            mask_slice=mask_slices[plane],
            title=f"{title_prefix} - {plane.title()}",
        )

        base64_images[plane] = png_base64

        png_path = output_dir / f"{plane}.png"

        png_path.write_bytes(
            base64.b64decode(png_base64)
        )

        png_paths[plane] = str(
            png_path.resolve()
        )

    zooms = tuple(
        float(value)
        for value in image.header.get_zooms()[:3]
    )

    orientation = "".join(
        nib.orientations.aff2axcodes(
            image.affine
        )
    )

    finite_values = data[
        np.isfinite(data)
    ]

    summary = {
        "nifti_path": str(
            Path(nifti_path).expanduser().resolve()
        ),
        "shape": list(data.shape),
        "voxel_spacing_mm": list(zooms),
        "orientation": orientation,
        "intensity_min": float(
            np.min(finite_values)
        ),
        "intensity_max": float(
            np.max(finite_values)
        ),
        "intensity_mean": float(
            np.mean(finite_values)
        ),
        "nonzero_voxel_count": int(
            np.count_nonzero(data)
        ),
        "mask_path": (
            str(
                Path(mask_path)
                .expanduser()
                .resolve()
            )
            if mask_path is not None
            else None
        ),
        "mask_voxel_count": (
            int(mask.sum())
            if mask is not None
            else None
        ),
        "png_paths": png_paths,
        "base64_images": base64_images,
    }

    with (
        output_dir / "mri_visualization_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                key: value
                for key, value in summary.items()
                if key != "base64_images"
            },
            handle,
            indent=2,
            sort_keys=True,
        )

    return summary

def create_mri_html_report(
    visualization: dict[str, Any],
    output_path: Path,
    title: str = "MRI Visualization",
) -> None:
    """Create a standalone browser-viewable HTML MRI report."""
    images = visualization["base64_images"]

    image_cards = []

    for plane in (
        "axial",
        "coronal",
        "sagittal",
    ):
        image_cards.append(
            f"""
            <div class="image-card">
                <h3>{plane.title()}</h3>
                <img src="data:image/png;base64,{images[plane]}">
            </div>
            """
        )

    image_html = "\n".join(image_cards)

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
body {{
    font-family: Arial, sans-serif;
    margin: 30px;
    background: #f3f5f7;
    color: #222;
}}
.card {{
    background: white;
    padding: 20px;
    margin-bottom: 20px;
    border-radius: 8px;
    box-shadow: 0 1px 5px #bbb;
}}
.grid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 20px;
}}
.image-card {{
    text-align: center;
}}
.image-card img {{
    width: 100%;
    max-width: 420px;
    border: 1px solid #aaa;
}}
table {{
    border-collapse: collapse;
    width: 100%;
}}
td, th {{
    border: 1px solid #ccc;
    padding: 8px;
    text-align: left;
}}
th {{
    background: #26384a;
    color: white;
}}
</style>
</head>
<body>
<div class="card">
<h1>{title}</h1>
<p><strong>NIfTI:</strong> {visualization["nifti_path"]}</p>
<p><strong>Shape:</strong> {visualization["shape"]}</p>
<p><strong>Voxel spacing:</strong> {visualization["voxel_spacing_mm"]} mm</p>
<p><strong>Orientation:</strong> {visualization["orientation"]}</p>
</div>

<div class="card">
<h2>MRI Slices</h2>
<div class="grid">
{image_html}
</div>
</div>

<div class="card">
<h2>Image Statistics</h2>
<table>
<tr><th>Property</th><th>Value</th></tr>
<tr><td>Intensity minimum</td><td>{visualization["intensity_min"]}</td></tr>
<tr><td>Intensity maximum</td><td>{visualization["intensity_max"]}</td></tr>
<tr><td>Intensity mean</td><td>{visualization["intensity_mean"]}</td></tr>
<tr><td>Nonzero voxel count</td><td>{visualization["nonzero_voxel_count"]}</td></tr>
<tr><td>Mask voxel count</td><td>{visualization["mask_voxel_count"]}</td></tr>
</table>
</div>
</body>
</html>
"""

    output_path.write_text(
        html,
        encoding="utf-8",
    )
