from pathlib import Path
import numpy as np

from adni_pipeline.segmentation import (
    OtsuMorphologicalBackend,
    compute_segmentation_metrics,
)

def test_otsu_morphological_backend() -> None:
    data = np.zeros((30, 30, 30), dtype=np.float32)
    # Synthetic sphere
    z, y, x = np.ogrid[:30, :30, :30]
    mask = (x - 15) ** 2 + (y - 15) ** 2 + (z - 15) ** 2 <= 100
    data[mask] = np.random.uniform(5.0, 50.0, size=np.sum(mask))

    backend = OtsuMorphologicalBackend()
    segmented = backend.segment(data)

    assert segmented.shape == (30, 30, 30)
    assert np.any(segmented == 1)
    assert np.all(segmented[~mask] == 0)

def test_segmentation_metrics() -> None:
    gt = np.zeros((20, 20, 20), dtype=np.uint8)
    pred = np.zeros((20, 20, 20), dtype=np.uint8)

    gt[5:15, 5:15, 5:15] = 1
    pred[5:15, 5:15, 5:15] = 1

    metrics = compute_segmentation_metrics(gt, pred)

    assert metrics.dice == 1.0
    assert metrics.iou == 1.0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.sensitivity == 1.0
    assert metrics.specificity == 1.0
    assert metrics.hd95 == 0.0
