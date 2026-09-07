from pathlib import Path
import nibabel as nib
import numpy as np

from adni_pipeline.single_pair_pipeline import run_live_single_pair_inference

def test_live_inference_imports() -> None:
    # Verify module loads
    from adni_pipeline.single_pair_pipeline import LiveInferenceResult
    assert LiveInferenceResult is not None
