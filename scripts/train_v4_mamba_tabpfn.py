#!/usr/bin/env python3
"""Wrapper redirecting to unified 3-class training script."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from train_unified_3class_mamba_tabpfn import main

if __name__ == "__main__":
    main()
