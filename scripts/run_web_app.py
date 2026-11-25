#!/usr/bin/env python3
"""Runner script for the ADNI Multimodal Radiomics & TabPFN Clinical Web Application."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from adni_pipeline.web_server import run_server


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch the ADNI Multimodal Alzheimer's Diagnostic Web Application."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port number to listen on (default: 8000)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Network interface to bind (default: 0.0.0.0)",
    )

    args = parser.parse_args()
    run_server(port=args.port, host=args.host)


if __name__ == "__main__":
    main()
