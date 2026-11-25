import sys
from pathlib import Path

# Automatically inject repo root and src/ into sys.path regardless of execution CWD
REPO_ROOT = Path(__file__).resolve().parent.parent
src_dir = str(REPO_ROOT / "src")
root_dir = str(REPO_ROOT)

if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
