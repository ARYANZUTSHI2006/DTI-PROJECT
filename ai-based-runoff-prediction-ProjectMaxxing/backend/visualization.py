from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_source_file = Path(__file__).resolve().parent.parent / "visualization.py"
if not _source_file.exists():
    raise ImportError(f"Missing source visualization module: {_source_file}")

_spec = importlib.util.spec_from_file_location("project_visualization", _source_file)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load source visualization module from {_source_file}")

_module = importlib.util.module_from_spec(_spec)
sys.modules["project_visualization"] = _module
_spec.loader.exec_module(_module)

generate_all_plots = _module.generate_all_plots
