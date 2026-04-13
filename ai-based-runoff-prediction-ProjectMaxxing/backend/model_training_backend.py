from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_source_file = Path(__file__).resolve().parent.parent / "model_training_backend.py"
if not _source_file.exists():
    raise ImportError(f"Missing source ML module: {_source_file}")

_spec = importlib.util.spec_from_file_location("project_model_training_backend", _source_file)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load source ML module from {_source_file}")

_module = importlib.util.module_from_spec(_spec)
sys.modules["project_model_training_backend"] = _module
_spec.loader.exec_module(_module)

auto_train_best_model = _module.auto_train_best_model
train_lstm_with_early_stopping = getattr(_module, "train_lstm_with_early_stopping", None)
