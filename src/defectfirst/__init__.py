"""DefectFirst research implementation; repository name: DetectFirst."""

import os

# Set before this package initializes CUDA; strict deterministic matmul needs it.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("XFORMERS_DISABLED", "1")

__version__ = "0.1.0"
