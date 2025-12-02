"""VolPredictor package wrapper for dash/FastAPI integration."""

# Expose the main run_cycle helper for other modules
from .VolPredictor import run_cycle, main, run_scheduled  # noqa: F401
