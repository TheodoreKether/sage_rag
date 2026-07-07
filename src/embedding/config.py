"""Configuration for embedding and vector index construction."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

MODEL_NAME = "BAAI/bge-m3"
BATCH_SIZE = 16
MAX_LENGTH = 8192

# "auto" resolves to cuda when available, otherwise cpu
DEVICE = "auto"

# Optional: point to a locally downloaded model directory instead of HuggingFace Hub.
# When set, overrides MODEL_NAME unless --model is passed on the CLI.
LOCAL_MODEL_PATH: str | None = None


def resolve_device(device: str | None = None) -> str:
    """Return a concrete torch device string, preferring GPU when available."""
    choice = (device or DEVICE).strip().lower()
    if choice != "auto":
        return choice

    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            logger.info("GPU available: %s", name)
            return "cuda"
    except ImportError:
        logger.debug("torch not installed; falling back to CPU")
    except Exception as exc:
        logger.warning("GPU detection failed (%s); falling back to CPU", exc)

    return "cpu"
