"""Compatibility shim.

This refactor keeps only the GME multimodal retriever. Text-only encoding is handled
through the same GME encoder, so this file only re-exports the factory.
"""

from .visual_encoder import GMEEncoder, create_encoder

__all__ = ["GMEEncoder", "create_encoder"]
