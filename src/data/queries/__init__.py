"""Query generation package.

Provides a generic engine that generates template-based queries for any
registered language via the language plugin registry.
"""
from .engine import generate, generate_for_language

__all__ = ["generate", "generate_for_language"]
