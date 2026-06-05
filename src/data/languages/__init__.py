"""Language plugin registry.

Auto-imports all built-in language modules on first access so they
self-register.  To add a new language, create ``xx.py`` in this package
and add ``from . import xx`` below.
"""
from .base import LanguagePlugin

_registry: dict[str, LanguagePlugin] = {}


def register_language(plugin: LanguagePlugin) -> None:
    """Register a language plugin instance by its ``code``."""
    _registry[plugin.code] = plugin


def get_language(code: str) -> LanguagePlugin:
    """Return the registered plugin for *code* or raise ``KeyError``."""
    if code not in _registry:
        raise KeyError(
            f"Language '{code}' not registered. "
            f"Available: {list(_registry.keys())}"
        )
    return _registry[code]


def list_languages() -> list[str]:
    """Return all registered language codes."""
    return list(_registry.keys())


# Auto-register built-in language plugins on import.
from . import en, ru, tr  # noqa: E402, F401

__all__ = [
    "LanguagePlugin",
    "register_language",
    "get_language",
    "list_languages",
]
