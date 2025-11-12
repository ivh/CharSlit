try:
    from .charslit import extract
    __all__ = ["extract"]
except ImportError:
    # C extension not built, only Python version available
    __all__ = []
