__all__ = ["ft"]


def __getattr__(name):
    if name == "ft":
        from importlib import import_module

        return import_module(".ft", __name__).ft
    if name == "hf":
        from importlib import import_module

        return import_module(".hf", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")