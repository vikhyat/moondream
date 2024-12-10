from importlib import metadata

try:
    __version__ = metadata.version("moondream")
except Exception:
    __version__ = "unknown"
