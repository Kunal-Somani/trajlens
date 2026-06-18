"""trajlens — the quality and synthesis layer for the open robot-learning data ecosystem."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__: str = version("trajlens")
except PackageNotFoundError:
    # Package is not installed (running from source without pip install -e .)
    __version__ = "0.0.0.dev0+local"

__all__ = ["__version__"]
