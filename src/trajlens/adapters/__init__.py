"""Format adapter sub-package: protocol, registry, and built-in adapters.

Importing this package registers all built-in adapters into the module-level
registry at import time, mirroring checks/__init__.py's pattern.

Public surface:
  - FormatAdapter (Protocol), Capabilities, FormatMatch, WriteResult
  - FormatAdapterRegistry, registry (the singleton), detect_format
  - LeRobotAdapter
"""

from trajlens.adapters.lerobot import LeRobotAdapter
from trajlens.adapters.protocol import Capabilities, FormatAdapter, FormatMatch, WriteResult
from trajlens.adapters.registry import FormatAdapterRegistry, detect_format, registry

registry.register(LeRobotAdapter())

__all__ = [
    "Capabilities",
    "FormatAdapter",
    "FormatAdapterRegistry",
    "FormatMatch",
    "LeRobotAdapter",
    "WriteResult",
    "detect_format",
    "registry",
]
