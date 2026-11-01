"""Compatibility shim for legacy ``region`` timing references.

The benchmark harness no longer profiles with external tools. Old reference
JSON files may still say ``{"type": "region"}``, but they are executed through
the same cudaEvent stdout path as new references.
"""

from .cuda_event_stdout import CudaEventStdout


class NvtxRegion(CudaEventStdout):
    def __init__(self, include=None, time_type=None):
        self.include = include or []
        self.time_type = time_type
