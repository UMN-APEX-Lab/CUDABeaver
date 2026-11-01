"""Compatibility shim for legacy ``kernels`` timing references.

The benchmark harness no longer profiles with external tools. Old reference
JSON files may still say ``{"type": "kernels"}``, but they are executed through
the same cudaEvent stdout path as new references.
"""

from .cuda_event_stdout import CudaEventStdout


class CudaKernelsAggregate(CudaEventStdout):
    pass
