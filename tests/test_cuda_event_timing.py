from harness.benchmark import (
    _rewrite_nvtx_regions_to_cuda_events,
    _rewrite_timed_loop_to_cuda_events,
    _wrap_solution_main_with_cuda_event,
)
from harness.timing_modes.cuda_event_stdout import CudaEventStdout


def test_cuda_event_stdout_parser_uses_mean():
    parsed = CudaEventStdout().parse(
        "CUDA_EVENT_TOTAL_MS=20.0\nCUDA_EVENT_N_ITERS=4\nCUDA_EVENT_MEAN_MS=5.0\n",
        "",
    )
    assert parsed["method"] == "cuda_event_stdout"
    assert parsed["mean_ms"] == 5.0
    assert parsed["total_ms"] == 20.0
    assert parsed["n_instances"] == 4


def test_cuda_event_stdout_parser_derives_mean_from_total():
    parsed = CudaEventStdout().parse(
        "CUDA_EVENT_TOTAL_MS=21.0\nCUDA_EVENT_N_ITERS=7\n",
        "",
    )
    assert parsed["mean_ms"] == 3.0


def test_rewrite_nvtx_region_to_cuda_event():
    source = """
#include <nvtx3/nvToolsExt.h>
#include <cuda_runtime.h>
int main() {
    cudaStream_t stream;
    const int timed_iters = 10;
    nvtxRangePushA("bench_region");
    for (int i = 0; i < timed_iters; i++) {
        kernel<<<1, 1, 0, stream>>>();
    }
    cudaStreamSynchronize(stream);
    nvtxRangePop();
}
"""
    rewritten, count = _rewrite_nvtx_regions_to_cuda_events(source)
    assert count == 1
    assert "nvtxRangePushA" not in rewritten
    assert "CUDA_EVENT_MEAN_MS" in rewritten
    assert "cudaEventRecord(__cudabeaver_start, stream)" in rewritten
    assert "static_cast<double>(timed_iters)" in rewritten


def test_wrap_solution_main_with_cuda_event():
    source = """
#include <cuda_runtime.h>
int main() {
    kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}
"""
    rewritten, wrapped = _wrap_solution_main_with_cuda_event(source)
    assert wrapped
    assert "int __cudabeaver_original_main()" in rewritten
    assert "int main()" in rewritten
    assert "CUDA_EVENT_MEAN_MS" in rewritten


def test_rewrite_timed_loop_without_nvtx():
    source = """
#include <cuda_runtime.h>
void benchmark() {
    cudaStream_t stream;
    const int WARMUP_ITERS = 3;
    const int TIMED_ITERS = 10;
    for (int i = 0; i < WARMUP_ITERS; i++) {
        kernel<<<1, 1, 0, stream>>>();
    }
    for (int i = 0; i < TIMED_ITERS; i++) {
        kernel<<<1, 1, 0, stream>>>();
    }
    cudaStreamSynchronize(stream);
}
"""
    rewritten, count = _rewrite_timed_loop_to_cuda_events(source)
    assert count == 1
    assert "CUDA_EVENT_MEAN_MS" in rewritten
    assert "static_cast<double>(TIMED_ITERS)" in rewritten
    assert "WARMUP_ITERS" in rewritten
