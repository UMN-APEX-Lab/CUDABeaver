#include "kernel_launch.h"
#include "cuda_helpers.h"
#include <nvtx3/nvToolsExt.h>

int main() {
    nvtxRangePushA("bench_region");
    launch(256, 256);
    cudaCheckErrors("benchmark kernel launch failed");
    cudaDeviceSynchronize();
    cudaCheckErrors("benchmark kernel sync failed");
    nvtxRangePop();
    return 0;
}
