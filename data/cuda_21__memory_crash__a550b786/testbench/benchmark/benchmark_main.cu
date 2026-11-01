#include "stencil3d.h"
#include <cuda_runtime.h>
#include <iostream>
#include <nvtx3/nvToolsExt.h>

int main() {
    constexpr unsigned int N = 128;
    const size_t count = size_t(N) * N * N;
    float *input = nullptr;
    float *output = nullptr;
    cudaMalloc(&input, count * sizeof(float));
    cudaMalloc(&output, count * sizeof(float));
    cudaMemset(input, 0, count * sizeof(float));
    cudaMemset(output, 0, count * sizeof(float));

    dim3 threads(BLOCK_DIM, BLOCK_DIM, BLOCK_DIM);
    dim3 blocks((N + BLOCK_DIM - 1) / BLOCK_DIM,
                (N + BLOCK_DIM - 1) / BLOCK_DIM,
                (N + BLOCK_DIM - 1) / BLOCK_DIM);
    nvtxRangePushA("bench_region");
    stencil3d_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
    nvtxRangePop();

    cudaError_t err = cudaGetLastError();
    cudaFree(input);
    cudaFree(output);
    if (err != cudaSuccess) {
        std::cerr << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    return 0;
}
