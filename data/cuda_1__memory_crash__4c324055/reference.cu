// file: solution.cu
#include "kernel_launch.h"
#include "kernel_helpers.h"
#include <cuda_runtime.h>

void launch(int gridSizeX, int blockSizeX, int gridSizeY, int blockSizeY, int gridSizeZ,
            int blockSizeZ)
{
    dim3 grid(gridSizeX, gridSizeY, gridSizeZ);
    dim3 block(blockSizeX, blockSizeY, blockSizeZ);
    size_t count = size_t(grid.x) * grid.y * grid.z * block.x * block.y * block.z;

    int *output = nullptr;
    int *input = nullptr;
    cudaMalloc(&output, count * sizeof(int));
    cudaMalloc(&input, count * sizeof(int));
    cudaMemset(input, 0, count * sizeof(int));

    int sharedMemory = blockSizeX * blockSizeY * blockSizeZ * sizeof(int);
    kernel<<<grid, block, sharedMemory>>>(output, input);
    cudaDeviceSynchronize();

    cudaFree(output);
    cudaFree(input);
}
