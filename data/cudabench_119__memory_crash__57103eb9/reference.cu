#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <string>

static void read_binary(const std::string& filename, float* data, size_t size) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Can not open: " << filename << std::endl;
        std::exit(1);
    }
    in.read(reinterpret_cast<char*>(data), size * sizeof(float));
}

static void write_binary(const std::string& filename, const float* data, size_t size) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        std::cerr << "Can not write: " << filename << std::endl;
        std::exit(1);
    }
    out.write(reinterpret_cast<const char*>(data), size * sizeof(float));
}

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
    constexpr int TM = 16;
    constexpr int TN = 16;
    constexpr int TK = 16;
    __shared__ float As[TM][TK];
    __shared__ float Bs[TK][TN];

    int row = blockIdx.y * TM + threadIdx.y;
    int col = blockIdx.x * TN + threadIdx.x;
    float sum = 0.0f;

    for (int base = 0; base < K; base += TK) {
        int a_col = base + threadIdx.x;
        int b_row = base + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TK; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

int main() {
    const int M = 262144;
    const int K = 4096;
    const int N = 2048;
    const size_t size_A = size_t(M) * K;
    const size_t size_B = size_t(K) * N;
    const size_t size_C = size_t(M) * N;

    float* h_A = new float[size_A];
    float* h_B = new float[size_B];
    float* h_C = new float[size_C];
    read_binary("data/matA.bin", h_A, size_A);
    read_binary("data/matB.bin", h_B, size_B);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    cudaMalloc(&d_A, size_A * sizeof(float));
    cudaMalloc(&d_B, size_B * sizeof(float));
    cudaMalloc(&d_C, size_C * sizeof(float));
    cudaMemcpy(d_A, h_A, size_A * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    matmul_kernel<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, size_C * sizeof(float), cudaMemcpyDeviceToHost);
    write_binary("data/matC_out.bin", h_C, size_C);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    return 0;
}
