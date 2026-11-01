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

__global__ void conv_transpose2d_kernel(const float* A, const float* W, float* C) {
    const int CIN = 64;
    const int COUT = 64;
    const int IN_H = 384;
    const int IN_W = 384;
    const int OUT_H = 768;
    const int OUT_W = 768;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = size_t(COUT) * OUT_H * OUT_W;
    if (idx >= total) {
        return;
    }

    int ow = idx % OUT_W;
    int oh = (idx / OUT_W) % OUT_H;
    int co = idx / (OUT_H * OUT_W);
    int kh = oh & 1;
    int kw = ow & 1;
    int ih = oh >> 1;
    int iw = ow >> 1;

    float sum = 0.0f;
    for (int ci = 0; ci < CIN; ++ci) {
        float x = A[(ci * IN_H + ih) * IN_W + iw];
        float w = W[((ci * COUT + co) * 2 + kh) * 2 + kw];
        sum += x * w;
    }
    C[idx] = sum;
}

int main() {
    const int M = 147456;
    const int K = 64;
    const int N = 256;
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

    int threads = 256;
    int blocks = (size_C + threads - 1) / threads;
    conv_transpose2d_kernel<<<blocks, threads>>>(d_A, d_B, d_C);
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
