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

__global__ void conv2d_valid_kernel(const float* input, const float* kernel, float* output) {
    const int IN_W = 2048;
    const int OUT_H = 2025;
    const int OUT_W = 2025;
    const int KH = 24;
    const int KW = 24;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = OUT_H * OUT_W;
    if (idx >= total) {
        return;
    }
    int oy = idx / OUT_W;
    int ox = idx % OUT_W;
    float sum = 0.0f;
    for (int ky = 0; ky < KH; ++ky) {
        for (int kx = 0; kx < KW; ++kx) {
            sum += input[(oy + ky) * IN_W + ox + kx] * kernel[ky * KW + kx];
        }
    }
    output[idx] = sum;
}

int main() {
    const size_t input_size = size_t(2048) * 2048;
    const size_t kernel_size = size_t(24) * 24;
    const size_t output_size = size_t(2025) * 2025;

    float* h_input = new float[input_size];
    float* h_kernel = new float[kernel_size];
    float* h_output = new float[output_size];
    read_binary("data/conv_input.bin", h_input, input_size);
    read_binary("data/conv_kernel.bin", h_kernel, kernel_size);

    float *d_input = nullptr, *d_kernel = nullptr, *d_output = nullptr;
    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_kernel, kernel_size * sizeof(float));
    cudaMalloc(&d_output, output_size * sizeof(float));
    cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernel_size * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (output_size + threads - 1) / threads;
    conv2d_valid_kernel<<<blocks, threads>>>(d_input, d_kernel, d_output);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);
    write_binary("data/conv_output.bin", h_output, output_size);

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
    delete[] h_input;
    delete[] h_kernel;
    delete[] h_output;
    return 0;
}
