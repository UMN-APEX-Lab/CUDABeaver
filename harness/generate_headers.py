"""
Auto-generate stub include/ headers from task prompts.

Each prompt mentions a header like `include/foo.h` and contains code blocks
that define what goes into that header (includes, using declarations, function
signatures). We collect those code blocks and write them out as the header.
"""

import re
from pathlib import Path

try:
    from .task_registry import resolve_task_paths
except ImportError:  # pragma: no cover - direct script execution fallback
    from task_registry import resolve_task_paths


def extract_header_info(prompt_text: str) -> tuple[str | None, str]:
    """
    Parse a prompt and return (header_filename, header_content).

    header_filename: e.g. "reductions.h"  (just the basename, no path)
    header_content:  synthesized content to write into include/<header_filename>
    """
    # 1. Find header filename
    m = re.search(r"`include/([\w./]+\.h)`", prompt_text)
    if not m:
        return None, ""
    header_name = m.group(1)   # e.g. "reductions.h"

    # 2. Find the "already provided and includes:" block first (verbatim header content)
    provided_blocks: list[str] = []
    # Look for pattern: "already provided and includes:\n```...\n```"
    provided = re.findall(
        r"already provided and includes?[:\s]*```(?:cuda|cpp|c\+\+)?\s*\n(.*?)```",
        prompt_text,
        re.DOTALL | re.IGNORECASE,
    )
    provided_blocks.extend(b.strip() for b in provided)

    # 3. Find the signatures block — usually the last code block in the prompt,
    #    after phrases like "Implement the functions" / "using the provided signatures"
    sig_blocks: list[str] = []
    all_blocks = re.findall(
        r"```(?:cuda|cpp|c\+\+)?\s*\n(.*?)```", prompt_text, re.DOTALL
    )
    for block in all_blocks:
        block = block.strip()
        if block in provided_blocks:
            continue
        # A block is a "signatures block" if it contains __device__, __global__,
        # or a plain function declaration (return_type funcname(...);)
        if re.search(r"(__device__|__global__|^\w[\w\s\*]+\w\s*\(.*\)\s*;)", block, re.MULTILINE):
            sig_blocks.append(block)

    # Build header content
    parts: list[str] = [f"// Auto-generated stub header for {header_name}"]
    parts.append(f"#pragma once")

    for blk in provided_blocks:
        parts.append(blk)

    if sig_blocks:
        parts.append("")
        parts.append("// Function signatures")
        for blk in sig_blocks:
            parts.append(blk)

    return header_name, "\n".join(parts) + "\n"



# ---------------------------------------------------------------------------
# Hardcoded helper .cu files for tasks that provide pre-written kernels.
# These exist in the original task workspace but are NOT in the testbench export.
# ---------------------------------------------------------------------------

CUDA_199_KERNELS_CU = """\
#include <stdint.h>

// Provided kernel: 3x3 morphological erosion (min of 3x3 neighborhood)
__global__ void k_erosion(uint8_t* input, uint8_t* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    uint8_t minVal = 255;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = x + dx, ny = y + dy;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height)
                minVal = min(minVal, input[ny * width + nx]);
            else
                minVal = 0;
        }
    }
    output[y * width + x] = minVal;
}

// Provided kernel: 3x3 morphological dilation (max of 3x3 neighborhood)
__global__ void k_dilation(uint8_t* input, uint8_t* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    uint8_t maxVal = 0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = x + dx, ny = y + dy;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height)
                maxVal = max(maxVal, input[ny * width + nx]);
        }
    }
    output[y * width + x] = maxVal;
}
"""

CUDA_199_MORPHOLOGY_H = """\
#pragma once
#include <stdint.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \\
    cudaError_t _e = (call); \\
    if (_e != cudaSuccess) { \\
        fprintf(stderr, "CUDA error %s:%d: %s\\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \\
        exit(1); \\
    } \\
} while(0)

__global__ void k_erosion(uint8_t* input, uint8_t* output, int width, int height);
__global__ void k_dilation(uint8_t* input, uint8_t* output, int width, int height);

void run(uint8_t* inputImage_h, uint8_t* inputImage_d,
         uint8_t* erosionOutputImage_d, uint8_t* dilationOutputImage_d,
         uint8_t* erosionOutputImage_h, uint8_t* dilationOutputImage_h,
         int width, int height,
         cudaStream_t stream1, cudaStream_t stream2, cudaEvent_t copyEnd);
"""

CUDA_210_HELPER_KERNELS_CU = """\
#include <cuda_runtime.h>
#include <math.h>

// Cross product z-component for 2D vectors (p1->p2) x (p1->pt)
__device__ float cross2d(float ax, float ay, float bx, float by) {
    return ax * by - ay * bx;
}

// Point-in-triangle test (winding number)
__device__ bool pointInTriangle(float px, float py,
                                 float ax, float ay,
                                 float bx, float by,
                                 float cx, float cy) {
    float d1 = cross2d(bx - ax, by - ay, px - ax, py - ay);
    float d2 = cross2d(cx - bx, cy - by, px - bx, py - by);
    float d3 = cross2d(ax - cx, ay - cy, px - cx, py - cy);
    bool has_neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
    bool has_pos = (d1 > 0) || (d2 > 0) || (d3 > 0);
    return !(has_neg && has_pos);
}

// Point-in-axis-aligned rectangle (4 vertices, ordered)
__device__ bool pointInRectangle(float px, float py,
                                  float* rectX, float* rectY, int rectIdx) {
    float minX = rectX[rectIdx * 4], maxX = minX;
    float minY = rectY[rectIdx * 4], maxY = minY;
    for (int i = 1; i < 4; i++) {
        float x = rectX[rectIdx * 4 + i], y = rectY[rectIdx * 4 + i];
        minX = fminf(minX, x); maxX = fmaxf(maxX, x);
        minY = fminf(minY, y); maxY = fmaxf(maxY, y);
    }
    return px >= minX && px <= maxX && py >= minY && py <= maxY;
}

__global__ void k_countCollisionsWithTriangles(
    int* pointCollisionCount_d,
    float* pointPositionX_d, float* pointPositionY_d,
    float* triangleVertexX_d, float* triangleVertexY_d,
    int numPoints, int numTriangles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numPoints) return;
    float px = pointPositionX_d[i], py = pointPositionY_d[i];
    for (int t = 0; t < numTriangles; t++) {
        if (pointInTriangle(px, py,
                            triangleVertexX_d[t*3],   triangleVertexY_d[t*3],
                            triangleVertexX_d[t*3+1], triangleVertexY_d[t*3+1],
                            triangleVertexX_d[t*3+2], triangleVertexY_d[t*3+2]))
            atomicAdd(&pointCollisionCount_d[i], 1);
    }
}

__global__ void k_countCollisionsWithRectangles(
    int* pointCollisionCount_d,
    float* pointPositionX_d, float* pointPositionY_d,
    float* rectangleVertexX_d, float* rectangleVertexY_d,
    int numPoints, int numRectangles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numPoints) return;
    float px = pointPositionX_d[i], py = pointPositionY_d[i];
    for (int r = 0; r < numRectangles; r++) {
        if (pointInRectangle(px, py, rectangleVertexX_d, rectangleVertexY_d, r))
            atomicAdd(&pointCollisionCount_d[i], 1);
    }
}
"""

CUDA_210_COLLISION_H = """\
#pragma once
#include <cuda_runtime.h>

__global__ void k_countCollisionsWithTriangles(
    int* pointCollisionCount_d,
    float* pointPositionX_d, float* pointPositionY_d,
    float* triangleVertexX_d, float* triangleVertexY_d,
    int numPoints, int numTriangles);

__global__ void k_countCollisionsWithRectangles(
    int* pointCollisionCount_d,
    float* pointPositionX_d, float* pointPositionY_d,
    float* rectangleVertexX_d, float* rectangleVertexY_d,
    int numPoints, int numRectangles);

void run(int numPoints, int numTriangles, int numRectangles,
         cudaStream_t stream1, cudaStream_t stream2,
         cudaEvent_t event1, cudaEvent_t event2,
         int* pointCollisionCount_h, float* pointPositionX_h, float* pointPositionY_h,
         int* pointCollisionCount_d, float* pointPositionX_d, float* pointPositionY_d,
         float* triangleVertexX_d, float* triangleVertexY_d,
         float* rectangleVertexX_d, float* rectangleVertexY_d);
"""

CUDA_210_CUDA_UTILS_H = """\
#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call) do { \\
    cudaError_t _e = (call); \\
    if (_e != cudaSuccess) { \\
        fprintf(stderr, "CUDA error %s:%d: %s\\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \\
        exit(1); \\
    } \\
} while(0)
"""

# ---------------------------------------------------------------------------
# Shared utility headers referenced by multiple test_main.cu files.
# All variations (cuda_helpers.h, cuda_utils.h, cuda_check.h) are
# essentially the same: a CUDA_CHECK / CHECK_CUDA error macro.
# ---------------------------------------------------------------------------

_CUDA_CHECK_H = """\
#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cusparse.h>
#include <cstdio>
#include <cstdlib>

#ifndef TOLERANCE
#define TOLERANCE 1e-3
#endif

#ifndef EPSILON
#define EPSILON 1e-5f
#endif

#define CUDA_CHECK(call) do { \\
    cudaError_t _e = (call); \\
    if (_e != cudaSuccess) { \\
        fprintf(stderr, "CUDA error %s:%d: %s\\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \\
        exit(1); \\
    } \\
} while(0)

#define CHECK_CUDA(call) CUDA_CHECK(call)

#define CHECK_CUBLAS(call) do { \\
    cublasStatus_t _s = (call); \\
    if (_s != CUBLAS_STATUS_SUCCESS) { \\
        fprintf(stderr, "cuBLAS error %s:%d: %d\\n", __FILE__, __LINE__, (int)_s); \\
        exit(1); \\
    } \\
} while(0)

#define CURAND_CHECK(call) do { \\
    curandStatus_t _s = (call); \\
    if (_s != CURAND_STATUS_SUCCESS) { \\
        fprintf(stderr, "cuRAND error %s:%d: %d\\n", __FILE__, __LINE__, (int)_s); \\
        exit(1); \\
    } \\
} while(0)

#define CUSPARSE_CHECK(call) do { \\
    cusparseStatus_t _s = (call); \\
    if (_s != CUSPARSE_STATUS_SUCCESS) { \\
        fprintf(stderr, "cuSPARSE error %s:%d: %d\\n", __FILE__, __LINE__, (int)_s); \\
        exit(1); \\
    } \\
} while(0)
"""

# ---------------------------------------------------------------------------
# Task-specific headers for the 14 failing tasks
# ---------------------------------------------------------------------------

# CUDA_138: ray_sphere.h — constants, Sphere/Ray structs, d_intersect, kernel decl
CUDA_138_RAY_SPHERE_H = """\
#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

// Constants required by the testbench
constexpr int NUM_TOTAL_RAYS     = 16;
constexpr int NUM_TOTAL_SPHERES  = 2;
constexpr int SCREEN_WIDTH       = 4;
constexpr int SCREEN_HEIGHT      = 4;
constexpr float BACKGROUND_COLOR = 0.5f;

// Sphere data structure (loaded into shared memory tiles)
struct Sphere {
    float x, y, z;
    float radius;
    float color;
};

// Ray state (per-thread)
struct Ray {
    float originX, originY, originZ;
    float dirX, dirY, dirZ;
    float currentDistance;  // -1.0f means no hit yet
    float color;
};

// Provided device helper: test single ray vs single sphere intersection.
// Requires ray direction to be normalized to unit length before calling.
// Updates ray.currentDistance and ray.color if a closer intersection is found.
inline __device__ void d_intersect(Ray& ray, Sphere& sphere) {
    if (sphere.radius <= 0.0f) return;
    float wx = ray.originX - sphere.x;
    float wy = ray.originY - sphere.y;
    float wz = ray.originZ - sphere.z;
    // b = dot(D, W) where D is unit direction
    float b = wx * ray.dirX + wy * ray.dirY + wz * ray.dirZ;
    float c = wx*wx + wy*wy + wz*wz - sphere.radius * sphere.radius;
    float disc = b*b - c;
    if (disc < 0.0f) return;
    float sqrtD = sqrtf(disc);
    float t = -b - sqrtD;
    if (t < 0.0f) t = -b + sqrtD;
    if (t < 0.0f) return;
    if (ray.currentDistance < 0.0f || t < ray.currentDistance) {
        ray.currentDistance = t;
        ray.color = sphere.color;
    }
}

__global__ void k_calculateRaySphereIntersection(
    float *rayOriginX_d, float *rayOriginY_d, float *rayOriginZ_d,
    float *rayDirectionX_d, float *rayDirectionY_d, float *rayDirectionZ_d,
    float *sphereX_d, float *sphereY_d, float *sphereZ_d,
    float *sphereRadius_d, float *sphereColor_d,
    float *rayColorOutput_d);
"""

# CUDA_140: cross_correlation.h — BLOCK_SIZE constant + kernel declaration
CUDA_140_CROSS_CORR_H = """\
#pragma once
#include <cuda_runtime.h>

constexpr int BLOCK_SIZE = 256;

__global__ void k_measureSimilarity(int *inputVector, int inputSize,
                                    int *referenceVector, int refVectorSize,
                                    int *output);
"""

# CUDA_145: dot_products.h — all constants + CUDA_CHECK + kernel declaration
CUDA_145_DOT_PRODUCTS_H = """\
#pragma once
#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>

// Tuning constants
constexpr int   NUM_THREADS_PER_BLOCK              = 128;
constexpr int   MAXIMUM_NUMBER_OF_ELEMENTS_PER_VECTOR = 1000;
constexpr int   MAXIMUM_NUMBER_OF_VECTORS             = 10000;
#define ALLOCATION_SIZE ((size_t)MAXIMUM_NUMBER_OF_ELEMENTS_PER_VECTOR * MAXIMUM_NUMBER_OF_VECTORS * sizeof(float))

// Vectorized-load widths
constexpr int FLOAT4_ELEMENTS = 4;
constexpr int FLOAT2_ELEMENTS = 2;

// Error thresholds
constexpr float RELATIVE_ERROR              = 1e-3f;
constexpr float SMOOTHING_FOR_DIVIDE_BY_ZERO = 1e-7f;

#define CUDA_CHECK(call) do { \\
    cudaError_t _e = (call); \\
    if (_e != cudaSuccess) { \\
        fprintf(stderr, "CUDA error %s:%d: %s\\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \\
        exit(1); \\
    } \\
} while(0)

__global__ void k_runDotProducts(
    const float *const __restrict__ commonVector_d,
    const float *const __restrict__ otherVectors_d,
    float *const __restrict__ result_d,
    int numVectors,
    int numElementsPerVector);
"""

# CUDA_162: symmetric_rank.h — cublas includes + function declaration
CUDA_162_SYMMETRIC_RANK_H = """\
#pragma once
#include <cublas_v2.h>
#include <cuComplex.h>

void symmetricRank(cublasHandle_t handle,
                   int n,
                   int k,
                   const cuComplex* alpha,
                   const cuComplex* A,
                   const cuComplex* B,
                   const cuComplex* beta,
                   cuComplex* C);
"""

# CUDA_200: k_compute_powers.h — VECTOR_WIDTH, EXPONENT_START + kernel declaration
CUDA_200_K_COMPUTE_POWERS_H = """\
#pragma once
#include <cuda_runtime.h>

constexpr int VECTOR_WIDTH   = 4;
constexpr int EXPONENT_START = 2;

__global__ void k_computeConsecutivePowers(const float *input_d,
                                           float4 *output_d4,
                                           int exponent,
                                           int numElements);
"""

# CUDA_205: sort_segments.h — CUB includes + function declaration
CUDA_205_SORT_SEGMENTS_H = """\
#pragma once
#include <cub/cub.cuh>
#include <cuda_runtime.h>

void sortSegments(
    cub::DoubleBuffer<int>& keys_d,
    cub::DoubleBuffer<int>& values_d,
    int numberOfItems,
    int numberOfSegments,
    int* beginningOfOffset_d,
    int* endOfOffset_d,
    cudaStream_t stream);
"""

# CUDA_85: place_in_order.h — thrust includes + function declaration
CUDA_85_PLACE_IN_ORDER_H = """\
#pragma once
#include <thrust/device_vector.h>

int place_in_order(thrust::device_vector<int> const &vec, int val);
"""

# cublas_level2_20: solve_tpsv.h — cublas + error macros + function declaration
CUBLAS_LEVEL2_20_SOLVE_TPSV_H = """\
#pragma once
#include <vector>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call) do { \\
    cudaError_t _e = (call); \\
    if (_e != cudaSuccess) { \\
        fprintf(stderr, "CUDA error %s:%d: %s\\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \\
        exit(1); \\
    } \\
} while(0)

#define CHECK_CUBLAS(call) do { \\
    cublasStatus_t _s = (call); \\
    if (_s != CUBLAS_STATUS_SUCCESS) { \\
        fprintf(stderr, "cuBLAS error %s:%d: %d\\n", __FILE__, __LINE__, (int)_s); \\
        exit(1); \\
    } \\
} while(0)

void solve_tpsv(int n, const std::vector<double>& AP, std::vector<double>& x,
                int incx);
"""

# cublas_level3_7: herk_example.h — data_type typedef + cublas + function declaration
# (test/test_utils.h includes this header to get data_type)
# data_type = cuDoubleComplex because test_utils.cu uses cuCreal/cuCimag (double versions)
# and the kernel is cublasZherk (Z = double-precision complex)
CUBLAS_LEVEL3_7_HERK_EXAMPLE_H = """\
#pragma once
#include <vector>
#include <cublas_v2.h>
#include <cuComplex.h>
#include <cstdio>
#include <cstdlib>

using data_type = cuDoubleComplex;

#define CUDA_CHECK(call) do { \\
    cudaError_t _e = (call); \\
    if (_e != cudaSuccess) { \\
        fprintf(stderr, "CUDA error %s:%d: %s\\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \\
        exit(1); \\
    } \\
} while(0)

#define CHECK_CUBLAS(call) do { \\
    cublasStatus_t _s = (call); \\
    if (_s != CUBLAS_STATUS_SUCCESS) { \\
        fprintf(stderr, "cuBLAS error %s:%d: %d\\n", __FILE__, __LINE__, (int)_s); \\
        exit(1); \\
    } \\
} while(0)

void herk_example(int n, int k, const std::vector<data_type>& A,
                  std::vector<data_type>& C, double alpha, double beta,
                  cublasFillMode_t uplo, cublasOperation_t transa, int lda,
                  int ldc);
"""

# curand_22: curand_sobol64_poisson.h — function declaration (macros from cuda_helpers.h)
CURAND_22_SOBOL64_POISSON_H = """\
#pragma once
#include <vector>
#include <iostream>
#include <curand.h>
#include <cuda_runtime.h>

void generate_scrambled_sobol64_poisson(int n, double lambda,
                                        unsigned long long offset,
                                        curandOrdering_t order,
                                        std::vector<unsigned int>& h_data);
"""

# curand_8: generate_lognormal.h — error macros + function declaration
# (test includes ONLY this header, so macros must live here)
CURAND_8_GENERATE_LOGNORMAL_H = """\
#pragma once
#include <vector>
#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
#include <curand.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call) do { \\
    cudaError_t _e = (call); \\
    if (_e != cudaSuccess) { \\
        fprintf(stderr, "CUDA error %s:%d: %s\\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \\
        exit(1); \\
    } \\
} while(0)

#define CURAND_CHECK(call) do { \\
    curandStatus_t _s = (call); \\
    if (_s != CURAND_STATUS_SUCCESS) { \\
        fprintf(stderr, "cuRAND error %s:%d: %d\\n", __FILE__, __LINE__, (int)_s); \\
        exit(1); \\
    } \\
} while(0)

void generate_lognormal(int n, float mean, float stddev,
                        unsigned long long seed, curandOrdering_t order,
                        std::vector<float>& h_data);
"""

# cusparse_0: axpby_example.h — function declaration (macros from cuda_helpers.h)
CUSPARSE_0_AXPBY_H = """\
#pragma once
#include <vector>
#include <cusparse.h>
#include <cuda_runtime.h>

void axpby_example(int size, int nnz, const std::vector<int>& hX_indices,
                   const std::vector<float>& hX_values,
                   const std::vector<float>& hY, float alpha, float beta,
                   std::vector<float>& hY_result);
"""

# cusparse_10: sddmm_csr.h — function declaration (macros from cuda_helpers.h)
CUSPARSE_10_SDDMM_CSR_H = """\
#pragma once
#include <vector>
#include <cusparse.h>
#include <cuda_runtime.h>

void sddmm_csr_batched_example(int A_num_rows, int A_num_cols,
                               int B_num_rows, int B_num_cols,
                               int C_nnz, int lda, int ldb,
                               int num_batches,
                               const std::vector<float>& hA,
                               const std::vector<float>& hB,
                               const std::vector<int>& hC_offsets,
                               const std::vector<int>& hC_columns,
                               std::vector<float>& hC_values);
"""


EXTRA_FILES: dict[str, dict[str, str]] = {
    # --- Tasks with pre-written helper kernels not in testbench export ---
    "CUDA_199": {
        "include/kernels.cu":   CUDA_199_KERNELS_CU,
        "include/morphology.h": CUDA_199_MORPHOLOGY_H,
    },
    "CUDA_210": {
        "helper/kernels.cu":    CUDA_210_HELPER_KERNELS_CU,
        "include/collision.h":  CUDA_210_COLLISION_H,
        "include/cuda_utils.h": CUDA_210_CUDA_UTILS_H,
    },

    # --- Tasks needing cuda_helpers.h (shared CUDA_CHECK / CURAND_CHECK / etc.) ---
    "CUDA_112":    {"include/cuda_helpers.h": _CUDA_CHECK_H},
    "CUDA_224":    {"include/cuda_check.h": _CUDA_CHECK_H},

    # --- Tasks with their own full headers ---
    "CUDA_138": {
        "include/ray_sphere.h": CUDA_138_RAY_SPHERE_H,
    },
    "CUDA_140": {
        "include/cross_correlation.h": CUDA_140_CROSS_CORR_H,
    },
    "CUDA_145": {
        "include/dot_products.h": CUDA_145_DOT_PRODUCTS_H,
    },
    "CUDA_162": {
        "include/cuda_utils.h":    _CUDA_CHECK_H,
        "include/symmetric_rank.h": CUDA_162_SYMMETRIC_RANK_H,
    },
    "CUDA_200": {
        "include/k_compute_powers.h": CUDA_200_K_COMPUTE_POWERS_H,
    },
    "CUDA_205": {
        "include/cuda_helpers.h":  _CUDA_CHECK_H,
        "include/sort_segments.h": CUDA_205_SORT_SEGMENTS_H,
    },
    "CUDA_85": {
        "include/place_in_order.h": CUDA_85_PLACE_IN_ORDER_H,
    },
    "cublas_level2_20": {
        "include/solve_tpsv.h": CUBLAS_LEVEL2_20_SOLVE_TPSV_H,
    },
    "cublas_level3_7": {
        "include/herk_example.h": CUBLAS_LEVEL3_7_HERK_EXAMPLE_H,
    },
    "curand_22": {
        "include/cuda_helpers.h":          _CUDA_CHECK_H,
        "include/curand_sobol64_poisson.h": CURAND_22_SOBOL64_POISSON_H,
    },
    "curand_8": {
        "include/generate_lognormal.h": CURAND_8_GENERATE_LOGNORMAL_H,
    },
    "cusparse_0": {
        "include/cuda_helpers.h": _CUDA_CHECK_H,
        "include/axpby_example.h": CUSPARSE_0_AXPBY_H,
    },
    "cusparse_10": {
        "include/cuda_helpers.h": _CUDA_CHECK_H,
        "include/sddmm_csr.h":    CUSPARSE_10_SDDMM_CSR_H,
    },
}


_LIBRARY_MACRO_SUFFIX: dict[str, str] = {
    "-lcurand": "\n#include <curand.h>\n#define CURAND_CHECK(call) do { \\\n"
                "    curandStatus_t _s = (call); \\\n"
                "    if (_s != CURAND_STATUS_SUCCESS) { \\\n"
                "        fprintf(stderr, \"cuRAND error %s:%d: %d\\\\n\", __FILE__, __LINE__, (int)_s); \\\n"
                "        exit(1); \\\n"
                "    } \\\n} while(0)\n",
    "-lcusparse": "\n#include <cusparse.h>\n#define CUSPARSE_CHECK(call) do { \\\n"
                  "    cusparseStatus_t _s = (call); \\\n"
                  "    if (_s != CUSPARSE_STATUS_SUCCESS) { \\\n"
                  "        fprintf(stderr, \"cuSPARSE error %s:%d: %d\\\\n\", __FILE__, __LINE__, (int)_s); \\\n"
                  "        exit(1); \\\n"
                  "    } \\\n} while(0)\n",
    "-lcublas": "\n#include <cublas_v2.h>\n#define CHECK_CUBLAS(call) do { \\\n"
                "    cublasStatus_t _s = (call); \\\n"
                "    if (_s != CUBLAS_STATUS_SUCCESS) { \\\n"
                "        fprintf(stderr, \"cuBLAS error %s:%d: %d\\\\n\", __FILE__, __LINE__, (int)_s); \\\n"
                "        exit(1); \\\n"
                "    } \\\n} while(0)\n",
}


def setup_include_dir(task_stem: str, dataset_dir: Path, task_workdir: Path) -> None:
    """
    Create include/ (and helper/) in task_workdir and write all needed support files.
    """
    task_paths = resolve_task_paths(task_stem, dataset_dir)

    # Read build command to know which libraries are linked
    input_path = task_paths.input_path
    build_cmd = ""
    if input_path.exists():
        import json as _json
        build_cmd = _json.loads(input_path.read_text())["build_command"]

    # 1. Auto-generated header from prompt; append library check macros if needed
    prompt_path = task_paths.prompt_path
    if prompt_path.exists():
        prompt_text = prompt_path.read_text(encoding="utf-8")
        header_name, header_content = extract_header_info(prompt_text)
        if header_name:
            include_dir = task_workdir / "include"
            include_dir.mkdir(exist_ok=True)
            target = include_dir / header_name
            # If testbench shipped a real header for this name (applied-kernels and v2
            # tasks do — applied-kernels copies the original task's include/ tree), keep
            # it. Auto-generated stubs would otherwise blow away a working
            # signature with `// Auto-generated stub header for foo.h`.
            if target.exists() and target.read_text().strip():
                pass  # keep shipped header
            else:
                # Append library-specific check macros based on build command
                for flag, macro_snippet in _LIBRARY_MACRO_SUFFIX.items():
                    if flag in build_cmd:
                        header_content += macro_snippet
                target.write_text(header_content, encoding="utf-8")

    # 2. Hardcoded helper files for special tasks (overwrite auto-generated if same name)
    for rel_path, content in EXTRA_FILES.get(task_stem, {}).items():
        dest = task_workdir / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    # Quick test: print generated headers for all tasks
    import sys
    dataset_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("../../CUDA-Debugger")
    for prompt_file in sorted((dataset_dir / "prompts").glob("*.txt")):
        stem = prompt_file.stem
        text = prompt_file.read_text(encoding="utf-8")
        name, content = extract_header_info(text)
        print(f"\n{'='*60}")
        print(f"Task: {stem}  ->  include/{name}")
        print(f"{'='*60}")
        print(content[:400])
