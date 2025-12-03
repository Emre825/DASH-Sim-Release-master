#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <unistd.h>
#include <stdint.h>
#include <time.h>

#include "include/cedr_types.h"

#define SEC2NANOSEC 1000000000

__global__ void softmax_kernel(float* input, float* output, int cols) {
    // Each block handles one row, row index based on block index.
    int row_idx = blockIdx.x;
    
    // Pointer to the start of this row
    float* row_input = input + row_idx * cols;
    float* row_output = output + row_idx * cols;

    // Shared memory for reduction
    extern __shared__ float shared[]; 

    int tid = threadIdx.x;
    int block_size = blockDim.x;

    // 1. Find MAX for numerical stability
    float local_max = -1e20f; // Neg infinity
    
    // Grid-Stride Loop: Handle rows larger than block size
    for (int i = tid; i < cols; i += block_size) {
        local_max = max(local_max, row_input[i]);
    }

    // Reduction to find block-wide max
    // Store in shared memory
    shared[tid] = local_max;
    __syncthreads();

    // Standard Tree Reduction
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] = max(shared[tid], shared[tid + stride]);
        }
        __syncthreads();
    }
    float row_max = shared[0];

    // 2. Compute Exponentials and Sum
    float local_sum = 0.0f;
    for (int i = tid; i < cols; i += block_size) {
        float val = expf(row_input[i] - row_max);
        local_sum += val;
        // Optimization: Store exp result temporarily in output to avoid re-computing
        row_output[i] = val; 
    }

    // Reduction to find block-wide sum
    shared[tid] = local_sum;
    __syncthreads();

    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    float row_sum = shared[0];

    // 3. Final Division
    float inv_sum = 1.0f / row_sum;
    for (int i = tid; i < cols; i += block_size) {
        row_output[i] *= inv_sum; // row_output currently holds exp(x-max)
    }
}

Latency_Profiling CEDR_SOFTMAX_flt_gpu(
    cedr_re_flt_type** Input, cedr_re_flt_type** Output,
    int* BatchSize, int* Heads, int* SeqLen
) {
    Latency_Profiling latency_profiling = {0, 0, 0};
    struct timespec start_timespec {}, end_timespec {};
    
    int B = *BatchSize; 
    int H = *Heads;
    int L = *SeqLen;

    // Total number of rows to process
    int total_rows = B * H * L; // BatchCount*L
    // Length of each row
    int row_len = L; 

    size_t size = (size_t)total_rows * row_len * sizeof(cedr_re_flt_type);

    cedr_re_flt_type *d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);

    clock_gettime(CLOCK_MONOTONIC_RAW, &start_timespec);
    cudaMemcpy(d_in, (*Input), size, cudaMemcpyHostToDevice);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end_timespec);
    latency_profiling.host_to_device_time = (end_timespec.tv_sec - start_timespec.tv_sec) * SEC2NANOSEC + (end_timespec.tv_nsec - start_timespec.tv_nsec);

    // Launch Config: One CUDA block per row, threads per block = row_len (capped at 1024, usually power of 2 for reduction)
    int threads = 256; 
    while (threads < row_len && threads < 1024) threads *= 2;
    
    // Shared mem size: threads * sizeof(float)
    size_t shared_mem_size = threads * sizeof(float);

    clock_gettime(CLOCK_MONOTONIC_RAW, &start_timespec);
    softmax_kernel<<<total_rows, threads, shared_mem_size>>>(d_in, d_out, row_len);
    // cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC_RAW, &end_timespec);
    latency_profiling.kernel_launch_time = (end_timespec.tv_sec - start_timespec.tv_sec) * SEC2NANOSEC + (end_timespec.tv_nsec - start_timespec.tv_nsec);

    clock_gettime(CLOCK_MONOTONIC_RAW, &start_timespec);
    cudaMemcpy((*Output), d_out, size, cudaMemcpyDeviceToHost);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end_timespec);
    latency_profiling.device_to_host_time = (end_timespec.tv_sec - start_timespec.tv_sec) * SEC2NANOSEC + (end_timespec.tv_nsec - start_timespec.tv_nsec);

    cudaFree(d_in); cudaFree(d_out);
    return latency_profiling;
}