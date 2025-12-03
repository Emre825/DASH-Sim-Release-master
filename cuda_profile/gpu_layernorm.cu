#include <stdio.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include "include/cedr_types.h"

#define SEC2NANOSEC 1000000000

// Input:  [rows, cols] (row-major, contiguous), Output: [rows, cols]
// gamma, beta: [cols]
// Normalize each row independently over the last dimension (cols).
__global__ void layernorm_kernel(const cedr_re_flt_type* __restrict__ input, cedr_re_flt_type* __restrict__ output,
    const cedr_re_flt_type* __restrict__ gamma,
    const cedr_re_flt_type* __restrict__ beta,
    int rows, int cols, float eps)
{
    int row_idx = blockIdx.x;
    if (row_idx >= rows) return;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    const cedr_re_flt_type* row_in  = input  + (size_t)row_idx * cols;
    cedr_re_flt_type* row_out = output + (size_t)row_idx * cols;

    // Shared memory layout: first blockDim.x floats for sum, next blockDim.x for sq_sum
    extern __shared__ float shm[];
    float* sh_sum = shm;
    float* sh_sq_sum = shm + block_size;

    // Compute sum(x) and sum(x^2) across the row
    float local_sum = 0.0f;
    float local_sq_sum = 0.0f;

    for (int i = tid; i < cols; i += block_size) {
        float v = (float)row_in[i];
        local_sum    += v;
        local_sq_sum += v * v;
    }

    sh_sum[tid] = local_sum;
    sh_sq_sum[tid] = local_sq_sum;
    __syncthreads();

    // Parallel reduction over threads in the block
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sh_sum[tid] += sh_sum[tid + stride];
            sh_sq_sum[tid] += sh_sq_sum[tid + stride];
        }
        __syncthreads();
    }

    float mean = sh_sum[0] / (float)cols;
    float mean_sq = sh_sq_sum[0] / (float)cols;
    float var = mean_sq - mean * mean;
    if (var < 0.0f) var = 0.0f;  // numerical guard
    float inv_std = rsqrtf(var + eps);

    // Normalize and apply affine transform
    for (int i = tid; i < cols; i += block_size) {
        float x = (float)row_in[i];
        float g = (float)gamma[i];
        float b = (float)beta[i];

        float norm = (x - mean) * inv_std;
        float y = g * norm + b;

        row_out[i] = (cedr_re_flt_type)y;
    }
}

Latency_Profiling CEDR_LAYERNORM_flt_gpu(cedr_re_flt_type** input, cedr_re_flt_type** output,
    cedr_re_flt_type** gamma, cedr_re_flt_type** beta,
    int* rows, int* cols, float* eps)
{
    Latency_Profiling latency_profiling = {0, 0, 0};
    struct timespec start_timespec {}, end_timespec {};
    uint64_t start_time, end_time;
    cudaError_t err = cudaSuccess;

    int r = *rows;
    int c = *cols;
    float eps_val = *eps;

    long long total_elems = (long long)r * (long long)c;
    size_t data_bytes = (size_t)total_elems * sizeof(cedr_re_flt_type);
    size_t param_bytes = (size_t)c * sizeof(cedr_re_flt_type);

    // Device pointers
    cedr_re_flt_type* d_input = NULL;
    cedr_re_flt_type* d_output = NULL;
    cedr_re_flt_type* d_gamma = NULL;
    cedr_re_flt_type* d_beta = NULL;

    // Allocate device memory
    err = cudaMalloc((void**)&d_input, data_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for LN input (error %s)\n", cudaGetErrorString(err));
        return latency_profiling;
    }

    err = cudaMalloc((void**)&d_output, data_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for LN output (error %s)\n", cudaGetErrorString(err));
        cudaFree(d_input);
        return latency_profiling;
    }

    err = cudaMalloc((void**)&d_gamma, param_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for LN gamma (error %s)\n", cudaGetErrorString(err));
        cudaFree(d_input); cudaFree(d_output);
        return latency_profiling;
    }

    err = cudaMalloc((void**)&d_beta, param_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for LN beta (error %s)\n", cudaGetErrorString(err));
        cudaFree(d_input); cudaFree(d_output); cudaFree(d_gamma);
        return latency_profiling;
    }

    clock_gettime(CLOCK_MONOTONIC_RAW, &start_timespec);
    err = cudaMemcpy(d_input, (*input), data_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy LN input to device (error %s)\n", cudaGetErrorString(err));
    }
    err = cudaMemcpy(d_gamma, (*gamma), param_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy LN gamma to device (error %s)\n", cudaGetErrorString(err));
    }
    err = cudaMemcpy(d_beta, (*beta), param_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy LN beta to device (error %s)\n", cudaGetErrorString(err));
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, &end_timespec);
    start_time = start_timespec.tv_nsec + (uint64_t)start_timespec.tv_sec * SEC2NANOSEC;
    end_time = end_timespec.tv_nsec + (uint64_t)end_timespec.tv_sec * SEC2NANOSEC;
    latency_profiling.host_to_device_time = (end_time - start_time);

    const int threadsPerBlock = 256;
    const int blocksPerGrid = r;  // one block per row
    size_t shared_bytes = 2 * threadsPerBlock * sizeof(float); // sum + sq_sum

    clock_gettime(CLOCK_MONOTONIC_RAW, &start_timespec);
    layernorm_kernel<<<blocksPerGrid, threadsPerBlock, shared_bytes>>>(d_input, d_output, d_gamma, d_beta, r, c, eps_val);
    // cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC_RAW, &end_timespec);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch LayerNorm kernel (error %s)\n", cudaGetErrorString(err));
    }
    start_time = start_timespec.tv_nsec + (uint64_t)start_timespec.tv_sec * SEC2NANOSEC;
    end_time = end_timespec.tv_nsec + (uint64_t)end_timespec.tv_sec * SEC2NANOSEC;
    latency_profiling.kernel_launch_time = (end_time - start_time);

    clock_gettime(CLOCK_MONOTONIC_RAW, &start_timespec);
    err = cudaMemcpy((*output), d_output, data_bytes, cudaMemcpyDeviceToHost);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end_timespec);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy LN output from device (error %s)\n", cudaGetErrorString(err));
    }
    start_time = start_timespec.tv_nsec + (uint64_t)start_timespec.tv_sec * SEC2NANOSEC;
    end_time = end_timespec.tv_nsec + (uint64_t)end_timespec.tv_sec * SEC2NANOSEC;
    latency_profiling.device_to_host_time = (end_time - start_time);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_gamma);
    cudaFree(d_beta);

    return latency_profiling;
}

