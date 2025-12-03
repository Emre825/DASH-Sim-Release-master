#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#include <time.h>
#include "include/cedr_types.h"

#define SEC2NANOSEC 1000000000

// Formula: 0.5 * x * (1 + erf(x / sqrt(2)))
__global__ void gelu_kernel(float* in, float* out, long long n) {
    long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in[i];
        out[i] = 0.5f * x * (1.0f + erff(x * M_SQRT1_2));
    }
}

Latency_Profiling CEDR_GELU_flt_gpu(
    cedr_re_flt_type** Input, cedr_re_flt_type** Output,
    long long* TotalElements
) {
    Latency_Profiling latency_profiling = {0, 0, 0};
    struct timespec start_timespec {}, end_timespec {};
    uint64_t start_time, end_time;

    long long n = *TotalElements;
    size_t size = n * sizeof(cedr_re_flt_type);

    cedr_re_flt_type *d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);

    clock_gettime(CLOCK_MONOTONIC_RAW, &start_timespec);
    cudaMemcpy(d_in, (*Input), size, cudaMemcpyHostToDevice);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end_timespec);
    start_time = start_timespec.tv_nsec + start_timespec.tv_sec * SEC2NANOSEC;
    end_time = end_timespec.tv_nsec + end_timespec.tv_sec * SEC2NANOSEC;
    latency_profiling.host_to_device_time = (end_time - start_time);

    // Grid Stride logic or simple 1D mapping
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    clock_gettime(CLOCK_MONOTONIC_RAW, &start_timespec);
    gelu_kernel<<<blocks, threads>>>(d_in, d_out, n);
    //cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC_RAW, &end_timespec);
    start_time = start_timespec.tv_nsec + start_timespec.tv_sec * SEC2NANOSEC;
    end_time = end_timespec.tv_nsec + end_timespec.tv_sec * SEC2NANOSEC;
    latency_profiling.kernel_launch_time = (end_time - start_time);

    clock_gettime(CLOCK_MONOTONIC_RAW, &start_timespec);
    cudaMemcpy((*Output), d_out, size, cudaMemcpyDeviceToHost);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end_timespec);
    start_time = start_timespec.tv_nsec + start_timespec.tv_sec * SEC2NANOSEC;
    end_time = end_timespec.tv_nsec + end_timespec.tv_sec * SEC2NANOSEC;
    latency_profiling.device_to_host_time = (end_time - start_time);

    cudaFree(d_in); cudaFree(d_out);
    return latency_profiling;
}