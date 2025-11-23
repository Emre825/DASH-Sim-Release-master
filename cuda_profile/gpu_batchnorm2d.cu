#include <stdio.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include "include/cedr_types.h"

#define SEC2NANOSEC 1000000000

// input/output: NCHW layout
// mean, var, gamma, beta: length C
__global__ void batchnorm2d_kernel(
    const cedr_re_flt_type* __restrict__ input,
    cedr_re_flt_type* __restrict__ output,
    const cedr_re_flt_type* __restrict__ mean,
    const cedr_re_flt_type* __restrict__ var,
    const cedr_re_flt_type* __restrict__ gamma,
    const cedr_re_flt_type* __restrict__ beta,
    int N, int C, int H, int W,
    float eps)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;

    if (idx >= total) return;

    int HW = H * W;

    // For NCHW, contiguous:
    // index layout: ((n * C + c) * H + h) * W + w
    // To get channel index c from linear idx:
    //  idx = n*(C*H*W) + c*(H*W) + (h*W + w)
    //  -> c = (idx / (H*W)) % C
    int c = (idx / HW) % C;

    cedr_re_flt_type x = input[idx];

    cedr_re_flt_type m   = mean[c];
    cedr_re_flt_type v   = var[c];
    cedr_re_flt_type g   = gamma[c];
    cedr_re_flt_type b   = beta[c];

    // 1 / sqrt(var + eps)
    float inv_std = rsqrtf((float)v + eps);

    cedr_re_flt_type y = g * (x - m) * (cedr_re_flt_type)inv_std + b;

    output[idx] = y;
}

Latency_Profiling CEDR_BATCHNORM2D_flt_gpu(
    cedr_re_flt_type** input,   // [N*C*H*W]
    int* N,
    int* C,
    int* H,
    int* W,
    cedr_re_flt_type** mean,    // [C]
    cedr_re_flt_type** var,     // [C]
    cedr_re_flt_type** gamma,   // [C]
    cedr_re_flt_type** beta,    // [C]
    float* eps,
    cedr_re_flt_type** output)  // [N*C*H*W]
{
    Latency_Profiling latency_profiling = {0, 0, 0};
    struct timespec start_timespec {}, end_timespec {};
    uint64_t start_time, end_time;

    cudaError_t err = cudaSuccess;

    int n = *N;
    int c = *C;
    int h = *H;
    int w = *W;
    float eps_val = *eps;

    int total_elems = n * c * h * w;
    size_t data_bytes   = (size_t)total_elems * sizeof(cedr_re_flt_type);
    size_t channel_bytes = (size_t)c * sizeof(cedr_re_flt_type);

    // Device pointers
    cedr_re_flt_type* d_input  = NULL;
    cedr_re_flt_type* d_output = NULL;
    cedr_re_flt_type* d_mean   = NULL;
    cedr_re_flt_type* d_var    = NULL;
    cedr_re_flt_type* d_gamma  = NULL;
    cedr_re_flt_type* d_beta   = NULL;

    // Allocate
    err = cudaMalloc((void**)&d_input,  data_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for input (error code %s)\n",
                cudaGetErrorString(err));
        return latency_profiling;
    }

    err = cudaMalloc((void**)&d_output, data_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for output (error code %s)\n",
                cudaGetErrorString(err));
        cudaFree(d_input);
        return latency_profiling;
    }

    err = cudaMalloc((void**)&d_mean, channel_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for mean (error code %s)\n",
                cudaGetErrorString(err));
        cudaFree(d_input); cudaFree(d_output);
        return latency_profiling;
    }

    err = cudaMalloc((void**)&d_var, channel_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for var (error code %s)\n",
                cudaGetErrorString(err));
        cudaFree(d_input); cudaFree(d_output);
        cudaFree(d_mean);
        return latency_profiling;
    }

    err = cudaMalloc((void**)&d_gamma, channel_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for gamma (error code %s)\n",
                cudaGetErrorString(err));
        cudaFree(d_input); cudaFree(d_output);
        cudaFree(d_mean);  cudaFree(d_var);
        return latency_profiling;
    }

    err = cudaMalloc((void**)&d_beta, channel_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for beta (error code %s)\n",
                cudaGetErrorString(err));
        cudaFree(d_input); cudaFree(d_output);
        cudaFree(d_mean);  cudaFree(d_var);
        cudaFree(d_gamma);
        return latency_profiling;
    }

    // Host -> Device timing (input + BN params)
    clock_gettime(CLOCK_MONOTONIC_RAW, &start_timespec);

    err = cudaMemcpy(d_input, (*input), data_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy input to device (error code %s)\n",
                cudaGetErrorString(err));
    }

    err = cudaMemcpy(d_mean, (*mean), channel_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy mean to device (error code %s)\n",
                cudaGetErrorString(err));
    }

    err = cudaMemcpy(d_var, (*var), channel_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy var to device (error code %s)\n",
                cudaGetErrorString(err));
    }

    err = cudaMemcpy(d_gamma, (*gamma), channel_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy gamma to device (error code %s)\n",
                cudaGetErrorString(err));
    }

    err = cudaMemcpy(d_beta, (*beta), channel_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy beta to device (error code %s)\n",
                cudaGetErrorString(err));
    }

    clock_gettime(CLOCK_MONOTONIC_RAW, &end_timespec);
    start_time = start_timespec.tv_nsec + (uint64_t)start_timespec.tv_sec * SEC2NANOSEC;
    end_time   = end_timespec.tv_nsec   + (uint64_t)end_timespec.tv_sec   * SEC2NANOSEC;
    latency_profiling.host_to_device_time = (end_time - start_time);

    // Kernel launch timing (host-side)
    const int threadsPerBlock = 512;
    const int blocksPerGrid = (total_elems + threadsPerBlock - 1) / threadsPerBlock;

    clock_gettime(CLOCK_MONOTONIC_RAW, &start_timespec);
    batchnorm2d_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_input, d_output,
        d_mean, d_var, d_gamma, d_beta,
        n, c, h, w,
        eps_val
    );
    //cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC_RAW, &end_timespec);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch kernel for BATCHNORM2D on gpu (error code %s)\n",
                cudaGetErrorString(err));
    }

    start_time = start_timespec.tv_nsec + (uint64_t)start_timespec.tv_sec * SEC2NANOSEC;
    end_time   = end_timespec.tv_nsec   + (uint64_t)end_timespec.tv_sec   * SEC2NANOSEC;
    latency_profiling.kernel_launch_time = (end_time - start_time);

    // Device -> Host timing (output only)
    clock_gettime(CLOCK_MONOTONIC_RAW, &start_timespec);
    err = cudaMemcpy((*output), d_output, data_bytes, cudaMemcpyDeviceToHost);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end_timespec);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy output from device (error code %s)\n",
                cudaGetErrorString(err));
    }

    start_time = start_timespec.tv_nsec + (uint64_t)start_timespec.tv_sec * SEC2NANOSEC;
    end_time   = end_timespec.tv_nsec   + (uint64_t)end_timespec.tv_sec   * SEC2NANOSEC;
    latency_profiling.device_to_host_time = (end_time - start_time);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_mean);
    cudaFree(d_var);
    cudaFree(d_gamma);
    cudaFree(d_beta);

    return latency_profiling;
}
