#include <stdio.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include "include/cedr_types.h"

#define SEC2NANOSEC 1000000000

// Kernel: AdaptiveAvgPool2d for NCHW
// input:  shape (N, C, Hin, Win)
// output: shape (N, C, Hout, Wout)
__global__ void adaptive_avg_pool2d_kernel(
    const cedr_re_flt_type* __restrict__ input,
    cedr_re_flt_type* __restrict__ output,
    int N, int C,
    int Hin, int Win,
    int Hout, int Wout)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * Hout * Wout;
    if (idx >= total) return;

    // Decode flattened index: (n, c, oh, ow)
    int ow = idx % Wout;
    int tmp = idx / Wout;
    int oh = tmp % Hout;
    tmp /= Hout;
    int c = tmp % C;
    int n = tmp / C;

    // Input base offset for (n, c, :, :)
    int base_in = ((n * C + c) * Hin) * Win;

    // Compute the input window for this output (oh, ow)
    // Using standard adaptive pooling formula:
    // h_start = floor(oh * Hin / Hout)
    // h_end   = ceil((oh + 1) * Hin / Hout)
    // same for width
    float stride_h = (float)Hin / (float)Hout;
    float stride_w = (float)Win / (float)Wout;

    int h_start = (int)floorf(oh * stride_h);
    int h_end   = (int)ceilf((oh + 1) * stride_h);
    int w_start = (int)floorf(ow * stride_w);
    int w_end   = (int)ceilf((ow + 1) * stride_w);

    // Clamp to bounds
    if (h_start < 0)      h_start = 0;
    if (h_end > Hin)      h_end = Hin;
    if (w_start < 0)      w_start = 0;
    if (w_end > Win)      w_end = Win;

    cedr_re_flt_type sum = 0.0f;
    int count = 0;

    for (int h = h_start; h < h_end; ++h) {
        for (int w = w_start; w < w_end; ++w) {
            int in_idx = base_in + h * Win + w;
            sum += input[in_idx];
            ++count;
        }
    }

    cedr_re_flt_type avg = (count > 0) ? (sum / (cedr_re_flt_type)count) : 0.0f;
    output[idx] = avg;
}

// Wrapper
// input:  pointer to host buffer of size N*C*Hin*Win
// output: pointer to host buffer of size N*C*Hout*Wout
Latency_Profiling CEDR_ADAPTIVE_AVGPOOL_2D_flt_gpu(
    cedr_re_flt_type** input,
    int* N, int* C,
    int* Hin, int* Win,
    int* Hout, int* Wout,
    cedr_re_flt_type** output)
{
    Latency_Profiling latency_profiling = {0, 0, 0};
    struct timespec start_timespec {}, end_timespec {};
    uint64_t start_time, end_time;

    int n = *N;
    int c = *C;
    int hin = *Hin;
    int win = *Win;
    int hout = *Hout;
    int wout = *Wout;

    // Sizes
    size_t input_elems  = (size_t)n * c * hin * win;
    size_t output_elems = (size_t)n * c * hout * wout;
    size_t input_bytes  = input_elems  * sizeof(cedr_re_flt_type);
    size_t output_bytes = output_elems * sizeof(cedr_re_flt_type);

    cedr_re_flt_type* d_input  = NULL;
    cedr_re_flt_type* d_output = NULL;

    cudaError_t err = cudaSuccess;

    // Device alloc
    err = cudaMalloc((void**)&d_input,  input_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for ADAPTIVE_AVGPOOL2D input (error %s)\n",
                cudaGetErrorString(err));
        return latency_profiling;
    }

    err = cudaMalloc((void**)&d_output, output_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for ADAPTIVE_AVGPOOL2D output (error %s)\n",
                cudaGetErrorString(err));
        cudaFree(d_input);
        return latency_profiling;
    }

    // Host → Device
    clock_gettime(CLOCK_MONOTONIC_RAW, &start_timespec);
    err = cudaMemcpy(d_input, (*input), input_bytes, cudaMemcpyHostToDevice);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end_timespec);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy input to device for ADAPTIVE_AVGPOOL2D (error %s)\n",
                cudaGetErrorString(err));
        cudaFree(d_input);
        cudaFree(d_output);
        return latency_profiling;
    }
    start_time = start_timespec.tv_nsec + (uint64_t)start_timespec.tv_sec * SEC2NANOSEC;
    end_time   = end_timespec.tv_nsec   + (uint64_t)end_timespec.tv_sec   * SEC2NANOSEC;
    latency_profiling.host_to_device_time = (end_time - start_time);

    // Kernel launch
    int threadsPerBlock = 256;
    int total_outputs   = n * c * hout * wout;
    int blocksPerGrid   = (total_outputs + threadsPerBlock - 1) / threadsPerBlock;

    clock_gettime(CLOCK_MONOTONIC_RAW, &start_timespec);
    adaptive_avg_pool2d_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_input, d_output,
        n, c, hin, win, hout, wout
    );
    //cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC_RAW, &end_timespec);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch ADAPTIVE_AVGPOOL2D kernel (error %s)\n",
                cudaGetErrorString(err));
        cudaFree(d_input);
        cudaFree(d_output);
        return latency_profiling;
    }

    start_time = start_timespec.tv_nsec + (uint64_t)start_timespec.tv_sec * SEC2NANOSEC;
    end_time   = end_timespec.tv_nsec   + (uint64_t)end_timespec.tv_sec   * SEC2NANOSEC;
    latency_profiling.kernel_launch_time = (end_time - start_time);

    // Device → Host
    clock_gettime(CLOCK_MONOTONIC_RAW, &start_timespec);
    err = cudaMemcpy((*output), d_output, output_bytes, cudaMemcpyDeviceToHost);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end_timespec);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy output from device for ADAPTIVE_AVGPOOL2D (error %s)\n",
                cudaGetErrorString(err));
        cudaFree(d_input);
        cudaFree(d_output);
        return latency_profiling;
    }

    start_time = start_timespec.tv_nsec + (uint64_t)start_timespec.tv_sec * SEC2NANOSEC;
    end_time   = end_timespec.tv_nsec   + (uint64_t)end_timespec.tv_sec   * SEC2NANOSEC;
    latency_profiling.device_to_host_time = (end_time - start_time);

    cudaFree(d_input);
    cudaFree(d_output);

    return latency_profiling;
}