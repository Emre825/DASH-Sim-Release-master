#include <stdio.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <stdint.h>
#include <math.h>

#include "include/cedr_types.h"

#define SEC2NANOSEC 1000000000

__global__ void relu_kernel(cedr_re_flt_type *input, cedr_re_flt_type *output, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        output[i] = fmaxf(0.0, input[i]);
    }
}

Latency_Profiling CEDR_RELU_flt_gpu(cedr_re_flt_type** input, int* size, cedr_re_flt_type** output) {
  const int threadsPerBlock = 512;
  const int blocksPerGrid = (*size + threadsPerBlock - 1) / threadsPerBlock;
  cudaError_t err = cudaSuccess;

  Latency_Profiling latency_profiling = {0, 0, 0};
  struct timespec start_timespec {}, end_timespec {};
  uint64_t start_time, end_time;

  cedr_re_flt_type* d_input = NULL;
  cedr_re_flt_type* d_output = NULL;
  err = cudaMalloc((void**)&d_input, (*size)*sizeof(cedr_re_flt_type));
  err = cudaMalloc((void**)&d_output, (*size)*sizeof(cedr_re_flt_type));

  clock_gettime(CLOCK_MONOTONIC_RAW, &start_timespec);
  err = cudaMemcpy(d_input, (*input), (*size)*sizeof(cedr_re_flt_type), cudaMemcpyHostToDevice);
  clock_gettime(CLOCK_MONOTONIC_RAW, &end_timespec);
  start_time = start_timespec.tv_nsec + start_timespec.tv_sec * SEC2NANOSEC;
  end_time = end_timespec.tv_nsec + end_timespec.tv_sec * SEC2NANOSEC;
  latency_profiling.host_to_device_time = (end_time - start_time);

  err = cudaGetLastError();

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate memory or copy data kernel for RELU on gpu (error code %s)\n", cudaGetErrorString(err));
  }

  clock_gettime(CLOCK_MONOTONIC_RAW, &start_timespec);
  relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, *size);
  clock_gettime(CLOCK_MONOTONIC_RAW, &end_timespec);
  start_time = start_timespec.tv_nsec + start_timespec.tv_sec * SEC2NANOSEC;
  end_time = end_timespec.tv_nsec + end_timespec.tv_sec * SEC2NANOSEC;
  latency_profiling.kernel_launch_time = (end_time - start_time);

  err = cudaGetLastError();

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to launch kernel for RELU on gpu (error code %s)\n", cudaGetErrorString(err));
  }
  //cudaDeviceSynchronize();
  clock_gettime(CLOCK_MONOTONIC_RAW, &start_timespec);
  err = cudaMemcpy((*output), d_output, (*size)*sizeof(cedr_re_flt_type), cudaMemcpyDeviceToHost);
  clock_gettime(CLOCK_MONOTONIC_RAW, &end_timespec);
  start_time = start_timespec.tv_nsec + start_timespec.tv_sec * SEC2NANOSEC;
  end_time = end_timespec.tv_nsec + end_timespec.tv_sec * SEC2NANOSEC;
  latency_profiling.device_to_host_time = (end_time - start_time);

  cudaFree(d_output); cudaFree(d_input);
  return latency_profiling;
}
