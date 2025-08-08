#include <stdio.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <stdint.h>
#include <math.h>

#include "cedr_types.h"
#include "platform.h"

__global__ void relu_kernel(cedr_re_flt_type *input, cedr_re_flt_type *output, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        output[i] = fmaxf(0.0, input[i]);
    }
}

extern "C" void CEDR_RELU_flt_gpu(cedr_re_flt_type** input, int* size, cedr_re_flt_type** output) {
  const int threadsPerBlock = 512;
  const int blocksPerGrid = (*size + threadsPerBlock - 1) / threadsPerBlock;
  cudaError_t err = cudaSuccess;

  cedr_re_flt_type* d_input = NULL;
  cedr_re_flt_type* d_output = NULL;
  err = cudaMalloc((void**)&d_input, (*size)*sizeof(cedr_re_flt_type));
  err = cudaMalloc((void**)&d_output, (*size)*sizeof(cedr_re_flt_type));

  err = cudaMemcpy(d_input, (*input), (*size)*sizeof(cedr_re_flt_type), cudaMemcpyHostToDevice);

  err = cudaGetLastError();

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate memory or copy data kernel for RELU on gpu (error code %s)\n", cudaGetErrorString(err));
  }
  relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, *size);

  err = cudaGetLastError();

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to launch kernel for RELU on gpu (error code %s)\n", cudaGetErrorString(err));
  }
  //cudaDeviceSynchronize();
  err = cudaMemcpy((*output), d_output, (*size)*sizeof(cedr_re_flt_type), cudaMemcpyDeviceToHost);

  cudaFree(d_output); cudaFree(d_input);

  return;
}

