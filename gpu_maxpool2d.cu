#include <stdio.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <stdint.h>

#include "cedr_types.h"
#include "platform.h"

__global__ void max_pooling_kernel(cedr_re_flt_type *input, cedr_re_flt_type *output, int input_w, int output_h, int output_w, int pool_size, int pool_stride) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  if(i < output_h && j < output_w) {
    double max_val = 0;
    for (int k = 0; k < pool_size; k++) {
      for (int l = 0; l < pool_size; l++) {
        max_val = max(max_val, input[(i * pool_stride + k) * input_w + j * pool_stride + l]);
      }
    }
    output[i * output_w + j] = max_val;
  }
}

extern "C" void CEDR_MAXPOOL_2D_flt_gpu(cedr_re_flt_type** input, int* size, int* height, int* width, int* pool_size, int* pool_stride, cedr_re_flt_type** output) {
  int output_h = (*height - *pool_size) / *pool_stride + 1;
  int output_w = (*width - *pool_size) / *pool_stride + 1;
  int block_size = 16;
  dim3 grid_dim((output_w + block_size - 1)/block_size,(output_h + block_size - 1)/block_size);
  dim3 block_dim(block_size,block_size);
  cudaError_t err = cudaSuccess;

  cedr_re_flt_type* d_input = NULL;
  cedr_re_flt_type* d_output = NULL;
  err = cudaMalloc((void**)&d_input, (*height)*(*width)*sizeof(cedr_re_flt_type));
  err = cudaMalloc((void**)&d_output, (*height)*(*width)*sizeof(cedr_re_flt_type));

  err = cudaMemcpy(d_input, (*input), (*height)*(*width)*sizeof(cedr_re_flt_type), cudaMemcpyHostToDevice);

  err = cudaGetLastError();

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate memory or copy data kernel for MAXPOOL_2D on gpu (error code %s)\n", cudaGetErrorString(err));
  }
  max_pooling_kernel<<<grid_dim, block_dim>>>(d_input, d_output, *width, output_h, output_w, *pool_size, *pool_stride);

  err = cudaGetLastError();

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to launch kernel for MAXPOOL_2D on gpu (error code %s)\n", cudaGetErrorString(err));
  }
  //cudaDeviceSynchronize();
  err = cudaMemcpy((*output), d_output, output_h*output_w*sizeof(cedr_re_flt_type), cudaMemcpyDeviceToHost);

  cudaFree(d_output); cudaFree(d_input);

  return;
}

