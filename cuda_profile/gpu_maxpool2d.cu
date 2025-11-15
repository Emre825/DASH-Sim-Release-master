#include <stdio.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include "include/cedr_types.h"

#define SEC2NANOSEC 1000000000

__global__ void max_pooling_kernel(cedr_re_flt_type *input, cedr_re_flt_type *output, 
                                   int input_h, int input_w, 
                                   int output_h, int output_w, 
                                   int pool_size, int pool_stride) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(i < output_h && j < output_w) {
        // Initialize to negative infinity to handle all negative values correctly
        cedr_re_flt_type max_val = -INFINITY;
        
        for (int k = 0; k < pool_size; k++) {
            for (int l = 0; l < pool_size; l++) {
                int input_row = i * pool_stride + k;
                int input_col = j * pool_stride + l;
                
                // Bounds checking to avoid out-of-bounds access
                if (input_row < input_h && input_col < input_w) {
                    cedr_re_flt_type val = input[input_row * input_w + input_col];
                    max_val = fmaxf(max_val, val);
                }
            }
        }
        output[i * output_w + j] = max_val;
    }
}

Latency_Profiling CEDR_MAXPOOL_2D_flt_gpu(cedr_re_flt_type** input, 
                                                       int* size, 
                                                       int* height, 
                                                       int* width, 
                                                       int* pool_size, 
                                                       int* pool_stride, 
                                                       cedr_re_flt_type** output) {
    // Initialize profiling structure
    Latency_Profiling latency_profiling = {0, 0, 0};
    struct timespec start_timespec {}, end_timespec {};
    uint64_t start_time, end_time;
    
    // Calculate output dimensions
    int output_h = (*height - *pool_size) / *pool_stride + 1;
    int output_w = (*width - *pool_size) / *pool_stride + 1;
    
    // Setup grid and block dimensions
    int block_size = 16;
    dim3 grid_dim((output_w + block_size - 1) / block_size, 
                  (output_h + block_size - 1) / block_size);
    dim3 block_dim(block_size, block_size);
    
    cudaError_t err = cudaSuccess;
    
    // Allocate device memory
    cedr_re_flt_type* d_input = NULL;
    cedr_re_flt_type* d_output = NULL;
    
    err = cudaMalloc((void**)&d_input, (*height) * (*width) * sizeof(cedr_re_flt_type));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for input (error code %s)\n", 
                cudaGetErrorString(err));
        return latency_profiling;
    }
    
    // Allocate correct size for output (not input size)
    err = cudaMalloc((void**)&d_output, output_h * output_w * sizeof(cedr_re_flt_type));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for output (error code %s)\n", 
                cudaGetErrorString(err));
        cudaFree(d_input);
        return latency_profiling;
    }
    
    // Time Host-to-Device memory transfer
    clock_gettime(CLOCK_MONOTONIC_RAW, &start_timespec);
    err = cudaMemcpy(d_input, (*input), (*height) * (*width) * sizeof(cedr_re_flt_type), 
                     cudaMemcpyHostToDevice);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end_timespec);
    
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy data to device (error code %s)\n", 
                cudaGetErrorString(err));
        cudaFree(d_input);
        cudaFree(d_output);
        return latency_profiling;
    }
    
    start_time = start_timespec.tv_nsec + start_timespec.tv_sec * SEC2NANOSEC;
    end_time = end_timespec.tv_nsec + end_timespec.tv_sec * SEC2NANOSEC;
    latency_profiling.host_to_device_time = (end_time - start_time);
    
    // Time kernel execution
    clock_gettime(CLOCK_MONOTONIC_RAW, &start_timespec);
    max_pooling_kernel<<<grid_dim, block_dim>>>(d_input, d_output, 
                                                 *height, *width,
                                                 output_h, output_w, 
                                                 *pool_size, *pool_stride);
    
    // Synchronize to ensure kernel completion before timing
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC_RAW, &end_timespec);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch kernel for MAXPOOL_2D on gpu (error code %s)\n", 
                cudaGetErrorString(err));
        cudaFree(d_input);
        cudaFree(d_output);
        return latency_profiling;
    }
    
    start_time = start_timespec.tv_nsec + start_timespec.tv_sec * SEC2NANOSEC;
    end_time = end_timespec.tv_nsec + end_timespec.tv_sec * SEC2NANOSEC;
    latency_profiling.kernel_launch_time = (end_time - start_time);
    
    // Time Device-to-Host memory transfer
    clock_gettime(CLOCK_MONOTONIC_RAW, &start_timespec);
    err = cudaMemcpy((*output), d_output, output_h * output_w * sizeof(cedr_re_flt_type), 
                     cudaMemcpyDeviceToHost);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end_timespec);
    
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy data from device (error code %s)\n", 
                cudaGetErrorString(err));
    }
    
    start_time = start_timespec.tv_nsec + start_timespec.tv_sec * SEC2NANOSEC;
    end_time = end_timespec.tv_nsec + end_timespec.tv_sec * SEC2NANOSEC;
    latency_profiling.device_to_host_time = (end_time - start_time);
    
    // Free device memory
    cudaFree(d_output);
    cudaFree(d_input);
    
    return latency_profiling;
}