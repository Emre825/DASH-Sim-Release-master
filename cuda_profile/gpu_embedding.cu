#include <stdio.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include "include/cedr_types.h"

#define SEC2NANOSEC 1000000000

// Embedding forward kernel
// indices: [B * L] (int32), embedding_table: [vocab_size, d_model]
// output: [B, L, d_model]
__global__ void embedding_kernel(const int32_t* __restrict__ indices, const cedr_re_flt_type* __restrict__ embedding_table,
    cedr_re_flt_type* __restrict__ output,
    int B, int L, int d_model, int vocab_size)
{
    int row = blockIdx.x; // row corresponds to (b, l)
    if (row >= B * L) return;

    int tid    = threadIdx.x;
    int stride = blockDim.x;

    // Fetch token id for this (b, l)
    int32_t token_id = indices[row];
    if (token_id < 0 || token_id >= vocab_size) {
        // Out-of-range guard
        return;
    }

    const cedr_re_flt_type* emb_row = embedding_table + (size_t)token_id * d_model;
    cedr_re_flt_type* out_row = output + (size_t)row * d_model;

    // Copy embedding vector
    for (int d = tid; d < d_model; d += stride) {
        out_row[d] = emb_row[d];
    }
}

Latency_Profiling CEDR_EMBEDDING_flt_gpu(int32_t** indices, cedr_re_flt_type** embedding,
    cedr_re_flt_type** output,
    int* B, int* L, int* vocab_size, int* d_model)
{
    Latency_Profiling latency_profiling = {0, 0, 0};
    struct timespec start_timespec {}, end_timespec {};
    uint64_t start_time, end_time;
    cudaError_t err = cudaSuccess;

    int b = *B; int l  = *L;
    int vs = *vocab_size;
    int dm = *d_model;

    long long num_indices = (long long)b * (long long)l;
    long long num_embeddings = (long long)vs * (long long)dm;
    long long num_output = num_indices * (long long)dm;

    size_t indices_bytes = (size_t)num_indices * sizeof(int32_t);
    size_t embedding_bytes = (size_t)num_embeddings * sizeof(cedr_re_flt_type);
    size_t output_bytes = (size_t)num_output * sizeof(cedr_re_flt_type);

    // Device pointers
    int32_t* d_indices = NULL;
    cedr_re_flt_type* d_embedding = NULL;
    cedr_re_flt_type* d_output = NULL;

    // Allocate device memory
    err = cudaMalloc((void**)&d_indices, indices_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for indices (error %s)\n", cudaGetErrorString(err));
        return latency_profiling;
    }

    err = cudaMalloc((void**)&d_embedding, embedding_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for embedding table (error %s)\n", cudaGetErrorString(err));
        cudaFree(d_indices);
        return latency_profiling;
    }

    err = cudaMalloc((void**)&d_output, output_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for embedding output (error %s)\n", cudaGetErrorString(err));
        cudaFree(d_indices); cudaFree(d_embedding);
        return latency_profiling;
    }

    // Host -> Device timing (indices + embedding)
    clock_gettime(CLOCK_MONOTONIC_RAW, &start_timespec);
    err = cudaMemcpy(d_indices, (*indices), indices_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy indices to device (error %s)\n", cudaGetErrorString(err));
    }
    err = cudaMemcpy(d_embedding, (*embedding), embedding_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy embedding table to device (error %s)\n", cudaGetErrorString(err));
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, &end_timespec);
    start_time = start_timespec.tv_nsec + (uint64_t)start_timespec.tv_sec * SEC2NANOSEC;
    end_time = end_timespec.tv_nsec + (uint64_t)end_timespec.tv_sec * SEC2NANOSEC;
    latency_profiling.host_to_device_time = (end_time - start_time);

    // Kernel launch timing
    int rows = b * l; // one block per (b, l) row
    const int threadsPerBlock = 256;
    const int blocksPerGrid = rows;

    clock_gettime(CLOCK_MONOTONIC_RAW, &start_timespec);
    embedding_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_indices, d_embedding, d_output, b, l, dm, vs);
    // cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC_RAW, &end_timespec);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch Embedding kernel (error %s)\n", cudaGetErrorString(err));
    }
    start_time = start_timespec.tv_nsec + (uint64_t)start_timespec.tv_sec * SEC2NANOSEC;
    end_time = end_timespec.tv_nsec + (uint64_t)end_timespec.tv_sec * SEC2NANOSEC;
    latency_profiling.kernel_launch_time = (end_time - start_time);

    // Device -> Host timing (output only)
    clock_gettime(CLOCK_MONOTONIC_RAW, &start_timespec);
    err = cudaMemcpy((*output), d_output, output_bytes, cudaMemcpyDeviceToHost);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end_timespec);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy embedding output from device (error %s)\n", cudaGetErrorString(err));
    }
    start_time = start_timespec.tv_nsec + (uint64_t)start_timespec.tv_sec * SEC2NANOSEC;
    end_time = end_timespec.tv_nsec + (uint64_t)end_timespec.tv_sec * SEC2NANOSEC;
    latency_profiling.device_to_host_time = (end_time - start_time);

    cudaFree(d_indices); cudaFree(d_embedding); cudaFree(d_output);

    return latency_profiling;
}