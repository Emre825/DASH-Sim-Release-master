#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <unistd.h>
#include <stdint.h>
#include <time.h>

#include "include/cedr_types.h"

#define SEC2NANOSEC 1000000000


//cudaStream_t stream;
cublasHandle_t handle;
extern "C" void __attribute__((constructor)) cuda_gemm_setup(void) {
  //cudaStreamCreate(&stream);
  cublasCreate(&handle);
  //cublasSetStream(handle, stream);
}

extern "C" void __attribute__((destructor)) cuda_gemm_teardown(void) {
  cublasDestroy(handle);
  //cudaStreamDestroy(stream);
}

// Compute C = A * B
__global__ void matrixMultiply(const cedr_cmplx_flt_type *A, const cedr_cmplx_flt_type *B, cedr_cmplx_flt_type *C,
				int numARows, int numAColumns,
				int numBRows, int numBColumns,
			       	int numCRows, int numCColumns) {
  //@@ Insert code to implement basic matrix multiplication for
  //@@ arbitrary size using global memory. 
  int ROW = blockIdx.y*blockDim.y+threadIdx.y; // Calculate Row based on threads position in the block
  int COL = blockIdx.x*blockDim.x+threadIdx.x; // Calculate Col based on threads position in the block
  if( (ROW<numCRows) && (COL<numCColumns) ){ // As long as the thread is in the boundries of C compute the result for sum(col*row)
    cedr_cmplx_flt_type partial_p; // Initialize output to be written to the C
    partial_p.re = 0;
    partial_p.im = 0;
    for( int i = 0; i<numAColumns ; i++ ){
      partial_p.re += A[ROW*numAColumns+i].re * B[i*numBColumns+COL].re - A[ROW*numAColumns+i].im * B[i*numBColumns+COL].im; // Compute the A[row][i]*B[i][col] real
      partial_p.im += A[ROW*numAColumns+i].re * B[i*numBColumns+COL].im + A[ROW*numAColumns+i].im * B[i*numBColumns+COL].re; // Compute the A[row][i]*B[i][col] imag
    }
    C[ROW*numCColumns+COL].re = partial_p.re; // Write back the result real
    C[ROW*numCColumns+COL].im = partial_p.im; // Write back the result imag
  }
}

Latency_Profiling CEDR_GEMM_flt_gpu(cedr_re_flt_type** A, cedr_re_flt_type** B, cedr_re_flt_type** C, size_t* A_ROWS, size_t* A_COLS, size_t* B_COLS, cedr_re_flt_type* maxA_real, cedr_re_flt_type* maxA_imag, cedr_re_flt_type* maxB_real, cedr_re_flt_type* maxB_imag, bool* isComplex) {
//    int dev_count;
//    cudaGetDeviceCount(&dev_count);
//    cudaSetDevice(resource_idx%dev_count);
//    printf("---------------------------------------\n");
//    printf("------------- GEMM on GPU --------------\n");
//    printf("---------------------------------------\n");
  Latency_Profiling latency_profiling = {0, 0, 0};
  struct timespec start_timespec {}, end_timespec {};
  uint64_t start_time, end_time;
  if(*isComplex){
    const int row_a = *A_ROWS;
    const int col_a = *A_COLS;
    const int col_b = *B_COLS;
    int block_size = 16;
    dim3 grid_dim(((col_b-1)/block_size)+1,((row_a-1)/block_size)+1,1);
    dim3 block_dim(block_size,block_size,1);
    cudaError_t err = cudaSuccess;


    cedr_cmplx_flt_type* d_A = NULL;
    cedr_cmplx_flt_type* d_B = NULL;
    cedr_cmplx_flt_type* d_C = NULL;
    err = cudaMalloc((void**)&d_A, row_a*col_a*sizeof(cedr_cmplx_flt_type));
    err = cudaMalloc((void**)&d_B, col_a*col_b*sizeof(cedr_cmplx_flt_type));
    err = cudaMalloc((void**)&d_C, row_a*col_b*sizeof(cedr_cmplx_flt_type));

    clock_gettime(CLOCK_MONOTONIC_RAW, &start_timespec);
    err = cudaMemcpy(d_A, (*A), row_a*col_a*sizeof(cedr_cmplx_flt_type), cudaMemcpyHostToDevice);
    err = cudaMemcpy(d_B, (*B), col_a*col_b*sizeof(cedr_cmplx_flt_type), cudaMemcpyHostToDevice);
    

    err = cudaGetLastError();

    if (err != cudaSuccess) {
      fprintf(stderr, "Failed to launch kernel for GEMM on gpu (error code %s)\n", cudaGetErrorString(err));
    }
    
//    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    matrixMultiply<<<grid_dim, block_dim>>>(d_A, d_B, d_C,
					row_a, col_a,
					col_a, col_b,
					row_a, col_b);

    err = cudaGetLastError();

    if (err != cudaSuccess) {
      fprintf(stderr, "Failed to launch kernel for GEMM on gpu (error code %s)\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
    err = cudaMemcpy((*C), d_C, row_a*col_b*sizeof(cedr_cmplx_flt_type), cudaMemcpyDeviceToHost);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    // return;
  } else{
    int M = *A_ROWS;
    int N = *B_COLS;
    int K = *A_COLS;

    size_t size_A = M * K * sizeof(cedr_re_flt_type);
    size_t size_B = N * K * sizeof(cedr_re_flt_type);
    size_t size_C = M * N * sizeof(cedr_re_flt_type);

    
    cedr_re_flt_type *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    clock_gettime(CLOCK_MONOTONIC_RAW, &start_timespec);
    cudaMemcpy(d_A, (*A), size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, (*B), size_B, cudaMemcpyHostToDevice);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end_timespec);
    start_time = start_timespec.tv_nsec + start_timespec.tv_sec * SEC2NANOSEC;
    end_time = end_timespec.tv_nsec + end_timespec.tv_sec * SEC2NANOSEC;
    latency_profiling.host_to_device_time = (end_time - start_time);

    float alpha = 1.0f;
    float beta = 0.0f;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start_timespec);
    cublasSgemm(
            handle,
            CUBLAS_OP_T,      // op(A) = B (no transpose)
            CUBLAS_OP_N,      // op(B) = Aᵀ
            M,                // m = colB
            N,                // n = rowA
            K,                // k = colA
            &alpha,
            d_A, K,           // A matrix (B), leading dimension = K
            d_B, K,           // B matrix (A), leading dimension = K
            &beta,
            d_C, M            // C matrix, leading dimension = N
        );
    //cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC_RAW, &end_timespec);
    start_time = start_timespec.tv_nsec + start_timespec.tv_sec * SEC2NANOSEC;
    end_time = end_timespec.tv_nsec + end_timespec.tv_sec * SEC2NANOSEC;
    latency_profiling.kernel_launch_time = (end_time - start_time);

    //cudaStreamSynchronize(stream);
    clock_gettime(CLOCK_MONOTONIC_RAW, &start_timespec);
    cudaMemcpy((*C), d_C, size_C, cudaMemcpyDeviceToHost);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end_timespec);
    start_time = start_timespec.tv_nsec + start_timespec.tv_sec * SEC2NANOSEC;
    end_time = end_timespec.tv_nsec + end_timespec.tv_sec * SEC2NANOSEC;
    latency_profiling.device_to_host_time = (end_time - start_time);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
  }
  return latency_profiling;
}

// Batch Matrix Multiplication (Attention)
Latency_Profiling CEDR_BATCHED_GEMM_flt_gpu(
    cedr_re_flt_type** A, cedr_re_flt_type** B, cedr_re_flt_type** C,
    int* BatchCount, // Number of matrices (B * Heads)
    int* M,          // Rows of A and C
    int* N,          // Cols of B and C
    int* K,          // Cols of A and Rows of B
    float* alpha_scale, // Scaling factor (e.g., 1/sqrt(d_head))
    int* transA_flag, // 0 = No Transpose, 1 = Transpose
    int* transB_flag  // 0 = No Transpose, 1 = Transpose
) {
    Latency_Profiling latency_profiling = {0, 0, 0};
    struct timespec start_timespec {}, end_timespec {};
    uint64_t start_time, end_time;

    int m = *M;
    int n = *N;
    int k = *K;
    int batch_count = *BatchCount;

    cublasOperation_t opA = (*transA_flag) ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = (*transB_flag) ? CUBLAS_OP_T : CUBLAS_OP_N;

    // Calculate Strides (Distance between Matrix i and Matrix i+1)
    long long strideA = (long long)m * k;
    long long strideB = (long long)k * n;
    long long strideC = (long long)m * n;

    size_t size_A = strideA * batch_count * sizeof(cedr_re_flt_type);
    size_t size_B = strideB * batch_count * sizeof(cedr_re_flt_type);
    size_t size_C = strideC * batch_count * sizeof(cedr_re_flt_type);

    cedr_re_flt_type *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    clock_gettime(CLOCK_MONOTONIC_RAW, &start_timespec);
    cudaMemcpy(d_A, (*A), size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, (*B), size_B, cudaMemcpyHostToDevice);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end_timespec);
    start_time = start_timespec.tv_nsec + start_timespec.tv_sec * SEC2NANOSEC;
    end_time = end_timespec.tv_nsec + end_timespec.tv_sec * SEC2NANOSEC;
    latency_profiling.host_to_device_time = (end_time - start_time);

    float alpha = *alpha_scale; 
    float beta = 0.0f;
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &start_timespec);

    // To compute C = A x B^T (Row Major), we need to compute C^T = B x A^T (Col Major).
    // cuBLAS is column-major, so it sees d_B (normally, K) as K^T and d_A (normally, Q) as Q^T. To get the result C (which cuBLAS sees it as C^T), we must calculate;
    // C^T = (K^T)^T x Q^T (so, Q^T in col major is actually Q in row major, and (K^T)^T is equal to K in col major but it is actually K^T in row major)
    // We pass d_B as the first matrix (Arg1, K), cuBLAS sees it as K^T and applies opB (which is CUBLAS_OP_T because of transB flag) and it becomes (K^T)^T.
    // We pass d_A as the second matrix (Arg2, Q), cuBLAS sees it as Q^T and applies opA (which is CUBLAS_OP_N because of transA flag) and it becomes Q^T.
    
    // StridedBatched, the reason we can't use normal cublasSgemm is that it is 2D, but matrix mul. operations in attention happens 4D (Batch, Heads, SeqL, HeadDim).
    // We have X batches and N number of heads so we basically applying cublasSgemm X*N times, this can be done with a for loop with cublasSgemm but this is faster.
    // The scaling in the calculation of scores (which is 1/sqrt(HeadDim)) happens in the API as well, because we passed alpha as 1/sqrt(HeadDim) into the API. 
    cublasSgemmStridedBatched(
        handle,
        opB,     // Arg1 op: Controlled by transB (Matrix B)
        opA,     // Arg2 op: Controlled by transA (Matrix A)
        n, m, k, // Swapped M and N, instead of passing MxK and KxN (which is what we want), we pass NxK and KxM because WE WANT TO COMPUTE C^T instead of C.
        &alpha,
        d_B, (*transB_flag) ? k : n, strideB, // Arg1 is B. Leading dim depends on transB.
        d_A, k, strideA, // Arg2 is A. Leading dim is always k for pass1 and pass2.
        &beta,
        d_C, n, strideC, 
        batch_count
    );

    //cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC_RAW, &end_timespec);
    start_time = start_timespec.tv_nsec + start_timespec.tv_sec * SEC2NANOSEC;
    end_time = end_timespec.tv_nsec + end_timespec.tv_sec * SEC2NANOSEC;
    latency_profiling.kernel_launch_time = (end_time - start_time);

    clock_gettime(CLOCK_MONOTONIC_RAW, &start_timespec);
    cudaMemcpy((*C), d_C, size_C, cudaMemcpyDeviceToHost);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end_timespec);
    start_time = start_timespec.tv_nsec + start_timespec.tv_sec * SEC2NANOSEC;
    end_time = end_timespec.tv_nsec + end_timespec.tv_sec * SEC2NANOSEC;
    latency_profiling.device_to_host_time = (end_time - start_time);
    
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return latency_profiling;
}

