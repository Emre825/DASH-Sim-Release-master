#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <unistd.h>
#include <stdint.h>

#include "cedr_types.h"
#include "platform.h"

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

extern "C" void CEDR_GEMM_flt_gpu(cedr_re_flt_type** A, cedr_re_flt_type** B, cedr_re_flt_type** C, size_t* A_ROWS, size_t* A_COLS, size_t* B_COLS, cedr_re_flt_type* maxA_real, cedr_re_flt_type* maxA_imag, cedr_re_flt_type* maxB_real, cedr_re_flt_type* maxB_imag, bool* isComplex) {
//    int dev_count;
//    cudaGetDeviceCount(&dev_count);
//    cudaSetDevice(resource_idx%dev_count);
//    printf("---------------------------------------\n");
//    printf("------------- GEMM on GPU --------------\n");
//    printf("---------------------------------------\n");
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

    return;
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

    cudaMemcpy(d_A, (*A), size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, (*B), size_B, cudaMemcpyHostToDevice);
    float alpha = 1.0f;
    float beta = 0.0f;
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
    //cudaStreamSynchronize(stream);
    cudaMemcpy((*C), d_C, size_C, cudaMemcpyDeviceToHost);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
  }
}

