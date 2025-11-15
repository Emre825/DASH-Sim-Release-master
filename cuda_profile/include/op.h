#include "cedr_types.h"

Latency_Profiling CEDR_RELU_flt_gpu(cedr_re_flt_type** input, int* size, cedr_re_flt_type** output);
Latency_Profiling CEDR_GEMM_flt_gpu(cedr_re_flt_type** A, cedr_re_flt_type** B, cedr_re_flt_type** C, size_t* A_ROWS, size_t* A_COLS, size_t* B_COLS, cedr_re_flt_type* maxA_real, cedr_re_flt_type* maxA_imag, cedr_re_flt_type* maxB_real, cedr_re_flt_type* maxB_imag, bool* isComplex);
Latency_Profiling CEDR_MAXPOOL_2D_flt_gpu(cedr_re_flt_type** input, int* size, int* height, int* width, int* pool_size, int* stride, cedr_re_flt_type** output);
