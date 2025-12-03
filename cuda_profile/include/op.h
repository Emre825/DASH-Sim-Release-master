#include "cedr_types.h"

Latency_Profiling CEDR_RELU_flt_gpu(cedr_re_flt_type** input, int* size, cedr_re_flt_type** output);
Latency_Profiling CEDR_GEMM_flt_gpu(cedr_re_flt_type** A, cedr_re_flt_type** B, cedr_re_flt_type** C, size_t* A_ROWS, size_t* A_COLS, size_t* B_COLS, cedr_re_flt_type* maxA_real, cedr_re_flt_type* maxA_imag, cedr_re_flt_type* maxB_real, cedr_re_flt_type* maxB_imag, bool* isComplex);
Latency_Profiling CEDR_BATCHED_GEMM_flt_gpu(cedr_re_flt_type** A, cedr_re_flt_type** B, cedr_re_flt_type** C, int* BatchCount, int* M, int* N, int* K, float* alpha_scale, int* transA_flag, int* transB_flag);
Latency_Profiling CEDR_MAXPOOL_2D_flt_gpu(cedr_re_flt_type** input, int* size, int* height, int* width, int* pool_size, int* stride, cedr_re_flt_type** output);
Latency_Profiling CEDR_ADAPTIVE_AVGPOOL_2D_flt_gpu(cedr_re_flt_type** input, int* N, int* C, int* Hin, int* Win, int* Hout, int* Wout, cedr_re_flt_type** output);
Latency_Profiling CEDR_BATCHNORM2D_flt_gpu(cedr_re_flt_type** input, int* N, int* C, int* H, int* W, cedr_re_flt_type** mean, cedr_re_flt_type** var, cedr_re_flt_type** gamma, cedr_re_flt_type** beta, float* eps, cedr_re_flt_type** output);
Latency_Profiling CEDR_SOFTMAX_flt_gpu(cedr_re_flt_type** Input, cedr_re_flt_type** Output, int* BatchCount, int* Heads, int* SeqLen);
Latency_Profiling CEDR_GELU_flt_gpu(cedr_re_flt_type** Input, cedr_re_flt_type** Output, long long* TotalElements);
Latency_Profiling CEDR_LAYERNORM_flt_gpu(cedr_re_flt_type** input, cedr_re_flt_type** output, cedr_re_flt_type** gamma, cedr_re_flt_type** beta, int* rows, int* cols, float* eps);
Latency_Profiling CEDR_EMBEDDING_flt_gpu(int32_t** indices, cedr_re_flt_type** embedding, cedr_re_flt_type** output, int* B, int* L, int* vocab_size, int* d_model);