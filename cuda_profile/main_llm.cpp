#include <stdio.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <stdint.h>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <iostream>
#include <math.h>
#include <tuple>
#include "include/cedr_types.h"
#include "include/op.h"

struct AttentionParams {
    int B;      // Batch Size
    int L;      // Sequence Length
    int H;      // Num Heads
    int D_head; // Head Dimension
};

struct GeluParams {
    int B;       // Batch Size
    int L;       // Sequence Length
    int D_ff;    // Hidden Dimension of the FFN
};

struct LayerNormParams {
    int B;       // Batch Size
    int L;       // Sequence Length
    int D_model; // Hidden dimension
};

struct EmbeddingParams {
    int B;          // Batch size
    int L;          // Sequence length
    int vocab_size; // Vocabulary size (or max_len for positional embedding)
    int D_model;    // Embedding dimension
};

void print_stats(const std::vector<uint64_t>& v, const char* name) {
    if (v.empty()) {
        printf("%s: no samples\n", name);
        return;
    }

    double sum = 0.0;
    for (auto x : v) sum += static_cast<double>(x);
    double mean = sum / v.size();

    std::vector<uint64_t> tmp = v;
    std::sort(tmp.begin(), tmp.end());

    double median;
    size_t n = tmp.size();
    if (n % 2)
        median = static_cast<double>(tmp[n / 2]);
    else
        median = (static_cast<double>(tmp[n / 2 - 1]) + static_cast<double>(tmp[n / 2])) / 2.0;

    uint64_t minv = tmp.front();
    uint64_t maxv = tmp.back();

    printf("%s (ns): mean=%.3f, median=%.3f, min=%llu, max=%llu\n",
           name, mean, median,
           static_cast<unsigned long long>(minv),
           static_cast<unsigned long long>(maxv));
}

// GEMM pass for attention in LLM
int batched_gemm_pass(int BatchCount, int M, int N, int K, float alpha, int transA, int transB, int trial, const char* name) {
    long long strideA = (long long)M * K;
    long long strideB = (long long)K * N;
    long long strideC = (long long)M * N;
    long long total_A = strideA * BatchCount;
    long long total_B = strideB * BatchCount;
    long long total_C = strideC * BatchCount;

    cedr_re_flt_type *inputA = (cedr_re_flt_type*)malloc(total_A * sizeof(cedr_re_flt_type));
    cedr_re_flt_type *inputB = (cedr_re_flt_type*)malloc(total_B * sizeof(cedr_re_flt_type));
    cedr_re_flt_type *output = (cedr_re_flt_type*)malloc(total_C * sizeof(cedr_re_flt_type));

    // Initialize random data
    for(long long i=0; i<total_A; i++) inputA[i] = (cedr_re_flt_type)(rand()%10)/10.0f;
    for(long long i=0; i<total_B; i++) inputB[i] = (cedr_re_flt_type)(rand()%10)/10.0f;

    std::vector<uint64_t> host_to_device_times(trial);
    std::vector<uint64_t> device_to_host_times(trial);
    std::vector<uint64_t> kernel_launch_times(trial);

    for (int t = 0; t < trial; ++t) {
        Latency_Profiling lp = {};
        lp = CEDR_BATCHED_GEMM_flt_gpu(&inputA, &inputB, &output, 
                                      &BatchCount, &M, &N, &K, &alpha, &transA, &transB);
        host_to_device_times[t] = lp.host_to_device_time;
        device_to_host_times[t] = lp.device_to_host_time;
        kernel_launch_times[t] = lp.kernel_launch_time;
    }

    printf("******************************************\n");
    printf("%s GPU Profiling Results over %d trials:\n", name, trial);
    printf("Dims: [%d, %d, %d] x [%d, %d, %d]\n", BatchCount, M, K, BatchCount, K, N);
    print_stats(host_to_device_times, "host_to_device");
    print_stats(device_to_host_times, "device_to_host");
    print_stats(kernel_launch_times, "kernel_launch");
    printf("******************************************\n");

    free(inputA); free(inputB); free(output);
    return 1;
}

int softmax_pass(int B, int H, int L, int trial, const char* name) {
    long long total_elements = (long long)B * H * L * L;
    
    cedr_re_flt_type *input = (cedr_re_flt_type*)malloc(total_elements * sizeof(cedr_re_flt_type));
    cedr_re_flt_type *output = (cedr_re_flt_type*)malloc(total_elements * sizeof(cedr_re_flt_type));

    // Random init
    for(long long i=0; i<total_elements; i++) input[i] = ((float)rand()/RAND_MAX);

    std::vector<uint64_t> k_times(trial);
    std::vector<uint64_t> h2d_times(trial);
    std::vector<uint64_t> d2h_times(trial);

    // We pass B, H separately to calculate total rows inside wrapper
    for (int t = 0; t < trial; ++t) {
        Latency_Profiling lp = CEDR_SOFTMAX_flt_gpu(&input, &output, &B, &H, &L);
        k_times[t] = lp.kernel_launch_time;
        h2d_times[t] = lp.host_to_device_time;
        d2h_times[t] = lp.device_to_host_time;
    }

    printf("******************************************\n");
    printf("%s GPU Profiling Results over %d trials:\n", name, trial);
    printf("Input Shape: (%d, %d, %d, %d)\n", B, H, L, L);
    print_stats(h2d_times, "host_to_device");
    print_stats(d2h_times, "device_to_host");
    print_stats(k_times, "kernel_launch");
    printf("******************************************\n");

    free(input); free(output);
    return 1;
}

int gelu_pass(long long total_elements, int trial, const char* name) {
    cedr_re_flt_type *input = (cedr_re_flt_type*)malloc(total_elements * sizeof(cedr_re_flt_type));
    cedr_re_flt_type *output = (cedr_re_flt_type*)malloc(total_elements * sizeof(cedr_re_flt_type));

    // Random init
    for(long long i=0; i<total_elements; i++) input[i] = ((float)rand()/RAND_MAX);

    std::vector<uint64_t> k_times(trial);
    std::vector<uint64_t> h2d_times(trial);
    std::vector<uint64_t> d2h_times(trial);

    for (int t = 0; t < trial; ++t) {
        Latency_Profiling lp = CEDR_GELU_flt_gpu(&input, &output, &total_elements);
        k_times[t] = lp.kernel_launch_time;
        h2d_times[t] = lp.host_to_device_time;
        d2h_times[t] = lp.device_to_host_time;
    }

    printf("******************************************\n");
    printf("%s GPU Profiling Results over %d trials:\n", name, trial);
    printf("Total Elements: %lld\n", total_elements);
    print_stats(h2d_times, "host_to_device");
    print_stats(d2h_times, "device_to_host");
    print_stats(k_times, "kernel_launch");
    printf("******************************************\n");

    free(input); free(output);
    return 1;
}

int layernorm_pass(int B, int L, int D_model, int trial, const char* name) {
    // Input shape is (B, L, D_model), we flatten to (rows, cols) = (B*L, D_model)
    int rows = B * L;
    int cols = D_model;
    long long total_elems = (long long)rows * (long long)cols;

    cedr_re_flt_type* input  = (cedr_re_flt_type*)malloc(total_elems * sizeof(cedr_re_flt_type));
    cedr_re_flt_type* output = (cedr_re_flt_type*)malloc(total_elems * sizeof(cedr_re_flt_type));
    cedr_re_flt_type* gamma  = (cedr_re_flt_type*)malloc(cols * sizeof(cedr_re_flt_type));
    cedr_re_flt_type* beta   = (cedr_re_flt_type*)malloc(cols * sizeof(cedr_re_flt_type));

    // Random input
    for (long long i = 0; i < total_elems; ++i) {
        input[i] = (cedr_re_flt_type)((float)rand() / RAND_MAX);
    }

    // Typical LayerNorm initialization: gamma = 1, beta = 0
    for (int j = 0; j < cols; ++j) {
        gamma[j] = (cedr_re_flt_type)1.0f;
        beta[j]  = (cedr_re_flt_type)0.0f;
    }

    std::vector<uint64_t> k_times(trial);
    std::vector<uint64_t> h2d_times(trial);
    std::vector<uint64_t> d2h_times(trial);

    float eps = 1e-5f;

    for (int t = 0; t < trial; ++t) {
        int rows_copy = rows;
        int cols_copy = cols;
        Latency_Profiling lp = CEDR_LAYERNORM_flt_gpu(
            &input, &output, &gamma, &beta,
            &rows_copy, &cols_copy, &eps
        );
        k_times[t]  = lp.kernel_launch_time;
        h2d_times[t] = lp.host_to_device_time;
        d2h_times[t] = lp.device_to_host_time;
    }

    printf("******************************************\n");
    printf("%s GPU Profiling Results over %d trials:\n", name, trial);
    printf("Input Shape: (%d, %d, %d)  -> rows=%d, cols=%d\n", B, L, D_model, rows, cols);
    print_stats(h2d_times, "host_to_device");
    print_stats(d2h_times, "device_to_host");
    print_stats(k_times,   "kernel_launch");
    printf("******************************************\n");

    free(input); free(output); free(gamma); free(beta);

    return 1;
}

int embedding_pass(const EmbeddingParams& ep, int trial, const char* name) {
    int B = ep.B; int L = ep.L;
    int vocab_size = ep.vocab_size;
    int D_model = ep.D_model;

    long long num_indices = (long long)B * (long long)L;
    long long num_embeddings = (long long)vocab_size * (long long)D_model;
    long long num_output = num_indices * (long long)D_model;

    // Host buffers
    int32_t* indices = (int32_t*)malloc(num_indices * sizeof(int32_t));
    cedr_re_flt_type* embedding = (cedr_re_flt_type*)malloc(num_embeddings * sizeof(cedr_re_flt_type));
    cedr_re_flt_type* output = (cedr_re_flt_type*)malloc(num_output * sizeof(cedr_re_flt_type));

    if (!indices || !embedding || !output) {
        fprintf(stderr, "Failed to allocate host memory for embedding pass\n");
        free(indices); free(embedding); free(output);
        return 0;
    }

    // Random indices in [0, vocab_size)
    for (long long i = 0; i < num_indices; ++i) {
        indices[i] = (int32_t)(rand() % vocab_size);
    }

    // Random embedding table
    for (long long i = 0; i < num_embeddings; ++i) {
        embedding[i] = (cedr_re_flt_type)((float)rand() / RAND_MAX);
    }

    std::vector<uint64_t> k_times(trial);
    std::vector<uint64_t> h2d_times(trial);
    std::vector<uint64_t> d2h_times(trial);

    for (int t = 0; t < trial; ++t) {
        int B_copy = B; int L_copy = L;
        int vocab_copy = vocab_size;
        int D_model_copy = D_model;
        Latency_Profiling lp = CEDR_EMBEDDING_flt_gpu(
            &indices, &embedding, &output,
            &B_copy, &L_copy, &vocab_copy, &D_model_copy
        );
        h2d_times[t] = lp.host_to_device_time;
        d2h_times[t] = lp.device_to_host_time;
        k_times[t]   = lp.kernel_launch_time;
    }

    printf("******************************************\n");
    printf("%s GPU Profiling Results over %d trials:\n", name, trial);
    printf("Input indices shape: (%d, %d)\n", B, L);
    printf("Embedding table shape: (%d, %d)\n", vocab_size, D_model);
    printf("Output shape: (%d, %d, %d)\n", B, L, D_model);
    print_stats(h2d_times, "host_to_device");
    print_stats(d2h_times, "device_to_host");
    print_stats(k_times,   "kernel_launch");
    printf("******************************************\n");

    free(indices); free(embedding); free(output);
    return 1;
}

int main(){
  AttentionParams attn;
  attn.B = 16;
  attn.L = 512;
  attn.H = 16;
  attn.D_head = 64; // (D_model / num_heads)

  GeluParams gelu_p;
  gelu_p.B = 16;
  gelu_p.L = 512;
  gelu_p.D_ff = 4096;

  LayerNormParams lnp;
  lnp.B = 16;
  lnp.L = 512;
  lnp.D_model = 1024;

  EmbeddingParams tok_emb;
  tok_emb.B = 16;
  tok_emb.L = 512;
  tok_emb.vocab_size = 32000; // vocab_size used for token embedding
  tok_emb.D_model = 1024;

  EmbeddingParams pos_emb;
  pos_emb.B = 1; // after torch.arange(L).unsqueeze(0)
  pos_emb.L = 512;
  pos_emb.vocab_size = 2048; // max sequence length used for positional embedding
  pos_emb.D_model = 1024;
  
  // Q * K_transpose * (1/sqrt(D_head)) (Calculating Scores), shape: (L, D_head) * (D_head, L) -> (L, L), we set transA=0 (for Q) and transB=1 (for K^T).
  printf("----------------Profiling for Batched GEMM----------------\n");
  int BatchCount = attn.B * attn.H; // B*H independent matrices
  int M1 = attn.L;
  int N1 = attn.L;
  int K1 = attn.D_head;
  float scale = 1.0f / sqrt((float)attn.D_head); 
  batched_gemm_pass(BatchCount, M1, N1, K1, scale, 0, 1, 100, "ATTN: QxK^T");

  // Score * V (Calculating Context), shape: (L, L) * (L, D_head) -> (L, D_head), no transpose needed.
  printf("----------------Profiling for Batched GEMM----------------\n");
  int BatchCount = attn.B * attn.H; // B*H independent matrices
  int M2 = attn.L;
  int N2 = attn.D_head;
  int K2 = attn.L;
  batched_gemm_pass(BatchCount, M2, N2, K2, 1.0f, 0, 0, 100, "ATTN: ScorexV");
  
  // SOFTMAX, input to softmax is the output of previous step: (B, H, L, L), we normalize over the last dimension (L).
  printf("----------------Profiling for SOFTMAX----------------\n");
  softmax_pass(attn.B, attn.H, attn.L, 100, "ATTN: Softmax");

  printf("----------------Profiling for GELU----------------\n");
  long long gelu_elements = (long long)gelu_p.B * gelu_p.L * gelu_p.D_ff;
  gelu_pass(gelu_elements, 100, "FFN: GELU");

  printf("----------------Profiling for LayerNorm----------------\n");
  layernorm_pass(lnp.B, lnp.L, lnp.D_model, 100, "LayerNorm");

  printf("----------------Profiling for Token Embedding----------------\n");
  embedding_pass(tok_emb, 100, "Token Embedding");

  printf("----------------Profiling for Positional Embedding----------------\n");
  embedding_pass(pos_emb, 100, "Positional Embedding");

  return 0;
}