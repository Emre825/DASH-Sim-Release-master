
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

struct LinearParams {
    int N;        // Batch size
    int Fin;      // Input features
    int Fout;     // Output features
};

// ------------------------------------------------------------
// Returns (A_rows, A_cols, B_cols)
// A_rows = Fout
// A_cols = Fin
// B_cols = N
// ------------------------------------------------------------
std::tuple<long long, long long, long long, long long, long long> compute_linear_gemm(const LinearParams& p) {
    long long A_rows = p.Fout;
    long long A_cols = p.Fin;
    long long B_cols = p.N;

    return {A_rows, A_cols, B_cols, A_rows, B_cols};
}


struct ConvParams {
    int N;      // Batch size
    int Cin;    // Input channels
    int Cout;   // Output channels
    int Hin;    // Input height
    int Win;    // Input width
    int Kh;     // Kernel height
    int Kw;     // Kernel width
    int Ph;     // Padding height
    int Pw;     // Padding width
    int Sh;     // Stride height
    int Sw;     // Stride width
    int Dh;     // Dilation height
    int Dw;     // Dilation width
};

struct MaxPoolParams {
    int N;          // Batch size
    int C;          // Number of channels
    int Hin;        // Input height
    int Win;        // Input width
    int pool_size;  // Pooling kernel size
    int stride;     // Pooling stride
};

std::tuple<long long, long long, long long, long long, long long> compute_conv_gemm(const ConvParams& p) {
    // Compute output dimensions
    int Hout = std::floor((p.Hin + 2 * p.Ph - p.Dh * (p.Kh - 1) - 1) / p.Sh + 1);
    int Wout = std::floor((p.Win + 2 * p.Pw - p.Dw * (p.Kw - 1) - 1) / p.Sw + 1);

    // Effective kernel size
    int Keff = p.Cin * p.Kh * p.Kw;

    // Matrix multiply dimensions
    long long A_rows = p.Cout;
    long long A_cols = Keff;
    long long B_rows = Keff;
    long long B_cols = (long long)p.N * Hout * Wout;
    long long C_rows = p.Cout;
    long long C_cols = (long long)p.N * Hout * Wout;

    return {A_rows, A_cols, B_cols, C_rows, C_cols};
}

std::tuple<int, int, int> compute_maxpool_output(const MaxPoolParams& p) {
    int Hout = (p.Hin - p.pool_size) / p.stride + 1;
    int Wout = (p.Win - p.pool_size) / p.stride + 1;
    int total_size = p.N * p.C * Hout * Wout;
    return {Hout, Wout, total_size};
}

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

int relu_pass(int size, int trial){
  cedr_re_flt_type *input, *output;

  // Initialize input and size
  input = (cedr_re_flt_type*)malloc(size * sizeof(cedr_re_flt_type));
  output = (cedr_re_flt_type*)malloc(size * sizeof(cedr_re_flt_type));
  for(int i = 0; i < size; i++) {
      input[i] = (cedr_re_flt_type)(i - 512); // Some values negative, some positive
  }

  std::vector<uint64_t> host_to_device_times(trial);
  std::vector<uint64_t> device_to_host_times(trial);
  std::vector<uint64_t> kernel_launch_times(trial);

  for (int t = 0; t < trial; ++t) {
    Latency_Profiling lp = {};
    lp = CEDR_RELU_flt_gpu(&input, &size, &output);
    host_to_device_times[t] = lp.host_to_device_time;
    device_to_host_times[t] = lp.device_to_host_time;
    kernel_launch_times[t] = lp.kernel_launch_time;
  }

  printf("******************************************\n");
  printf("RELU GPU Profiling Results over %d trials:\n", trial);
  print_stats(host_to_device_times, "host_to_device");
  print_stats(device_to_host_times, "device_to_host");
  print_stats(kernel_launch_times, "kernel_launch");
  printf("******************************************\n");

  // Free allocated memory
  free(input);
  free(output);

  return 1;
}

int gemm_pass(size_t A_ROWS, size_t A_COLS, size_t B_COLS, int trial){
  cedr_re_flt_type *inputA, *inputB, *output;

  // Initialize input and size
  inputA = (cedr_re_flt_type*)malloc(A_ROWS * A_COLS * sizeof(cedr_re_flt_type));
  inputB = (cedr_re_flt_type*)malloc(A_COLS * B_COLS * sizeof(cedr_re_flt_type));
  output = (cedr_re_flt_type*)malloc(A_ROWS * B_COLS * sizeof(cedr_re_flt_type));
  for(int i = 0; i < A_ROWS * A_COLS; i++) {
      inputA[i] = (cedr_re_flt_type)(i - 512); // Some values negative, some positive
  }
  for(int i = 0; i < A_COLS * B_COLS; i++) {
      inputB[i] = (cedr_re_flt_type)(i - 512); // Some values negative, some positive
  }

  std::vector<uint64_t> host_to_device_times(trial);
  std::vector<uint64_t> device_to_host_times(trial);
  std::vector<uint64_t> kernel_launch_times(trial);

  bool isComplex = false;

  for (int t = 0; t < trial; ++t) {
    Latency_Profiling lp = {};
    lp = CEDR_GEMM_flt_gpu(&inputA, &inputB, &output, &A_ROWS, &A_COLS, &B_COLS, 0, 0, 0, 0, &isComplex);
    host_to_device_times[t] = lp.host_to_device_time;
    device_to_host_times[t] = lp.device_to_host_time;
    kernel_launch_times[t] = lp.kernel_launch_time;
  }

  printf("******************************************\n");
  printf("GEMM GPU Profiling Results over %d trials:\n", trial);
  print_stats(host_to_device_times, "host_to_device");
  print_stats(device_to_host_times, "device_to_host");
  print_stats(kernel_launch_times, "kernel_launch");
  printf("******************************************\n");

  // Free allocated memory
  free(inputA);
  free(inputB);
  free(output);

  return 1;
}

int maxpool_pass(int height, int width, int pool_size, int stride, int trial) {
  cedr_re_flt_type *input, *output;
  
  // Calculate output dimensions
  int output_h = (height - pool_size) / stride + 1;
  int output_w = (width - pool_size) / stride + 1;
  int input_size = height * width;
  int output_size = output_h * output_w;
  
  // Allocate and initialize input
  input = (cedr_re_flt_type*)malloc(input_size * sizeof(cedr_re_flt_type));
  output = (cedr_re_flt_type*)malloc(output_size * sizeof(cedr_re_flt_type));
  
  for(int i = 0; i < input_size; i++) {
      input[i] = (cedr_re_flt_type)(rand() % 1000) / 10.0f; // Random values 0-100
  }
  
  std::vector<uint64_t> host_to_device_times(trial);
  std::vector<uint64_t> device_to_host_times(trial);
  std::vector<uint64_t> kernel_launch_times(trial);
  
  int size = input_size; // Dummy parameter if needed by API
  
  for (int t = 0; t < trial; ++t) {
    Latency_Profiling lp = {};
    lp = CEDR_MAXPOOL_2D_flt_gpu(&input, &size, &height, &width, 
                                  &pool_size, &stride, &output);
    host_to_device_times[t] = lp.host_to_device_time;
    device_to_host_times[t] = lp.device_to_host_time;
    kernel_launch_times[t] = lp.kernel_launch_time;
  }
  
  printf("******************************************\n");
  printf("MAXPOOL2D GPU Profiling Results over %d trials:\n", trial);
  printf("Input: %dx%d, Output: %dx%d, Pool: %dx%d, Stride: %d\n", 
         height, width, output_h, output_w, pool_size, pool_size, stride);
  print_stats(host_to_device_times, "host_to_device");
  print_stats(device_to_host_times, "device_to_host");
  print_stats(kernel_launch_times, "kernel_launch");
  printf("******************************************\n");
  
  // Free allocated memory
  free(input);
  free(output);
  
  return 1;
}

//     int N;      // Batch size
//     int Cin;    // Input channels
//     int Cout;   // Output channels
//     int Hin;    // Input height
//     int Win;    // Input width
//     int Kh;     // Kernel height
//     int Kw;     // Kernel width
//     int Ph;     // Padding height
//     int Pw;     // Padding width
//     int Sh;     // Stride height
//     int Sw;     // Stride width
//     int Dh;     // Dilation height
//     int Dw;     // Dilation width

int main(){

  ConvParams p;
  p.N = 1;
  p.Cin = 8;  
  p.Cout = 8; 
  p.Hin = 512; 
  p.Win = 512;  
  p.Kh = 3;
  p.Kw = 3;
  p.Ph = 1;
  p.Pw = 1;
  p.Sh = 1;
  p.Sw = 1;
  p.Dh = 1;
  p.Dw = 1;

  MaxPoolParams maxp;
  maxp.N = 1;
  maxp.C = 64;
  maxp.Hin = 64;
  maxp.Win = 64;
  maxp.pool_size = 2;
  maxp.stride = 2;

  auto [A_rows, A_cols, B_cols, C_rows, C_cols] = compute_conv_gemm(p);

  std::cout << "A_rows = " << A_rows << "\n";
  std::cout << "A_cols = " << A_cols << "\n";
  std::cout << "B_cols = " << B_cols << "\n";
  std::cout << "C_rows = " << C_rows << "\n";
  std::cout << "C_cols = " << C_cols << "\n";

  LinearParams lp;
  lp.N = 1;
  lp.Fin = 1024;
  lp.Fout = 2048;

  auto [A_Frows, A_Fcols, B_Fcols, C_Frows, C_Fcols] = compute_linear_gemm(lp);

  std::cout << "A_Frows = " << A_Frows << "\n";
  std::cout << "A_Fcols = " << A_Fcols << "\n";
  std::cout << "B_Fcols = " << B_Fcols << "\n";

  printf("----------------Profiling for CONV2D----------------\n");
  gemm_pass(A_rows, A_cols, B_cols, 100);
  relu_pass(C_rows*C_cols, 100);

  printf("----------------Profiling for Maxpool2D----------------\n");
  maxpool_pass(maxp.Hin, maxp.Win, maxp.pool_size, maxp.stride, 100);

  printf("----------------Profiling for LINEAR----------------\n");
  gemm_pass(A_Frows, A_Fcols, B_Fcols, 100);
  relu_pass(C_Frows*C_Fcols, 100);

  return 0;
}