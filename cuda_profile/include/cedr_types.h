#pragma once
#include <stdint.h>
#ifdef __cplusplus
#include <atomic>
#else
#include <stdatomic.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif
// Not the actual flt min/max just some large enough values
#define CONV_2D_MIN_FLT -2147483648
#define CONV_2D_MAX_FLT 2147483647
// Based on what CONV 2D accelerator supports
#define CONV_2D_MIN 0
#define CONV_2D_MAX 32767
enum resource_type { cpu = 0, fft = 1, mmult = 2, zip = 3, gpu = 4, conv_2d = 5, fec = 6, dap = 7, NUM_RESOURCE_TYPES = 8 };

typedef short cedr_re_int_type; // change 'short' to 'int32_t' if you are using CONV2D uint8 APIs (for HW accelerators)
typedef struct cedr_cmplx_int_type {
cedr_re_int_type im;
cedr_re_int_type re;
} cedr_cmplx_int_type;

typedef float cedr_re_flt_type;
typedef struct cedr_cmplx_flt_type {
cedr_re_flt_type re;
cedr_re_flt_type im;
} cedr_cmplx_flt_type;

typedef enum zip_op {
  ZIP_ADD = 0,
  ZIP_SUB = 1,
  ZIP_MULT = 2,
  ZIP_DIV = 3,
} zip_op_t;

// TODO: Can we pack the atomic counter into the PE struct above?
#ifdef __cplusplus
typedef std::atomic<uint64_t> thread_count;
typedef struct struct_schedule{
	void *func_pointer;
	resource_type PE_type;
  int Cluster_id;
	thread_count* PE_seeker_list;
} schedule;
#else
typedef atomic_int thread_count; // This will probably break some things...
typedef struct struct_schedule{
	void *func_pointer;
	enum resource_type PE_type;
  int Cluster_id;
	thread_count* PE_seeker_list;
} schedule;
#endif

typedef struct cedr_task_constraints {
#ifdef __cplusplus
  bool supported_resources[resource_type::NUM_RESOURCE_TYPES] = {};
#else
  bool supported_resources[NUM_RESOURCE_TYPES];
#endif
} cedr_task_constraints_t;

typedef struct cedr_task_sched {
  uint32_t task_count;
  schedule* task_schedule;
} cedr_task_sched_t;

typedef struct cedr_task_config {
  cedr_task_sched_t* sched;
  cedr_task_constraints_t* constr;
} cedr_task_config_t;

typedef struct fec_result {
  uint8_t* decoded; // Decoded output - packed binary
  uint32_t status; // Yes or No Decoding Success - Future Use (Always just hits MaxIter)
} fec_result_t;


#ifdef __cplusplus
} // Close 'extern "C"'
#endif


struct Latency_Profiling {
    uint64_t host_to_device_time;
    uint64_t device_to_host_time;
    uint64_t kernel_launch_time;
};