// Minimal stub so our_solver.cu compiles as plain C++ on the host (self-test).
// When compiled with nvcc, the real CUDA headers are used instead (no -I cuda_stub).
#ifndef CUDA_RUNTIME_STUB_H
#define CUDA_RUNTIME_STUB_H

#include <cstddef>
#include <cstring>

typedef int cudaError_t;
#define cudaSuccess 0
#define cudaMemcpyHostToDevice 1
#define cudaMemcpyDeviceToHost 2

struct dim3 {
    unsigned x, y, z;
    dim3(unsigned x_ = 0, unsigned y_ = 1, unsigned z_ = 1) : x(x_), y(y_), z(z_) {}
};
struct cudaDeviceProp { char name[256]; };
typedef void* cudaEvent_t;

// built-in thread indexing (values irrelevant — brute_kernel is never called on host)
static dim3 blockIdx, blockDim, threadIdx;

static inline cudaError_t cudaSetDevice(int) { return cudaSuccess; }
static inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp* p, int) {
    std::strcpy(p->name, "CPU-STUB");
    return cudaSuccess;
}
static inline cudaError_t cudaMalloc(void** p, size_t n) { *p = new char[n]; return cudaSuccess; }
static inline cudaError_t cudaMemcpy(void*, const void*, size_t, int) { return cudaSuccess; }
static inline cudaError_t cudaMemset(void*, int, size_t) { return cudaSuccess; }
static inline cudaError_t cudaGetLastError() { return cudaSuccess; }
static inline cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }
static inline cudaError_t cudaEventCreate(cudaEvent_t*) { return cudaSuccess; }
static inline cudaError_t cudaEventRecord(cudaEvent_t) { return cudaSuccess; }
static inline cudaError_t cudaEventSynchronize(cudaEvent_t) { return cudaSuccess; }
static inline cudaError_t cudaEventElapsedTime(float*, cudaEvent_t, cudaEvent_t) { return cudaSuccess; }
static inline const char* cudaGetErrorString(cudaError_t) { return "stub"; }

#endif
