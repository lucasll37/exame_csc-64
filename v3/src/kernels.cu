#include <cuda_runtime.h>
#include <cstdio>
#include <float.h>
#include "kernels.h"

__device__ float atomicMaxFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);

    return __int_as_float(old);
}

__device__ float atomicMinFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed, __float_as_int(fminf(val, __int_as_float(assumed))));
    } while (assumed != old);

    return __int_as_float(old);
}

__global__ void filter_records(Record *recordsA, Record *recordsB, int num_records, float threshold_ca_min, float threshold_cb_max) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_records) {
        if (recordsA[idx].value <= threshold_ca_min) {
            recordsA[idx].valid = 0;
        }
        if (recordsB[idx].value >= threshold_cb_max) {
            recordsB[idx].valid = 0;
        }
    }
}

__global__ void find_max_min(Record *records, int num_records, float *max_value, int *max_index, float *min_value, int *min_index) {
    __shared__ float shared_max[256];
    __shared__ int shared_max_idx[256];
    __shared__ float shared_min[256];
    __shared__ int shared_min_idx[256];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (idx < num_records) {
        shared_max[tid] = records[idx].valid ? records[idx].value : -FLT_MAX;
        shared_max_idx[tid] = idx;
        shared_min[tid] = records[idx].valid ? records[idx].value : FLT_MAX;
        shared_min_idx[tid] = idx;
    } else {
        shared_max[tid] = -FLT_MAX;
        shared_max_idx[tid] = -1;
        shared_min[tid] = FLT_MAX;
        shared_min_idx[tid] = -1;
    }

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared_max[tid] < shared_max[tid + s]) {
                shared_max[tid] = shared_max[tid + s];
                shared_max_idx[tid] = shared_max_idx[tid + s];
            }
            if (shared_min[tid] > shared_min[tid + s]) {
                shared_min[tid] = shared_min[tid + s];
                shared_min_idx[tid] = shared_min_idx[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMaxFloat(max_value, shared_max[0]);
        atomicExch(max_index, shared_max_idx[0]);
        atomicMinFloat(min_value, shared_min[0]);
        atomicExch(min_index, shared_min_idx[0]);
    }
}