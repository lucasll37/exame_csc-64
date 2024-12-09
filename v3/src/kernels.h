#ifndef KERNELS_H
#define KERNELS_H

typedef struct {
    char id[6];
    float value;
    int valid;
} Record;

#ifdef __cplusplus
extern "C" {
#endif

// Declare as CUDA kernels
__global__ void filter_records(Record *recordsA, Record *recordsB, int num_records, float threshold_ca_min, float threshold_cb_max);
__global__ void find_max_min(Record *records, int num_records, float *max_value, int *max_index, float *min_value, int *min_index);

#ifdef __cplusplus
}
#endif

#endif // KERNELS_H
