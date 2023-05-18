//
// Created by Derry Lin on 2023/5/11.
//

#ifndef TENSORRT_OPS_BEVPOOLKERNEL_H
#define TENSORRT_OPS_BEVPOOLKERNEL_H

#include <cuda_runtime.h>


// CUDA function declarations
void bev_pool_v2(int c, int n_intervals, const float* depth, const float* feat,
                 const int* ranks_depth, const int* ranks_feat, const int* ranks_bev,
                 const int* interval_starts, const int* interval_lengths, float* out, cudaStream_t stream);

void bev_pool_v2_set_zero(int n_points, float* out);

#endif //TENSORRT_OPS_BEVPOOLKERNEL_H
