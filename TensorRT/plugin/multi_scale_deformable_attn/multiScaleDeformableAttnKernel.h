//
// Created by Derry Lin on 2022/11/7.
//

#ifndef TENSORRT_OPS_MULTISCALEDEFORMABLEATTNKERNEL_H
#define TENSORRT_OPS_MULTISCALEDEFORMABLEATTNKERNEL_H

#include <cuda_runtime.h>

template <typename T>
void ms_deformable_im2col_cuda(const T *data_value,
                               const int32_t *data_spatial_shapes,
                               const T *data_sampling_loc,
                               const T *data_attn_weight, const int batch_size,
                               const int spatial_size, const int num_heads,
                               const int channels, const int num_levels,
                               const int num_query, const int num_point,
                               T *data_col, cudaStream_t stream);

#endif // TENSORRT_OPS_MULTISCALEDEFORMABLEATTNKERNEL_H
