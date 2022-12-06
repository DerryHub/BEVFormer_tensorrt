//
// Created by Derry Lin on 2022/11/7.
//

#include <cstdio>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "cuda_helper.h"
#include "helper.h"
#include "multiScaleDeformableAttnKernel.h"
#include <cstdio>

template <typename scalar_t>
__device__ scalar_t ms_deform_attn_im2col_bilinear(
    const scalar_t *&bottom_data, const int &height, const int &width,
    const int &nheads, const int &channels, const scalar_t &h,
    const scalar_t &w, const int &m, const int &c) {
  const int h_low = floorf(h);
  const int w_low = floorf(w);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const scalar_t lh = h - h_low;
  const scalar_t lw = w - w_low;
  const scalar_t hh = 1 - lh, hw = 1 - lw;

  const int w_stride = nheads * channels;
  const int h_stride = width * w_stride;
  const int h_low_ptr_offset = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
  const int base_ptr = m * channels + c;

  scalar_t v1 = 0;
  if (h_low >= 0 && w_low >= 0) {
    const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
    v1 = bottom_data[ptr1];
  }
  scalar_t v2 = 0;
  if (h_low >= 0 && w_high <= width - 1) {
    const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
    v2 = bottom_data[ptr2];
  }
  scalar_t v3 = 0;
  if (h_high <= height - 1 && w_low >= 0) {
    const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
    v3 = bottom_data[ptr3];
  }
  scalar_t v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1) {
    const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
    v4 = bottom_data[ptr4];
  }

  const scalar_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  const scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

template <>
__device__ __half ms_deform_attn_im2col_bilinear(
    const __half *&bottom_data, const int &height, const int &width,
    const int &nheads, const int &channels, const __half &h, const __half &w,
    const int &m, const int &c) {
  const __half h_low = hfloor(h);
  const __half w_low = hfloor(w);
  const __half h_high = __hadd(h_low, __float2half(1.f));
  const __half w_high = __hadd(w_low, __float2half(1.f));

  const __half lh = __hsub(h, h_low);
  const __half lw = __hsub(w, w_low);
  const __half hh = __hsub(__float2half(1.f), lh),
               hw = __hsub(__float2half(1.f), lw);

  const int w_stride = nheads * channels;
  const int h_stride = width * w_stride;
  const int h_low_ptr_offset = static_cast<int>(h_low) * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = static_cast<int>(w_low) * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
  const int base_ptr = m * channels + c;

  __half v1 = 0;
  if (__hge(h_low, __float2half(0.f)) && __hge(w_low, __float2half(0.f))) {
    const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
    v1 = bottom_data[ptr1];
  }
  __half v2 = 0;
  if (__hge(h_low, __float2half(0.f)) &&
      __hle(w_high, __float2half(static_cast<float>(width - 1)))) {
    const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
    v2 = bottom_data[ptr2];
  }
  __half v3 = 0;
  if (__hle(h_high, __float2half(static_cast<float>(height - 1))) &&
      __hge(w_low, __float2half(0.f))) {
    const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
    v3 = bottom_data[ptr3];
  }
  __half v4 = 0;
  if (__hle(h_high, __float2half(static_cast<float>(height - 1))) &&
      __hle(w_high, __float2half(static_cast<float>(width - 1)))) {
    const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
    v4 = bottom_data[ptr4];
  }
  const __half w1 = __hmul(hh, hw), w2 = __hmul(hh, lw), w3 = __hmul(lh, hw),
               w4 = __hmul(lh, lw);
  return __hadd(__hadd(__hmul(w1, v1), __hmul(w2, v2)),
                __hadd(__hmul(w3, v3), __hmul(w4, v4)));
}

__device__ __half2 ms_deform_attn_im2col_bilinear_h2(
    const __half *&bottom_data, const int &height, const int &width,
    const int &nheads, const int &channels, const __half &h, const __half &w,
    const int &m, const int &c, const bool &last_one) {
  const __half h_low = hfloor(h);
  const __half w_low = hfloor(w);
  const __half h_high = __hadd(h_low, __float2half(1.f));
  const __half w_high = __hadd(w_low, __float2half(1.f));

  const __half2 lh = __half2half2(__hsub(h, h_low));
  const __half2 lw = __half2half2(__hsub(w, w_low));
  const __half2 hh = __hsub2(__float2half2_rn(1.f), lh),
                hw = __hsub2(__float2half2_rn(1.f), lw);

  const int w_stride = nheads * channels;
  const int h_stride = width * w_stride;
  const int h_low_ptr_offset = static_cast<int>(h_low) * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = static_cast<int>(w_low) * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
  const int base_ptr = m * channels + c * 2;
  const int max_p = channels * nheads * height * width - 2;

  int ptr;

  __half2 v1 = __float2half2_rn(0.f);
  if (__hge(h_low, __float2half(0.f)) && __hge(w_low, __float2half(0.f))) {
    ptr = min(h_low_ptr_offset + w_low_ptr_offset + base_ptr, max_p);
    v1 = __halves2half2(bottom_data[ptr], bottom_data[ptr + 1]);
  }

  __half2 v2 = __float2half2_rn(0.f);
  if (__hge(h_low, __float2half(0.f)) &&
      __hle(w_high, __float2half(static_cast<float>(width - 1)))) {
    ptr = min(h_low_ptr_offset + w_high_ptr_offset + base_ptr, max_p);
    v2 = __halves2half2(bottom_data[ptr], bottom_data[ptr + 1]);
  }

  __half2 v3 = __float2half2_rn(0.f);
  if (__hle(h_high, __float2half(static_cast<float>(height - 1))) &&
      __hge(w_low, __float2half(0.f))) {
    ptr = min(h_high_ptr_offset + w_low_ptr_offset + base_ptr, max_p);
    v3 = __halves2half2(bottom_data[ptr], bottom_data[ptr + 1]);
  }

  __half2 v4 = __float2half2_rn(0.f);
  if (__hle(h_high, __float2half(static_cast<float>(height - 1))) &&
      __hle(w_high, __float2half(static_cast<float>(width - 1)))) {
    ptr = min(h_high_ptr_offset + w_high_ptr_offset + base_ptr, max_p);
    v4 = __halves2half2(bottom_data[ptr], bottom_data[ptr + 1]);
  }

  const __half2 w1 = __hmul2(hh, hw), w2 = __hmul2(hh, lw),
                w3 = __hmul2(lh, hw), w4 = __hmul2(lh, lw);
  return __hadd2(__hadd2(__hmul2(w1, v1), __hmul2(w2, v2)),
                 __hadd2(__hmul2(w3, v3), __hmul2(w4, v4)));
}

template <typename scalar_t>
__global__ void ms_deformable_im2col_gpu_kernel(
    const int n, const scalar_t *data_value, const int32_t *data_spatial_shapes,
    const scalar_t *data_sampling_loc, const scalar_t *data_attn_weight,
    const int batch_size, const int spatial_size, const int num_heads,
    const int channels, const int num_levels, const int num_query,
    const int num_point, scalar_t *data_col) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp;
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    _temp /= num_query;
    const int b_col = _temp;

    scalar_t *data_col_ptr = data_col + index;
    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;
    scalar_t col = 0;

    const scalar_t *data_value_ptr = data_value + data_value_ptr_init_offset;
    for (int l_col = 0; l_col < num_levels; ++l_col) {
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];

      for (int p_col = 0; p_col < num_point; ++p_col) {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t h_im = loc_h * spatial_h - 0.5;
        const scalar_t w_im = loc_w * spatial_w - 0.5;

        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
          col += ms_deform_attn_im2col_bilinear(data_value_ptr, spatial_h,
                                                spatial_w, num_heads, channels,
                                                h_im, w_im, m_col, c_col) *
                 weight;
        }

        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
      }
      data_value_ptr += qid_stride * spatial_h * spatial_w;
    }
    *data_col_ptr = col;
  }
}

template <>
__global__ void ms_deformable_im2col_gpu_kernel(
    const int n, const __half *data_value, const int32_t *data_spatial_shapes,
    const __half *data_sampling_loc, const __half *data_attn_weight,
    const int batch_size, const int spatial_size, const int num_heads,
    const int channels, const int num_levels, const int num_query,
    const int num_point, __half *data_col) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp;
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    _temp /= num_query;
    const int b_col = _temp;

    __half *data_col_ptr = data_col + index;
    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;
    __half col = __float2half(0.f);

    const __half *data_value_ptr = data_value + data_value_ptr_init_offset;
    for (int l_col = 0; l_col < num_levels; ++l_col) {
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];

      for (int p_col = 0; p_col < num_point; ++p_col) {
        const __half loc_w = data_sampling_loc[data_loc_w_ptr];
        const __half loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const __half weight = data_attn_weight[data_weight_ptr];

        const __half h_im =
            __hsub(__hmul(loc_h, __float2half(static_cast<float>(spatial_h))),
                   __float2half(0.5f));
        const __half w_im =
            __hsub(__hmul(loc_w, __float2half(static_cast<float>(spatial_w))),
                   __float2half(0.5f));

        if (__hgt(h_im, __float2half(-1.f)) &&
            __hgt(w_im, __float2half(-1.f)) &&
            __hlt(h_im, __float2half(static_cast<float>(spatial_h))) &&
            __hlt(w_im, __float2half(static_cast<float>(spatial_w)))) {
          col = __hfma(ms_deform_attn_im2col_bilinear(
                           data_value_ptr, spatial_h, spatial_w, num_heads,
                           channels, h_im, w_im, m_col, c_col),
                       weight, col);
        }

        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
      }
      data_value_ptr += qid_stride * spatial_h * spatial_w;
    }
    *data_col_ptr = col;
  }
}

template <>
__global__ void ms_deformable_im2col_gpu_kernel(
    const int n, const __half2 *data_value, const int32_t *data_spatial_shapes,
    const __half2 *data_sampling_loc, const __half2 *data_attn_weight,
    const int batch_size, const int spatial_size, const int num_heads,
    const int channels, const int num_levels, const int num_query,
    const int num_point, __half2 *data_col) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    int _temp = index;
    const int c_col = _temp % ((channels + 1) / 2);
    const bool last_one = c_col * 2 == channels - 1;
    _temp /= ((channels + 1) / 2);
    const int sampling_index = _temp;
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    _temp /= num_query;
    const int b_col = _temp;

    __half *data_col_ptr = (__half *)data_col +
                           index / ((channels + 1) / 2) * channels + c_col * 2;
    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;
    __half2 col = __float2half2_rn(0.f);

    const __half *data_value_ptr =
        (__half *)data_value + data_value_ptr_init_offset;
    for (int l_col = 0; l_col < num_levels; ++l_col) {
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];

      for (int p_col = 0; p_col < num_point; ++p_col) {
        const __half loc_w = ((__half *)data_sampling_loc)[data_loc_w_ptr];
        const __half loc_h = ((__half *)data_sampling_loc)[data_loc_w_ptr + 1];
        const __half weight = ((__half *)data_attn_weight)[data_weight_ptr];

        const __half h_im =
            __hsub(__hmul(loc_h, __float2half(static_cast<float>(spatial_h))),
                   __float2half(0.5f));
        const __half w_im =
            __hsub(__hmul(loc_w, __float2half(static_cast<float>(spatial_w))),
                   __float2half(0.5f));

        if (__hgt(h_im, __float2half(-1.f)) &&
            __hgt(w_im, __float2half(-1.f)) &&
            __hlt(h_im, __float2half(static_cast<float>(spatial_h))) &&
            __hlt(w_im, __float2half(static_cast<float>(spatial_w)))) {
          col = __hfma2(ms_deform_attn_im2col_bilinear_h2(
                            data_value_ptr, spatial_h, spatial_w, num_heads,
                            channels, h_im, w_im, m_col, c_col, last_one),
                        __half2half2(weight), col);
        }

        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
      }
      data_value_ptr += qid_stride * spatial_h * spatial_w;
    }
    *data_col_ptr = __low2half(col);
    if (!last_one) {
      *(data_col_ptr + 1) = __high2half(col);
    }
  }
}

template <typename scalar_t>
void ms_deformable_im2col_cuda(
    const scalar_t *data_value, const int32_t *data_spatial_shapes,
    const scalar_t *data_sampling_loc, const scalar_t *data_attn_weight,
    const int batch_size, const int spatial_size, const int num_heads,
    const int channels, const int num_levels, const int num_query,
    const int num_point, scalar_t *data_col, cudaStream_t stream) {
  const int num_kernels = batch_size * num_query * num_heads * channels;
  const int num_actual_kernels = batch_size * num_query * num_heads * channels;

  ms_deformable_im2col_gpu_kernel<scalar_t>
      <<<GET_BLOCKS(num_actual_kernels), THREADS_PER_BLOCK, 0, stream>>>(
          num_kernels, data_value, data_spatial_shapes, data_sampling_loc,
          data_attn_weight, batch_size, spatial_size, num_heads, channels,
          num_levels, num_query, num_point, data_col);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in ms_deformable_im2col_cuda: %s\n", cudaGetErrorString(err));
  }
}

template <>
void ms_deformable_im2col_cuda(
    const __half2 *data_value, const int32_t *data_spatial_shapes,
    const __half2 *data_sampling_loc, const __half2 *data_attn_weight,
    const int batch_size, const int spatial_size, const int num_heads,
    const int channels, const int num_levels, const int num_query,
    const int num_point, __half2 *data_col, cudaStream_t stream) {
  const int num_kernels =
      batch_size * num_query * num_heads * ((channels + 1) / 2);
  const int num_actual_kernels =
      batch_size * num_query * num_heads * ((channels + 1) / 2);

  ms_deformable_im2col_gpu_kernel<__half2>
      <<<GET_BLOCKS(num_actual_kernels), THREADS_PER_BLOCK, 0, stream>>>(
          num_kernels, data_value, data_spatial_shapes, data_sampling_loc,
          data_attn_weight, batch_size, spatial_size, num_heads, channels,
          num_levels, num_query, num_point, data_col);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in ms_deformable_im2col_cuda: %s\n", cudaGetErrorString(err));
  }
}

template void ms_deformable_im2col_cuda<float>(
    const float *data_value, const int32_t *data_spatial_shapes,
    const float *data_sampling_loc, const float *data_attn_weight,
    const int batch_size, const int spatial_size, const int num_heads,
    const int channels, const int num_levels, const int num_query,
    const int num_point, float *data_col, cudaStream_t stream);

template void ms_deformable_im2col_cuda<__half>(
    const __half *data_value, const int32_t *data_spatial_shapes,
    const __half *data_sampling_loc, const __half *data_attn_weight,
    const int batch_size, const int spatial_size, const int num_heads,
    const int channels, const int num_levels, const int num_query,
    const int num_point, __half *data_col, cudaStream_t stream);

template void ms_deformable_im2col_cuda<__half2>(
    const __half2 *data_value, const int32_t *data_spatial_shapes,
    const __half2 *data_sampling_loc, const __half2 *data_attn_weight,
    const int batch_size, const int spatial_size, const int num_heads,
    const int channels, const int num_levels, const int num_query,
    const int num_point, __half2 *data_col, cudaStream_t stream);
