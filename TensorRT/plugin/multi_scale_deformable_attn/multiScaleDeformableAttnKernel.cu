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

template <typename T> __forceinline__ __device__ T sign_05(T x) {
  if (x > 0) {
    return 0.5f;
  }
  return -0.5f;
}

template <typename T> __forceinline__ __device__ int8_t T2int8(T a) {
  a = a > 127 ? 127 : a;
  a = a < -128 ? -128 : a;
  return int8_t(a + sign_05<T>(a));
}

template <> __forceinline__ __device__ int8_t T2int8(__half a) {
  a = __hgt(a, __int2half_rn(127)) ? __int2half_rn(127) : a;
  a = __hlt(a, __int2half_rn(-128)) ? __int2half_rn(-128) : a;
  return int8_t(__half2int_rn(a));
}

__forceinline__ __device__ int8_t half2int8(const __half &hval,
                                            const float &scale) {
  __half ret = __hdiv(hval, __float2half(scale));
  return T2int8<__half>(ret);
}

__forceinline__ __device__ void qmulf(const int32_4 &a, int8_4 &c,
                                      const float &b) {
  c.x = T2int8<float>(a.x * b);
  c.y = T2int8<float>(a.y * b);
  c.z = T2int8<float>(a.z * b);
  c.w = T2int8<float>(a.w * b);
}

__forceinline__ __device__ void dp4a(const int32_t *a, const int32_t *b,
                                     int32_t &c) {
#if __CUDA_ARCH__ >= 610
  asm("dp4a.s32.s32 %0, %1, %2, %3;" : "+r"(c) : "r"(*a), "r"(*b), "r"(c));
#else
  auto ap = (int8_4 *)a, bp = (int8_4 *)b;

  c += ap->x * bp->x;
  c += ap->y * bp->y;
  c += ap->z * bp->z;
  c += ap->w * bp->w;
#endif
}

template <typename scalar_t>
__device__ scalar_t ms_deform_attn_im2col_bilinear(
    const scalar_t *&bottom_data, const int &height, const int &width,
    const int &nheads, const int &channels, const scalar_t &h,
    const scalar_t &w) {
  const int h_low = floorf(h);
  const int w_low = floorf(w);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const scalar_t lh = h - h_low;
  const scalar_t lw = w - w_low;
  const scalar_t hh = 1 - lh, hw = 1 - lw;

  const int h_low_ptr_offset = h_low * width;
  const int h_high_ptr_offset = h_low_ptr_offset + width;
  const int w_low_ptr_offset = w_low;
  const int w_high_ptr_offset = w_low_ptr_offset + 1;
  const int step = channels * nheads;

  scalar_t v1 = 0;
  if (h_low >= 0 && w_low >= 0) {
    const int ptr1 = (h_low_ptr_offset + w_low_ptr_offset) * step;
    v1 = bottom_data[ptr1];
  }
  scalar_t v2 = 0;
  if (h_low >= 0 && w_high <= width - 1) {
    const int ptr2 = (h_low_ptr_offset + w_high_ptr_offset) * step;
    v2 = bottom_data[ptr2];
  }
  scalar_t v3 = 0;
  if (h_high <= height - 1 && w_low >= 0) {
    const int ptr3 = (h_high_ptr_offset + w_low_ptr_offset) * step;
    v3 = bottom_data[ptr3];
  }
  scalar_t v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1) {
    const int ptr4 = (h_high_ptr_offset + w_high_ptr_offset) * step;
    v4 = bottom_data[ptr4];
  }

  const scalar_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  const scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

template <>
__device__ __half ms_deform_attn_im2col_bilinear(
    const __half *&bottom_data, const int &height, const int &width,
    const int &nheads, const int &channels, const __half &h, const __half &w) {
  const __half h_low = hfloor(h);
  const __half w_low = hfloor(w);
  const __half h_high = __hadd(h_low, __float2half(1.f));
  const __half w_high = __hadd(w_low, __float2half(1.f));

  const __half lh = __hsub(h, h_low);
  const __half lw = __hsub(w, w_low);
  const __half hh = __hsub(__float2half(1.f), lh),
               hw = __hsub(__float2half(1.f), lw);

  const int h_low_ptr_offset = __half2int_rn(h_low) * width;
  const int h_high_ptr_offset = h_low_ptr_offset + width;
  const int w_low_ptr_offset = __half2int_rn(w_low);
  const int w_high_ptr_offset = w_low_ptr_offset + 1;
  const int step = channels * nheads;

  __half v1 = 0;
  if (__hge(h_low, __float2half(0.f)) && __hge(w_low, __float2half(0.f))) {
    const int ptr1 = (h_low_ptr_offset + w_low_ptr_offset) * step;
    v1 = bottom_data[ptr1];
  }
  __half v2 = 0;
  if (__hge(h_low, __float2half(0.f)) &&
      __hle(w_high, __float2half(static_cast<float>(width - 1)))) {
    const int ptr2 = (h_low_ptr_offset + w_high_ptr_offset) * step;
    v2 = bottom_data[ptr2];
  }
  __half v3 = 0;
  if (__hle(h_high, __float2half(static_cast<float>(height - 1))) &&
      __hge(w_low, __float2half(0.f))) {
    const int ptr3 = (h_high_ptr_offset + w_low_ptr_offset) * step;
    v3 = bottom_data[ptr3];
  }
  __half v4 = 0;
  if (__hle(h_high, __float2half(static_cast<float>(height - 1))) &&
      __hle(w_high, __float2half(static_cast<float>(width - 1)))) {
    const int ptr4 = (h_high_ptr_offset + w_high_ptr_offset) * step;
    v4 = bottom_data[ptr4];
  }
  const __half w1 = __hmul(hh, hw), w2 = __hmul(hh, lw), w3 = __hmul(lh, hw),
               w4 = __hmul(lh, lw);
  return __hadd(__hadd(__hmul(w1, v1), __hmul(w2, v2)),
                __hadd(__hmul(w3, v3), __hmul(w4, v4)));
}

__device__ __half2 ms_deform_attn_im2col_bilinear_h2(
    const __half2 *&bottom_data, const int &height, const int &width,
    const int &nheads, const int &channels, const __half2 &wh) {
  const __half2 wh_low = h2floor(wh);
  const __half2 wh_high = __hadd2(wh_low, __float2half2_rn(1.f));

  const __half2 lwh = __hsub2(wh, wh_low);
  const __half2 hwh = __hsub2(__float2half2_rn(1.f), lwh);

  const int h_low_ptr_offset = __high2float(wh_low) * width;
  const int h_high_ptr_offset = h_low_ptr_offset + width;
  const int w_low_ptr_offset = __low2float(wh_low);
  const int w_high_ptr_offset = w_low_ptr_offset + 1;
  const int step = channels * nheads;

  __half2 v1 = __float2half2_rn(0.f);
  if (__hge(__high2half(wh_low), __float2half(0.f)) &&
      __hge(__low2half(wh_low), __float2half(0.f))) {
    const int ptr1 = (h_low_ptr_offset + w_low_ptr_offset) * step;
    v1 = bottom_data[ptr1];
  }
  __half2 v2 = __float2half2_rn(0.f);
  if (__hge(__high2half(wh_low), __float2half(0.f)) &&
      __hle(__low2half(wh_high), __float2half(static_cast<float>(width - 1)))) {
    const int ptr2 = (h_low_ptr_offset + w_high_ptr_offset) * step;
    v2 = bottom_data[ptr2];
  }
  __half2 v3 = __float2half2_rn(0.f);
  if (__hle(__high2half(wh_high),
            __float2half(static_cast<float>(height - 1))) &&
      __hge(__low2half(wh_low), __float2half(0.f))) {
    const int ptr3 = (h_high_ptr_offset + w_low_ptr_offset) * step;
    v3 = bottom_data[ptr3];
  }
  __half2 v4 = __float2half2_rn(0.f);
  if (__hle(__high2half(wh_high),
            __float2half(static_cast<float>(height - 1))) &&
      __hle(__low2half(wh_high), __float2half(static_cast<float>(width - 1)))) {
    const int ptr4 = (h_high_ptr_offset + w_high_ptr_offset) * step;
    v4 = bottom_data[ptr4];
  }
  const __half2 w1 = __half2half2(__hmul(__high2half(hwh), __low2half(hwh))),
                w2 = __half2half2(__hmul(__high2half(hwh), __low2half(lwh))),
                w3 = __half2half2(__hmul(__high2half(lwh), __low2half(hwh))),
                w4 = __half2half2(__hmul(__high2half(lwh), __low2half(lwh)));
  return __hadd2(__hadd2(__hmul2(w1, v1), __hmul2(w2, v2)),
                 __hadd2(__hmul2(w3, v3), __hmul2(w4, v4)));
}

template <typename scalar_t>
__device__ int8_4 ms_deform_attn_im2col_bilinear_int8(
    const int8_4 *&bottom_data, const int &height, const int &width,
    const int &nheads, const int &channels, const scalar_t &h,
    const scalar_t &w) {
  const int h_low = floorf(h);
  const int w_low = floorf(w);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const scalar_t lh = h - h_low;
  const scalar_t lw = w - w_low;
  const scalar_t hh = 1 - lh, hw = 1 - lw;

  const int h_low_ptr_offset = h_low * width;
  const int h_high_ptr_offset = h_low_ptr_offset + width;
  const int w_low_ptr_offset = w_low;
  const int w_high_ptr_offset = w_low_ptr_offset + 1;
  const int step = channels * nheads;

  const float scale_area = 1 / 127.f;
  int8_4 weight = {
      T2int8<float>((hh * hw) / scale_area),
      T2int8<float>((hh * lw) / scale_area),
      T2int8<float>((lh * hw) / scale_area),
      T2int8<float>((lh * lw) / scale_area),
  };
  int8_4 inps[4] = {0, 0, 0, 0}, output;
  int32_t output_temp;

  if (h_low >= 0 && w_low >= 0) {
    const int ptr1 = (h_low_ptr_offset + w_low_ptr_offset) * step;
    const int8_4 &inp = bottom_data[ptr1];
    inps[0].x = inp.x;
    inps[1].x = inp.y;
    inps[2].x = inp.z;
    inps[3].x = inp.w;
  }

  if (h_low >= 0 && w_high <= width - 1) {
    const int ptr2 = (h_low_ptr_offset + w_high_ptr_offset) * step;
    const int8_4 &inp = bottom_data[ptr2];
    inps[0].y = inp.x;
    inps[1].y = inp.y;
    inps[2].y = inp.z;
    inps[3].y = inp.w;
  }

  if (h_high <= height - 1 && w_low >= 0) {
    const int ptr3 = (h_high_ptr_offset + w_low_ptr_offset) * step;
    const int8_4 &inp = bottom_data[ptr3];
    inps[0].z = inp.x;
    inps[1].z = inp.y;
    inps[2].z = inp.z;
    inps[3].z = inp.w;
  }

  if (h_high <= height - 1 && w_high <= width - 1) {
    const int ptr4 = (h_high_ptr_offset + w_high_ptr_offset) * step;
    const int8_4 &inp = bottom_data[ptr4];
    inps[0].w = inp.x;
    inps[1].w = inp.y;
    inps[2].w = inp.z;
    inps[3].w = inp.w;
  }

  output_temp = 0;
  dp4a((const int32_t *)inps, (const int32_t *)&weight, output_temp);
  output.x = T2int8<float>(output_temp * scale_area);

  output_temp = 0;
  dp4a((const int32_t *)(inps + 1), (const int32_t *)&weight, output_temp);
  output.y = T2int8<float>(output_temp * scale_area);

  output_temp = 0;
  dp4a((const int32_t *)(inps + 2), (const int32_t *)&weight, output_temp);
  output.z = T2int8<float>(output_temp * scale_area);

  output_temp = 0;
  dp4a((const int32_t *)(inps + 3), (const int32_t *)&weight, output_temp);
  output.w = T2int8<float>(output_temp * scale_area);

  return output;
}

__device__ int8_4 ms_deform_attn_im2col_bilinear_int8_h2(
    const int8_4 *&bottom_data, const int &height, const int &width,
    const int &nheads, const int &channels, const __half2 &wh) {
  const __half2 wh_low = h2floor(wh);
  const __half2 wh_high = __hadd2(wh_low, __float2half2_rn(1.f));

  const __half2 lwh = __hsub2(wh, wh_low);
  const __half2 hwh = __hsub2(__float2half2_rn(1.f), lwh);

  const int h_low_ptr_offset = __high2float(wh_low) * width;
  const int h_high_ptr_offset = h_low_ptr_offset + width;
  const int w_low_ptr_offset = __low2float(wh_low);
  const int w_high_ptr_offset = w_low_ptr_offset + 1;
  const int step = channels * nheads;

  const float scale_area = 1 / 127.f;
  int8_4 weight = {
      half2int8(__hmul(__high2half(hwh), __low2half(hwh)), scale_area),
      half2int8(__hmul(__high2half(hwh), __low2half(lwh)), scale_area),
      half2int8(__hmul(__high2half(lwh), __low2half(hwh)), scale_area),
      half2int8(__hmul(__high2half(lwh), __low2half(lwh)), scale_area),
  };
  int8_4 inps[4] = {0, 0, 0, 0}, output;
  int32_t output_temp;

  if (__hge(__high2half(wh_low), __float2half(0.f)) &&
      __hge(__low2half(wh_low), __float2half(0.f))) {
    const int ptr1 = (h_low_ptr_offset + w_low_ptr_offset) * step;
    const int8_4 &inp = bottom_data[ptr1];
    inps[0].x = inp.x;
    inps[1].x = inp.y;
    inps[2].x = inp.z;
    inps[3].x = inp.w;
  }

  if (__hge(__high2half(wh_low), __float2half(0.f)) &&
      __hle(__low2half(wh_high), __float2half(static_cast<float>(width - 1)))) {
    const int ptr2 = (h_low_ptr_offset + w_high_ptr_offset) * step;
    const int8_4 &inp = bottom_data[ptr2];
    inps[0].y = inp.x;
    inps[1].y = inp.y;
    inps[2].y = inp.z;
    inps[3].y = inp.w;
  }

  if (__hle(__high2half(wh_high),
            __float2half(static_cast<float>(height - 1))) &&
      __hge(__low2half(wh_low), __float2half(0.f))) {
    const int ptr3 = (h_high_ptr_offset + w_low_ptr_offset) * step;
    const int8_4 &inp = bottom_data[ptr3];
    inps[0].z = inp.x;
    inps[1].z = inp.y;
    inps[2].z = inp.z;
    inps[3].z = inp.w;
  }

  if (__hle(__high2half(wh_high),
            __float2half(static_cast<float>(height - 1))) &&
      __hle(__low2half(wh_high), __float2half(static_cast<float>(width - 1)))) {
    const int ptr4 = (h_high_ptr_offset + w_high_ptr_offset) * step;
    const int8_4 &inp = bottom_data[ptr4];
    inps[0].w = inp.x;
    inps[1].w = inp.y;
    inps[2].w = inp.z;
    inps[3].w = inp.w;
  }

  output_temp = 0;
  dp4a((const int32_t *)inps, (const int32_t *)&weight, output_temp);
  output.x = T2int8<float>(output_temp * scale_area);

  output_temp = 0;
  dp4a((const int32_t *)(inps + 1), (const int32_t *)&weight, output_temp);
  output.y = T2int8<float>(output_temp * scale_area);

  output_temp = 0;
  dp4a((const int32_t *)(inps + 2), (const int32_t *)&weight, output_temp);
  output.z = T2int8<float>(output_temp * scale_area);

  output_temp = 0;
  dp4a((const int32_t *)(inps + 3), (const int32_t *)&weight, output_temp);
  output.w = T2int8<float>(output_temp * scale_area);

  return output;
}

template <typename scalar_t>
__global__ void ms_deformable_im2col_gpu_kernel(
    const int n, const scalar_t *data_value, const int32_t *data_spatial_shapes,
    const scalar_t *data_reference_points,
    const scalar_t *data_sampling_offsets, const scalar_t *data_attn_weight,
    const int batch_size, const int spatial_size, const int num_heads,
    const int channels, const int num_levels, const int num_query,
    const int num_point, const int points_per_group, scalar_t *data_col) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    const int temp = index;
    const int channel_index = index % channels;
    index /= channels;
    const int head_index = index % num_heads;
    index /= num_heads;
    const int query_index = index % num_query;
    const int batch_index = index / num_query;

    const scalar_t *data_value_ptr =
        data_value +
        (batch_index * spatial_size * num_heads + head_index) * channels +
        channel_index;
    int data_weight_ptr =
        ((batch_index * num_query + query_index) * num_heads + head_index) *
        num_levels * num_point;
    int data_offset_w_ptr = data_weight_ptr << 1;
    int data_points_ptr =
        (batch_index * num_query + query_index) * points_per_group * 2;
    scalar_t *data_output_ptr = data_col + temp;
    scalar_t output = 0;

    for (int level_index = 0; level_index < num_levels; ++level_index) {
      const int spatial_h_ptr = level_index << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];

      for (int point_index = 0; point_index < num_point; ++point_index) {
        const int point_index_per_group = point_index % points_per_group;
        const scalar_t reference_point_x =
            data_reference_points[data_points_ptr + point_index_per_group * 2];
        const scalar_t reference_point_y =
            data_reference_points[data_points_ptr + point_index_per_group * 2 +
                                  1];
        const scalar_t loc_w = reference_point_x * spatial_w +
                               data_sampling_offsets[data_offset_w_ptr];
        const scalar_t loc_h = reference_point_y * spatial_h +
                               data_sampling_offsets[data_offset_w_ptr + 1];

        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t h_im = loc_h - 0.5f;
        const scalar_t w_im = loc_w - 0.5f;

        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
          output += ms_deform_attn_im2col_bilinear(data_value_ptr, spatial_h,
                                                   spatial_w, num_heads,
                                                   channels, h_im, w_im) *
                    weight;
        }

        data_weight_ptr += 1;
        data_offset_w_ptr += 2;
      }
      data_value_ptr += spatial_h * spatial_w * channels * num_heads;
    }
    *data_output_ptr = output;
  }
}

template <>
__global__ void ms_deformable_im2col_gpu_kernel(
    const int n, const __half *data_value, const int32_t *data_spatial_shapes,
    const __half *data_reference_points, const __half *data_sampling_offsets,
    const __half *data_attn_weight, const int batch_size,
    const int spatial_size, const int num_heads, const int channels,
    const int num_levels, const int num_query, const int num_point,
    const int points_per_group, __half *data_col) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    const int temp = index;
    const int channel_index = index % channels;
    index /= channels;
    const int head_index = index % num_heads;
    index /= num_heads;
    const int query_index = index % num_query;
    const int batch_index = index / num_query;

    const __half *data_value_ptr =
        data_value +
        (batch_index * spatial_size * num_heads + head_index) * channels +
        channel_index;
    int data_weight_ptr =
        ((batch_index * num_query + query_index) * num_heads + head_index) *
        num_levels * num_point;
    int data_offset_w_ptr = data_weight_ptr << 1;
    int data_points_ptr =
        (batch_index * num_query + query_index) * points_per_group * 2;
    __half *data_output_ptr = data_col + temp;
    __half output = 0;

    for (int level_index = 0; level_index < num_levels; ++level_index) {
      const int spatial_h_ptr = level_index << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];

      for (int point_index = 0; point_index < num_point; ++point_index) {
        const int point_index_per_group = point_index % points_per_group;
        const __half reference_point_x =
            data_reference_points[data_points_ptr + point_index_per_group * 2];
        const __half reference_point_y =
            data_reference_points[data_points_ptr + point_index_per_group * 2 +
                                  1];
        const __half loc_w = __hfma(reference_point_x, __int2half_rn(spatial_w),
                                    data_sampling_offsets[data_offset_w_ptr]);
        const __half loc_h =
            __hfma(reference_point_y, __int2half_rn(spatial_h),
                   data_sampling_offsets[data_offset_w_ptr + 1]);

        const __half weight = data_attn_weight[data_weight_ptr];

        const __half h_im = __hsub(loc_h, __float2half(0.5f));
        const __half w_im = __hsub(loc_w, __float2half(0.5f));

        if (__hgt(h_im, __float2half(-1.f)) &&
            __hgt(w_im, __float2half(-1.f)) &&
            __hlt(h_im, __int2half_rn(spatial_h)) &&
            __hlt(w_im, __int2half_rn(spatial_w))) {
          output += ms_deform_attn_im2col_bilinear(data_value_ptr, spatial_h,
                                                   spatial_w, num_heads,
                                                   channels, h_im, w_im) *
                    weight;
        }

        data_weight_ptr += 1;
        data_offset_w_ptr += 2;
      }
      data_value_ptr += spatial_h * spatial_w * channels * num_heads;
    }
    *data_output_ptr = output;
  }
}

__global__ void ms_deformable_im2col_gpu_kernel_h2(
    const int n, const __half2 *data_value, const int32_t *data_spatial_shapes,
    const __half2 *data_reference_points, const __half2 *data_sampling_offsets,
    const __half *data_attn_weight, const int batch_size,
    const int spatial_size, const int num_heads, const int channels,
    const int num_levels, const int num_query, const int num_point,
    const int points_per_group, __half2 *data_col) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    const int temp = index;
    const int channel_index = index % channels;
    index /= channels;
    const int head_index = index % num_heads;
    index /= num_heads;
    const int query_index = index % num_query;
    const int batch_index = index / num_query;

    const __half2 *data_value_ptr =
        data_value +
        (batch_index * spatial_size * num_heads + head_index) * channels +
        channel_index;
    int data_weight_ptr =
        ((batch_index * num_query + query_index) * num_heads + head_index) *
        num_levels * num_point;
    int data_points_ptr =
        (batch_index * num_query + query_index) * points_per_group;
    __half2 *data_output_ptr = data_col + temp;
    __half2 output = __float2half2_rn(0.f);

    for (int level_index = 0; level_index < num_levels; ++level_index) {
      const int spatial_h_ptr = level_index << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];

      for (int point_index = 0; point_index < num_point; ++point_index) {
        const int point_index_per_group = point_index % points_per_group;
        const __half2 reference_point_xy =
            data_reference_points[data_points_ptr + point_index_per_group];
        const __half2 offset = __hsub2(data_sampling_offsets[data_weight_ptr],
                                       __float2half2_rn(0.5f));
        const __half2 wh_im =
            __hfma2(reference_point_xy,
                    __floats2half2_rn(static_cast<float>(spatial_w),
                                      static_cast<float>(spatial_h)),
                    offset);

        const __half2 weight = __half2half2(data_attn_weight[data_weight_ptr]);

        const __half2 condition = __hmul2(
            __hgt2(wh_im, __float2half2_rn(-1.f)),
            __hlt2(wh_im, __floats2half2_rn(static_cast<float>(spatial_w),
                                            static_cast<float>(spatial_h))));

        if (__low2float(condition) * __high2float(condition)) {
          output = __hfma2(ms_deform_attn_im2col_bilinear_h2(
                               data_value_ptr, spatial_h, spatial_w, num_heads,
                               channels, wh_im),
                           weight, output);
        }
        data_weight_ptr += 1;
      }
      data_value_ptr += spatial_h * spatial_w * channels * num_heads;
    }
    *data_output_ptr = output;
  }
}

template <typename scalar_t>
__global__ void ms_deformable_im2col_gpu_kernel_int8(
    const int n, const int8_4 *data_value, float scale_value,
    const int32_t *data_spatial_shapes, const scalar_t *data_reference_points,
    const int8_t *data_sampling_offsets, float scale_offset,
    const int8_t *data_attn_weight, float scale_weight, const int batch_size,
    const int spatial_size, const int num_heads, const int channels,
    const int num_levels, const int num_query, const int num_point,
    const int points_per_group, int8_4 *data_col, float scale_out) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    const int temp = index;
    const int channel_index = index % channels;
    index /= channels;
    const int head_index = index % num_heads;
    index /= num_heads;
    const int query_index = index % num_query;
    const int batch_index = index / num_query;

    const int8_4 *data_value_ptr =
        data_value +
        (batch_index * spatial_size * num_heads + head_index) * channels +
        channel_index;
    int data_weight_ptr =
        ((batch_index * num_query + query_index) * num_heads + head_index) *
        num_levels * num_point;
    int data_offset_w_ptr = data_weight_ptr << 1;
    int data_points_ptr =
        (batch_index * num_query + query_index) * points_per_group * 2;
    int8_4 *data_output_ptr = data_col + temp;
    int32_4 output = 0;
    const float scale_o = scale_weight * scale_value / scale_out;

    for (int level_index = 0; level_index < num_levels; ++level_index) {
      const int spatial_h_ptr = level_index << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];

      for (int point_index = 0; point_index < num_point / 4; point_index++) {
        int8_t values[16] = {0};
        int8_t weights[16] = {0};
#pragma unroll 4
        for (int i = 0; i < 4; i++, data_weight_ptr++, data_offset_w_ptr += 2) {
          const int point_index_per_group =
              (point_index * 4 + i) % points_per_group;
          const float reference_point_x =
              data_reference_points[data_points_ptr +
                                    point_index_per_group * 2];
          const float reference_point_y =
              data_reference_points[data_points_ptr +
                                    point_index_per_group * 2 + 1];
          const float loc_w =
              reference_point_x * spatial_w +
              data_sampling_offsets[data_offset_w_ptr] * scale_offset;
          const float loc_h =
              reference_point_y * spatial_h +
              data_sampling_offsets[data_offset_w_ptr + 1] * scale_offset;

          const float h_im = loc_h - 0.5f;
          const float w_im = loc_w - 0.5f;

          if (!(h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w))
            continue;
          const int8_4 &val = ms_deform_attn_im2col_bilinear_int8(
              data_value_ptr, spatial_h, spatial_w, num_heads, channels, h_im,
              w_im);

          values[i] = val.x;
          values[i + 4] = val.y;
          values[i + 8] = val.z;
          values[i + 12] = val.w;
          const int8_4 weight = data_attn_weight[data_weight_ptr];
          weights[i] = weight.x;
          weights[i + 4] = weight.y;
          weights[i + 8] = weight.z;
          weights[i + 12] = weight.w;
        }

        dp4a((const int32_t *)values, (const int32_t *)weights, output.x);
        dp4a((const int32_t *)(values + 4), (const int32_t *)(weights + 4),
             output.y);
        dp4a((const int32_t *)(values + 8), (const int32_t *)(weights + 8),
             output.z);
        dp4a((const int32_t *)(values + 12), (const int32_t *)(weights + 12),
             output.w);
      }
      data_value_ptr += spatial_h * spatial_w * channels * num_heads;
    }
    qmulf(output, *data_output_ptr, scale_o);
  }
}

template <>
__global__ void ms_deformable_im2col_gpu_kernel_int8(
    const int n, const int8_4 *data_value, float scale_value,
    const int32_t *data_spatial_shapes, const __half2 *data_reference_points,
    const int8_t *data_sampling_offsets, float scale_offset,
    const int8_t *data_attn_weight, float scale_weight, const int batch_size,
    const int spatial_size, const int num_heads, const int channels,
    const int num_levels, const int num_query, const int num_point,
    const int points_per_group, int8_4 *data_col, float scale_out) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    const int temp = index;
    const int channel_index = index % channels;
    index /= channels;
    const int head_index = index % num_heads;
    index /= num_heads;
    const int query_index = index % num_query;
    const int batch_index = index / num_query;

    const int8_4 *data_value_ptr =
        data_value +
        (batch_index * spatial_size * num_heads + head_index) * channels +
        channel_index;
    int data_weight_ptr =
        ((batch_index * num_query + query_index) * num_heads + head_index) *
        num_levels * num_point;
    int data_offset_w_ptr = data_weight_ptr << 1;
    int data_points_ptr =
        (batch_index * num_query + query_index) * points_per_group;
    int8_4 *data_output_ptr = data_col + temp;
    int32_4 output = 0;
    const float scale_o = scale_weight * scale_value / scale_out;

    for (int level_index = 0; level_index < num_levels; ++level_index) {
      const int spatial_h_ptr = level_index << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];

      for (int point_index = 0; point_index < num_point / 4; point_index++) {
        int8_t values[16] = {0};
        int8_t weights[16] = {0};
#pragma unroll 4
        for (int i = 0; i < 4; i++, data_weight_ptr++, data_offset_w_ptr += 2) {
          const int point_index_per_group =
              (point_index * 4 + i) % points_per_group;
          const __half2 reference_point_xy =
              data_reference_points[data_points_ptr + point_index_per_group];
          const __half2 offset = __hsub2(
              __floats2half2_rn(
                  data_sampling_offsets[data_offset_w_ptr] * scale_offset,
                  data_sampling_offsets[data_offset_w_ptr + 1] * scale_offset),
              __float2half2_rn(0.5f));
          const __half2 wh_im =
              __hfma2(reference_point_xy,
                      __floats2half2_rn(static_cast<float>(spatial_w),
                                        static_cast<float>(spatial_h)),
                      offset);

          const __half2 condition = __hmul2(
              __hgt2(wh_im, __float2half2_rn(-1.f)),
              __hlt2(wh_im, __floats2half2_rn(static_cast<float>(spatial_w),
                                              static_cast<float>(spatial_h))));

          if (!(__low2float(condition) * __high2float(condition)))
            continue;

          const int8_4 &val = ms_deform_attn_im2col_bilinear_int8_h2(
              data_value_ptr, spatial_h, spatial_w, num_heads, channels, wh_im);

          values[i] = val.x;
          values[i + 4] = val.y;
          values[i + 8] = val.z;
          values[i + 12] = val.w;
          const int8_4 weight = data_attn_weight[data_weight_ptr];
          weights[i] = weight.x;
          weights[i + 4] = weight.y;
          weights[i + 8] = weight.z;
          weights[i + 12] = weight.w;
        }

        dp4a((const int32_t *)values, (const int32_t *)weights, output.x);
        dp4a((const int32_t *)(values + 4), (const int32_t *)(weights + 4),
             output.y);
        dp4a((const int32_t *)(values + 8), (const int32_t *)(weights + 8),
             output.z);
        dp4a((const int32_t *)(values + 12), (const int32_t *)(weights + 12),
             output.w);
      }
      data_value_ptr += spatial_h * spatial_w * channels * num_heads;
    }
    qmulf(output, *data_output_ptr, scale_o);
  }
}

template <typename scalar_t>
void ms_deformable_im2col_cuda(const scalar_t *data_value,
                               const int32_t *data_spatial_shapes,
                               const scalar_t *data_reference_points,
                               const scalar_t *data_sampling_offsets,
                               const scalar_t *data_attn_weight,
                               const int batch_size, const int spatial_size,
                               const int num_heads, const int channels,
                               const int num_levels, const int num_query,
                               const int num_point, const int points_per_group,
                               scalar_t *data_col, cudaStream_t stream) {
  const int num_kernels = num_query * num_heads * channels;
  const int value_step = num_heads * spatial_size * channels;
  const int output_step = num_heads * num_query * channels;
  const int points_step = num_query * points_per_group * 2;
  const int weight_step = num_heads * num_query * num_levels * num_point;
  const int offset_step = weight_step * 2;

  for (int batch_index = 0; batch_index < batch_size; batch_index++) {
    ms_deformable_im2col_gpu_kernel<scalar_t>
        <<<GET_BLOCKS(num_kernels), THREADS_PER_BLOCK, 0, stream>>>(
            num_kernels, data_value, data_spatial_shapes, data_reference_points,
            data_sampling_offsets, data_attn_weight, 1, spatial_size, num_heads,
            channels, num_levels, num_query, num_point, points_per_group,
            data_col);
    data_value += value_step;
    data_col += output_step;
    data_reference_points += points_step;
    data_sampling_offsets += offset_step;
    data_attn_weight += weight_step;
  }
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in ms_deformable_im2col_cuda: %s\n", cudaGetErrorString(err));
  }
}

template <>
void ms_deformable_im2col_cuda(
    const __half *data_value, const int32_t *data_spatial_shapes,
    const __half *data_reference_points, const __half *data_sampling_offsets,
    const __half *data_attn_weight, const int batch_size,
    const int spatial_size, const int num_heads, const int channels,
    const int num_levels, const int num_query, const int num_point,
    const int points_per_group, __half *data_col, cudaStream_t stream) {
  const int num_kernels = num_query * num_heads * channels;
  const int value_step = num_heads * spatial_size * channels;
  const int output_step = num_heads * num_query * channels;
  const int points_step = num_query * points_per_group * 2;
  const int weight_step = num_heads * num_query * num_levels * num_point;
  const int offset_step = weight_step * 2;

  for (int batch_index = 0; batch_index < batch_size; batch_index++) {
    ms_deformable_im2col_gpu_kernel<__half>
        <<<GET_BLOCKS(num_kernels), THREADS_PER_BLOCK, 0, stream>>>(
            num_kernels, data_value, data_spatial_shapes, data_reference_points,
            data_sampling_offsets, data_attn_weight, 1, spatial_size, num_heads,
            channels, num_levels, num_query, num_point, points_per_group,
            data_col);
    data_value += value_step;
    data_col += output_step;
    data_reference_points += points_step;
    data_sampling_offsets += offset_step;
    data_attn_weight += weight_step;
  }
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in ms_deformable_im2col_cuda: %s\n", cudaGetErrorString(err));
  }
}

void ms_deformable_im2col_cuda_h2(
    const __half2 *data_value, const int32_t *data_spatial_shapes,
    const __half2 *data_reference_points, const __half2 *data_sampling_offsets,
    const __half *data_attn_weight, const int batch_size,
    const int spatial_size, const int num_heads, int channels,
    const int num_levels, const int num_query, const int num_point,
    const int points_per_group, __half2 *data_col, cudaStream_t stream) {
  channels /= 2;
  const int num_kernels = num_query * num_heads * channels;
  const int value_step = num_heads * spatial_size * channels;
  const int output_step = num_heads * num_query * channels;
  const int points_step = num_query * points_per_group;
  const int weight_step = num_heads * num_query * num_levels * num_point;

  for (int batch_index = 0; batch_index < batch_size; batch_index++) {
    ms_deformable_im2col_gpu_kernel_h2<<<GET_BLOCKS(num_kernels),
                                         THREADS_PER_BLOCK, 0, stream>>>(
        num_kernels, data_value, data_spatial_shapes, data_reference_points,
        data_sampling_offsets, data_attn_weight, 1, spatial_size, num_heads,
        channels, num_levels, num_query, num_point, points_per_group, data_col);
    data_value += value_step;
    data_col += output_step;
    data_reference_points += points_step;
    data_sampling_offsets += weight_step;
    data_attn_weight += weight_step;
  }
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in ms_deformable_im2col_cuda: %s\n", cudaGetErrorString(err));
  }
}

template <typename scalar_t>
void ms_deformable_im2col_cuda_int8(
    const int8_4 *data_value, float scale_value,
    const int32_t *data_spatial_shapes, const scalar_t *data_reference_points,
    const int8_t *data_sampling_offsets, float scale_offset,
    const int8_t *data_attn_weight, float scale_weight, const int batch_size,
    const int spatial_size, const int num_heads, int channels,
    const int num_levels, const int num_query, const int num_point,
    const int points_per_group, int8_4 *data_col, float scale_out,
    cudaStream_t stream) {
  channels /= 4;
  const int num_kernels = num_query * num_heads * channels;
  const int value_step = num_heads * spatial_size * channels;
  const int output_step = num_heads * num_query * channels;
  const int points_step = num_query * points_per_group * 2;
  const int weight_step = num_heads * num_query * num_levels * num_point;
  const int offset_step = weight_step * 2;

  for (int batch_index = 0; batch_index < batch_size; batch_index++) {
    ms_deformable_im2col_gpu_kernel_int8<scalar_t>
        <<<GET_BLOCKS(num_kernels), THREADS_PER_BLOCK, 0, stream>>>(
            num_kernels, data_value, scale_value, data_spatial_shapes,
            data_reference_points, data_sampling_offsets, scale_offset,
            data_attn_weight, scale_weight, 1, spatial_size, num_heads,
            channels, num_levels, num_query, num_point, points_per_group,
            data_col, scale_out);
    data_value += value_step;
    data_col += output_step;
    data_reference_points += points_step;
    data_sampling_offsets += offset_step;
    data_attn_weight += weight_step;
  }
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in ms_deformable_im2col_cuda: %s\n", cudaGetErrorString(err));
  }
}

template <>
void ms_deformable_im2col_cuda_int8(
    const int8_4 *data_value, float scale_value,
    const int32_t *data_spatial_shapes, const __half2 *data_reference_points,
    const int8_t *data_sampling_offsets, float scale_offset,
    const int8_t *data_attn_weight, float scale_weight, const int batch_size,
    const int spatial_size, const int num_heads, int channels,
    const int num_levels, const int num_query, const int num_point,
    const int points_per_group, int8_4 *data_col, float scale_out,
    cudaStream_t stream) {
  channels /= 4;
  const int num_kernels = num_query * num_heads * channels;
  const int value_step = num_heads * spatial_size * channels;
  const int output_step = num_heads * num_query * channels;
  const int points_step = num_query * points_per_group;
  const int weight_step = num_heads * num_query * num_levels * num_point;
  const int offset_step = weight_step * 2;

  for (int batch_index = 0; batch_index < batch_size; batch_index++) {
    ms_deformable_im2col_gpu_kernel_int8<__half2>
        <<<GET_BLOCKS(num_kernels), THREADS_PER_BLOCK, 0, stream>>>(
            num_kernels, data_value, scale_value, data_spatial_shapes,
            data_reference_points, data_sampling_offsets, scale_offset,
            data_attn_weight, scale_weight, 1, spatial_size, num_heads,
            channels, num_levels, num_query, num_point, points_per_group,
            data_col, scale_out);
    data_value += value_step;
    data_col += output_step;
    data_reference_points += points_step;
    data_sampling_offsets += offset_step;
    data_attn_weight += weight_step;
  }
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in ms_deformable_im2col_cuda: %s\n", cudaGetErrorString(err));
  }
}

template void ms_deformable_im2col_cuda<float>(
    const float *data_value, const int32_t *data_spatial_shapes,
    const float *data_reference_points, const float *data_sampling_offsets,
    const float *data_attn_weight, const int batch_size, const int spatial_size,
    const int num_heads, const int channels, const int num_levels,
    const int num_query, const int num_point, const int points_per_group,
    float *data_col, cudaStream_t stream);

template void ms_deformable_im2col_cuda<__half>(
    const __half *data_value, const int32_t *data_spatial_shapes,
    const __half *data_reference_points, const __half *data_sampling_offsets,
    const __half *data_attn_weight, const int batch_size,
    const int spatial_size, const int num_heads, const int channels,
    const int num_levels, const int num_query, const int num_point,
    const int points_per_group, __half *data_col, cudaStream_t stream);

template void ms_deformable_im2col_cuda_int8<float>(
    const int8_4 *data_value, float scale_value,
    const int32_t *data_spatial_shapes, const float *data_reference_points,
    const int8_t *data_sampling_offsets, float scale_offset,
    const int8_t *data_attn_weight, float scale_weight, const int batch_size,
    const int spatial_size, const int num_heads, int channels,
    const int num_levels, const int num_query, const int num_point,
    const int points_per_group, int8_4 *data_col, float scale_out,
    cudaStream_t stream);

template void ms_deformable_im2col_cuda_int8<__half2>(
    const int8_4 *data_value, float scale_value,
    const int32_t *data_spatial_shapes, const __half2 *data_reference_points,
    const int8_t *data_sampling_offsets, float scale_offset,
    const int8_t *data_attn_weight, float scale_weight, const int batch_size,
    const int spatial_size, const int num_heads, int channels,
    const int num_levels, const int num_query, const int num_point,
    const int points_per_group, int8_4 *data_col, float scale_out,
    cudaStream_t stream);
