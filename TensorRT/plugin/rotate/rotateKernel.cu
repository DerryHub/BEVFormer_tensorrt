//
// Created by Derry Lin on 2022/11/21.
//

#include <cstdio>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "cuda_helper.h"
#include "helper.h"
#include "rotateKernel.h"
#include <cstdio>

static __forceinline__ __device__ bool within_bounds_2d(int h, int w, int H,
                                                        int W) {
  return h >= 0 && h < H && w >= 0 && w < W;
}

static __forceinline__ __device__ __half2 within_bounds_2d_half2(__half2 h,
                                                                 __half2 w,
                                                                 __half2 H,
                                                                 __half2 W) {
  return __hmul2(__hmul2(__hge2(h, __float2half2_rn(0.)), __hlt2(h, H)),
                 __hmul2(__hge2(w, __float2half2_rn(0.)), __hlt2(w, W)));
}

template <typename scalar_t>
__forceinline__ __device__ scalar_t safe_downgrade_to_int_range(scalar_t x) {
  if (x > INT_MAX - 1 || x < INT_MIN || !::isfinite(static_cast<double>(x)))
    return static_cast<scalar_t>(-100.0);
  return x;
}

template <>
__forceinline__ __device__ __half safe_downgrade_to_int_range(__half x) {
  if (__hisinf(x))
    return static_cast<__half>(-100.0);
  return x;
}

template <>
__forceinline__ __device__ __half2 safe_downgrade_to_int_range(__half2 x) {
  __half low, high;

  low = __low2half(x);
  if (__hisinf(low)) {
    low = static_cast<__half>(-100.0);
  }

  high = __high2half(x);
  if (__hisinf(high)) {
    high = static_cast<__half>(-100.0);
  }

  return __halves2half2(low, high);
}

template <typename scalar_t>
__forceinline__ __device__ scalar_t
grid_sampler_compute_source_index(scalar_t coord, int size) {
  coord = ((coord + 1.f) * size - 1) / 2;
  coord = safe_downgrade_to_int_range(coord);
  return coord;
}

template <>
__forceinline__ __device__ __half
grid_sampler_compute_source_index(__half coord, int size) {
  coord = __hfma(__hfma(coord, __float2half(0.5), __float2half(0.5)),
                 __float2half(static_cast<float>(size)), __float2half(-0.5));
  coord = safe_downgrade_to_int_range(coord);
  return coord;
}

template <>
__forceinline__ __device__ __half2
grid_sampler_compute_source_index(__half2 coord, int size) {
  coord = __hfma2(__hfma2(coord, __float2half2_rn(0.5), __float2half2_rn(0.5)),
                  __float2half2_rn(static_cast<float>(size)),
                  __float2half2_rn(-0.5));
  coord = safe_downgrade_to_int_range(coord);
  return coord;
}

template <typename scalar_t>
__global__ void rotateKernel(const int nthreads, scalar_t *output,
                             const scalar_t *input, const scalar_t *angle,
                             const scalar_t *center, int channel, int height,
                             int width, RotateInterpolation interp) {
  int inp_sC = width * height;
  int inp_sH = width;
  int inp_sW = 1;

  const scalar_t ang = -(*angle) * M_PI / 180.f;
  const scalar_t cx = center[0] - 0.5 * width, cy = center[1] - 0.5 * height;
  const scalar_t matrix[6] = {std::cos(ang),
                              std::sin(ang),
                              -cx * std::cos(ang) - cy * std::sin(ang) + cx,
                              -std::sin(ang),
                              std::cos(ang),
                              cx * std::sin(ang) - cy * std::cos(ang) + cy};

  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int w = index % width;
    const int h = (index / width) % height;
    const scalar_t x = -width * 0.5 + 0.5 + w, y = -height * 0.5 + 0.5 + h;

    const scalar_t grid_x =
        (matrix[0] * x + matrix[1] * y + matrix[2]) / (0.5 * width);
    const scalar_t grid_y =
        (matrix[3] * x + matrix[4] * y + matrix[5]) / (0.5 * height);

    scalar_t ix = grid_sampler_compute_source_index(grid_x, width);
    scalar_t iy = grid_sampler_compute_source_index(grid_y, height);

    if (interp == RotateInterpolation::Bilinear) {
      // get NE, NW, SE, SW pixel values from (x, y)
      int ix_nw = static_cast<int>(::floor(ix));
      int iy_nw = static_cast<int>(::floor(iy));
      int ix_ne = ix_nw + 1;
      int iy_ne = iy_nw;
      int ix_sw = ix_nw;
      int iy_sw = iy_nw + 1;
      int ix_se = ix_nw + 1;
      int iy_se = iy_nw + 1;

      // get surfaces to each neighbor:
      scalar_t nw = (ix_se - ix) * (iy_se - iy);
      scalar_t ne = (ix - ix_sw) * (iy_sw - iy);
      scalar_t sw = (ix_ne - ix) * (iy - iy_ne);
      scalar_t se = (ix - ix_nw) * (iy - iy_nw);

      // calculate bilinear weighted pixel value and set output pixel
      auto inp_ptr = input;
      auto out_ptr = output + h * inp_sH + w * inp_sW;
      for (int c = 0; c < channel; ++c, inp_ptr += inp_sC, out_ptr += inp_sC) {
        *out_ptr = static_cast<scalar_t>(0);
        if (within_bounds_2d(iy_nw, ix_nw, height, width)) {
          *out_ptr += inp_ptr[iy_nw * inp_sH + ix_nw * inp_sW] * nw;
        }
        if (within_bounds_2d(iy_ne, ix_ne, height, width)) {
          *out_ptr += inp_ptr[iy_ne * inp_sH + ix_ne * inp_sW] * ne;
        }
        if (within_bounds_2d(iy_sw, ix_sw, height, width)) {
          *out_ptr += inp_ptr[iy_sw * inp_sH + ix_sw * inp_sW] * sw;
        }
        if (within_bounds_2d(iy_se, ix_se, height, width)) {
          *out_ptr += inp_ptr[iy_se * inp_sH + ix_se * inp_sW] * se;
        }
      }
    } else if (interp == RotateInterpolation::Nearest) {
      int ix_nearest = static_cast<int>(::round(ix));
      int iy_nearest = static_cast<int>(::round(iy));

      // assign nearest neighbor pixel value to output pixel
      auto inp_ptr = input;
      auto out_ptr = output + h * inp_sH + w * inp_sW;
      for (int c = 0; c < channel; ++c, inp_ptr += inp_sC, out_ptr += inp_sC) {
        if (within_bounds_2d(iy_nearest, ix_nearest, height, width)) {
          *out_ptr = inp_ptr[iy_nearest * inp_sH + ix_nearest * inp_sW];
        } else {
          *out_ptr = static_cast<scalar_t>(0);
        }
      }
    }
  }
}

template <>
__global__ void rotateKernel(const int nthreads, __half *output,
                             const __half *input, const __half *angle,
                             const __half *center, int channel, int height,
                             int width, RotateInterpolation interp) {
  int inp_sC = width * height;
  int inp_sH = width;
  int inp_sW = 1;

  const __half ang = __hmul(*angle, __float2half(-M_PI / 180.f));
  const __half cx = __hsub(center[0], __float2half(0.5 * width)),
               cy = __hsub(center[1], __float2half(0.5 * height));
  const __half matrix[6] = {
      hcos(ang),  hsin(ang), -cx * hcos(ang) - cy * hsin(ang) + cx,
      -hsin(ang), hcos(ang), cx * hsin(ang) - cy * hcos(ang) + cy};

  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int w = index % width;
    const int h = (index / width) % height;

    const __half x = __hadd(__hfma(__float2half(-0.5),
                                   __float2half(static_cast<float>(width)),
                                   __float2half(0.5)),
                            __float2half(static_cast<float>(w)));
    const __half y = __hadd(__hfma(__float2half(-0.5),
                                   __float2half(static_cast<float>(height)),
                                   __float2half(0.5)),
                            __float2half(static_cast<float>(h)));

    const __half grid_x = __hdiv(
        __hadd(__hmul(matrix[0], x), __hfma(matrix[1], y, matrix[2])),
        __hmul(__float2half(0.5), __float2half(static_cast<float>(width))));
    const __half grid_y = __hdiv(
        __hadd(__hmul(matrix[3], x), __hfma(matrix[4], y, matrix[5])),
        __hmul(__float2half(0.5), __float2half(static_cast<float>(height))));

    __half ix = grid_sampler_compute_source_index(grid_x, width);
    __half iy = grid_sampler_compute_source_index(grid_y, height);

    if (interp == RotateInterpolation::Bilinear) {
      // get NE, NW, SE, SW pixel values from (x, y)
      int ix_nw = static_cast<int>(hfloor(ix));
      int iy_nw = static_cast<int>(hfloor(iy));
      int ix_ne = ix_nw + 1;
      int iy_ne = iy_nw;
      int ix_sw = ix_nw;
      int iy_sw = iy_nw + 1;
      int ix_se = ix_nw + 1;
      int iy_se = iy_nw + 1;

      // get surfaces to each neighbor:
      __half nw = __hmul(__hsub(static_cast<__half>(ix_se), ix),
                         __hsub(static_cast<__half>(iy_se), iy));
      __half ne = __hmul(__hsub(ix, static_cast<__half>(ix_sw)),
                         __hsub(static_cast<__half>(iy_sw), iy));
      __half sw = __hmul(__hsub(static_cast<__half>(ix_ne), ix),
                         __hsub(iy, static_cast<__half>(iy_ne)));
      __half se = __hmul(__hsub(ix, static_cast<__half>(ix_nw)),
                         __hsub(iy, static_cast<__half>(iy_nw)));

      // calculate bilinear weighted pixel value and set output pixel
      auto inp_ptr = input;
      auto out_ptr = output + h * inp_sH + w * inp_sW;
      for (int c = 0; c < channel; ++c, inp_ptr += inp_sC, out_ptr += inp_sC) {
        *out_ptr = static_cast<__half>(0);
        if (within_bounds_2d(iy_nw, ix_nw, height, width)) {
          *out_ptr =
              __hfma(inp_ptr[iy_nw * inp_sH + ix_nw * inp_sW], nw, *out_ptr);
        }
        if (within_bounds_2d(iy_ne, ix_ne, height, width)) {
          *out_ptr =
              __hfma(inp_ptr[iy_ne * inp_sH + ix_ne * inp_sW], ne, *out_ptr);
        }
        if (within_bounds_2d(iy_sw, ix_sw, height, width)) {
          *out_ptr =
              __hfma(inp_ptr[iy_sw * inp_sH + ix_sw * inp_sW], sw, *out_ptr);
        }
        if (within_bounds_2d(iy_se, ix_se, height, width)) {
          *out_ptr =
              __hfma(inp_ptr[iy_se * inp_sH + ix_se * inp_sW], se, *out_ptr);
        }
      }
    } else if (interp == RotateInterpolation::Nearest) {
      int ix_nearest = static_cast<int>(hrint(ix));
      int iy_nearest = static_cast<int>(hrint(iy));

      // assign nearest neighbor pixel value to output pixel
      auto inp_ptr = input;
      auto out_ptr = output + h * inp_sH + w * inp_sW;
      for (int c = 0; c < channel; ++c, inp_ptr += inp_sC, out_ptr += inp_sC) {
        if (within_bounds_2d(iy_nearest, ix_nearest, height, width)) {
          *out_ptr = inp_ptr[iy_nearest * inp_sH + ix_nearest * inp_sW];
        } else {
          *out_ptr = static_cast<__half>(0);
        }
      }
    }
  }
}

__global__ void rotateKernel_h2(const int nthreads, __half2 *output,
                                __half2 *input, __half *angle, __half *center,
                                int channel, int height, int width,
                                RotateInterpolation interp) {
  int inp_sC = width * height;
  int inp_sH = width;
  int inp_sW = 1;

  const __half2 ang = __half2half2(__hmul(*angle, __float2half(-M_PI / 180.f)));
  const __half2 cx = __half2half2(__hsub(center[0], __float2half(0.5 * width))),
                cy =
                    __half2half2(__hsub(center[1], __float2half(0.5 * height)));
  const __half2 matrix[6] = {
      h2cos(ang),
      h2sin(ang),
      __hadd2(
          __hsub2(__hmul2(__hneg2(cx), h2cos(ang)), __hmul2(cy, h2sin(ang))),
          cx),
      -h2sin(ang),
      h2cos(ang),
      __hadd2(__hsub2(__hmul2(cx, h2sin(ang)), __hmul2(cy, h2cos(ang))), cy)};
  int width_h2 = (width + 1) / 2;
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int w = index % width_h2;
    const int h = (index / width_h2) % height;

    const bool last_one = width_h2 * 2 == width - 1;

    const __half x1 = __hadd(__hfma(__float2half(-0.5),
                                    __float2half(static_cast<float>(width)),
                                    __float2half(0.5)),
                             __float2half(static_cast<float>(w * 2)));
    const __half y1 = __hadd(__hfma(__float2half(-0.5),
                                    __float2half(static_cast<float>(height)),
                                    __float2half(0.5)),
                             __float2half(static_cast<float>(h)));

    const __half2 x = __halves2half2(
        x1, __hadd(x1, __float2half(1.f - static_cast<float>(last_one))));
    const __half2 y = __half2half2(y1);

    const __half2 grid_x = __h2div(
        __hadd2(__hmul2(matrix[0], x), __hfma2(matrix[1], y, matrix[2])),
        __hmul2(__float2half2_rn(0.5),
                __float2half2_rn(static_cast<float>(width))));
    const __half2 grid_y = __h2div(
        __hadd2(__hmul2(matrix[3], x), __hfma2(matrix[4], y, matrix[5])),
        __hmul2(__float2half2_rn(0.5),
                __float2half2_rn(static_cast<float>(height))));

    __half2 ix = grid_sampler_compute_source_index(grid_x, width);
    __half2 iy = grid_sampler_compute_source_index(grid_y, height);

    __half2 inp_H_h2 = __float2half2_rn(static_cast<float>(height));
    __half2 inp_W_h2 = __float2half2_rn(static_cast<float>(width));
    __half2 within_b, out_h2;
    __half inp_l, inp_h;

    if (interp == RotateInterpolation::Bilinear) {
      // get NE, NW, SE, SW pixel values from (x, y)
      __half2 ix_nw = h2floor(ix);
      __half2 iy_nw = h2floor(iy);
      __half2 ix_ne = __hadd2(ix_nw, __float2half2_rn(1.));
      __half2 iy_ne = iy_nw;
      __half2 ix_sw = ix_nw;
      __half2 iy_sw = __hadd2(iy_nw, __float2half2_rn(1.));
      __half2 ix_se = __hadd2(ix_nw, __float2half2_rn(1.));
      __half2 iy_se = __hadd2(iy_nw, __float2half2_rn(1.));

      // get surfaces to each neighbor:
      __half2 nw = __hmul2(__hsub2(ix_se, ix), __hsub2(iy_se, iy));
      __half2 ne = __hmul2(__hsub2(ix, ix_sw), __hsub2(iy_sw, iy));
      __half2 sw = __hmul2(__hsub2(ix_ne, ix), __hsub2(iy, iy_ne));
      __half2 se = __hmul2(__hsub2(ix, ix_nw), __hsub2(iy, iy_nw));

      // calculate bilinear weighted pixel value and set output pixel
      auto inp_ptr = (__half *)input;
      auto out_ptr = (__half *)output + h * inp_sH + w * inp_sW * 2;
      for (int c = 0; c < channel; ++c, inp_ptr += inp_sC, out_ptr += inp_sC) {
        out_h2 = __float2half2_rn(0.);
        within_b = within_bounds_2d_half2(iy_nw, ix_nw, inp_H_h2, inp_W_h2);
        inp_l = inp_ptr[(static_cast<int>(__low2float(iy_nw)) * inp_sH +
                         static_cast<int>(__low2float(ix_nw)) * inp_sW) *
                        static_cast<int>(__low2float(within_b))];
        inp_h = inp_ptr[(static_cast<int>(__high2float(iy_nw)) * inp_sH +
                         static_cast<int>(__high2float(ix_nw)) * inp_sW) *
                        static_cast<int>(__high2float(within_b))];
        out_h2 = __hfma2(__hmul2(__halves2half2(inp_l, inp_h), nw), within_b,
                         out_h2);

        within_b = within_bounds_2d_half2(iy_ne, ix_ne, inp_H_h2, inp_W_h2);
        inp_l = inp_ptr[(static_cast<int>(__low2float(iy_ne)) * inp_sH +
                         static_cast<int>(__low2float(ix_ne)) * inp_sW) *
                        static_cast<int>(__low2float(within_b))];
        inp_h = inp_ptr[(static_cast<int>(__high2float(iy_ne)) * inp_sH +
                         static_cast<int>(__high2float(ix_ne)) * inp_sW) *
                        static_cast<int>(__high2float(within_b))];
        out_h2 = __hfma2(__hmul2(__halves2half2(inp_l, inp_h), ne), within_b,
                         out_h2);

        within_b = within_bounds_2d_half2(iy_sw, ix_sw, inp_H_h2, inp_W_h2);
        inp_l = inp_ptr[(static_cast<int>(__low2float(iy_sw)) * inp_sH +
                         static_cast<int>(__low2float(ix_sw)) * inp_sW) *
                        static_cast<int>(__low2float(within_b))];
        inp_h = inp_ptr[(static_cast<int>(__high2float(iy_sw)) * inp_sH +
                         static_cast<int>(__high2float(ix_sw)) * inp_sW) *
                        static_cast<int>(__high2float(within_b))];
        out_h2 = __hfma2(__hmul2(__halves2half2(inp_l, inp_h), sw), within_b,
                         out_h2);

        within_b = within_bounds_2d_half2(iy_se, ix_se, inp_H_h2, inp_W_h2);
        inp_l = inp_ptr[(static_cast<int>(__low2float(iy_se)) * inp_sH +
                         static_cast<int>(__low2float(ix_se)) * inp_sW) *
                        static_cast<int>(__low2float(within_b))];
        inp_h = inp_ptr[(static_cast<int>(__high2float(iy_se)) * inp_sH +
                         static_cast<int>(__high2float(ix_se)) * inp_sW) *
                        static_cast<int>(__high2float(within_b))];
        out_h2 = __hfma2(__hmul2(__halves2half2(inp_l, inp_h), se), within_b,
                         out_h2);

        *out_ptr = __low2half(out_h2);
        if (!last_one) {
          *(out_ptr + 1) = __high2half(out_h2);
        }
      }
    } else if (interp == RotateInterpolation::Nearest) {
      __half2 ix_nearest = h2rint(ix);
      __half2 iy_nearest = h2rint(iy);

      // assign nearest neighbor pixel value to output pixel
      auto inp_ptr = (__half *)input;
      auto out_ptr = (__half *)output + h * inp_sH + w * inp_sW * 2;
      for (int c = 0; c < channel; ++c, inp_ptr += inp_sC, out_ptr += inp_sC) {
        out_h2 = __float2half2_rn(0.);
        within_b =
            within_bounds_2d_half2(iy_nearest, ix_nearest, inp_H_h2, inp_W_h2);
        inp_l = inp_ptr[(static_cast<int>(__low2float(iy_nearest)) * inp_sH +
                         static_cast<int>(__low2float(ix_nearest)) * inp_sW) *
                        static_cast<int>(__low2float(within_b))];
        inp_h = inp_ptr[(static_cast<int>(__high2float(iy_nearest)) * inp_sH +
                         static_cast<int>(__high2float(ix_nearest)) * inp_sW) *
                        static_cast<int>(__high2float(within_b))];
        out_h2 = __hmul2(__halves2half2(inp_l, inp_h), within_b);

        *out_ptr = __low2half(out_h2);
        if (!last_one) {
          *(out_ptr + 1) = __high2half(out_h2);
        }
      }
    }
  }
}

template <typename T>
void rotate(T *output, T *input, T *angle, T *center, int *input_dims,
            RotateInterpolation interp, cudaStream_t stream) {
  int channel = input_dims[0];
  int height = input_dims[1];
  int width = input_dims[2];

  int count = height * width;

  rotateKernel<<<GET_BLOCKS(count), THREADS_PER_BLOCK, 0, stream>>>(
      count, output, input, angle, center, channel, height, width, interp);
}

void rotate_h2(__half2 *output, __half2 *input, __half *angle, __half *center,
               int *input_dims, RotateInterpolation interp,
               cudaStream_t stream) {
  int channel = input_dims[0];
  int height = input_dims[1];
  int width = input_dims[2];

  int count = height * (width + 1) / 2;

  rotateKernel_h2<<<GET_BLOCKS(count), THREADS_PER_BLOCK, 0, stream>>>(
      count, output, input, angle, center, channel, height, width, interp);
}

template void rotate(float *output, float *input, float *angle, float *center,
                     int *input_dims, RotateInterpolation interp,
                     cudaStream_t stream);

template void rotate(__half *output, __half *input, __half *angle,
                     __half *center, int *input_dims,
                     RotateInterpolation interp, cudaStream_t stream);
