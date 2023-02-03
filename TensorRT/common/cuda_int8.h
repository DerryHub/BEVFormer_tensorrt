//
// Created by Derry Lin on 2023/2/1.
//

#ifndef TENSORRT_OPS_CUDA_INT8_H
#define TENSORRT_OPS_CUDA_INT8_H

#include <cuda_fp16.h>
#include "cuda_helper.h"

struct int8_4 {
     int8_t x=0, y=0, z=0, w=0;
    __device__ int8_4() {};
     __device__ int8_4(const int8_t &val) : x(val), y(val), z(val), w(val) {}
    __device__ int8_4(const int8_t &x, const int8_t &y, const int8_t &z, const int8_t &w) : x(x), y(y), z(z), w(w) {}
};

struct int32_4{
    int32_t x=0, y=0, z=0, w=0;
    __device__ int32_4() {};
    __device__ int32_4(const int32_t &val) : x(val), y(val), z(val), w(val) {}
};

struct float_4{
    float x=0, y=0, z=0, w=0;
    __device__ float_4() {};
    __device__ float_4(const float &val) : x(val), y(val), z(val), w(val) {}
};

typedef struct int8_4 int8_4;
typedef struct int32_4 int32_4;

#endif //TENSORRT_OPS_CUDA_INT8_H
