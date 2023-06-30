//
// Created by Derry Lin on 2023/6/21.
//

#include "multiHeadFlashAttnKernel.h"
#include "cuda_helper.h"
#include <cuda_fp16.h>
#include <mma.h>
#include <cmath>
#include <cuda/std/limits>

#define TC_SIZE 16
#define WARP_SIZE 32


template <typename T>
__forceinline__ __device__ T hmax(const T &a, const T &b) {
    return max(a, b);
}

#if __CUDA_ARCH__ >= 800
template <>
__forceinline__ __device__ __half hmax(const __half &a, const __half &b) {
    return __hmax(a, b);
}
template <>
__forceinline__ __device__ __half2 hmax(const __half2 &a, const __half2 &b) {
    return __hmax2(a, b);
}
#else
template <>
__forceinline__ __device__ __half hmax(const __half &a, const __half &b) {
    return __hgt(a, b) ? a : b;
}
template <>
__forceinline__ __device__ __half2 hmax(const __half2 &a, const __half2 &b) {
    return __hfma2(__hgt2(a, b), a, __hmul2(__hle2(a, b), b));
}
#endif

template <int HEAD_DIM, int BASE_SEQ_LEN>
__global__ void FMHAInferKernel(const float * __restrict__ query, const float * __restrict__ key, const float * __restrict__ value, float sqrt_d, float * output, const int KV_LEN) {
    static_assert(BASE_SEQ_LEN % TC_SIZE == 0 && HEAD_DIM % TC_SIZE == 0 && BASE_SEQ_LEN >= HEAD_DIM, "");
    const int NUM_WARPS = BASE_SEQ_LEN / TC_SIZE;
    const int NUM_HEAD_WARPS = HEAD_DIM / TC_SIZE;

    __shared__ __half share_buffer[BASE_SEQ_LEN*TC_SIZE*5];
    auto qk_buffer_f = reinterpret_cast<float*>(share_buffer + BASE_SEQ_LEN*TC_SIZE*2);
    auto qk_buffer_h = reinterpret_cast<__half2*>(share_buffer + BASE_SEQ_LEN*TC_SIZE*4);

    const unsigned int batch = blockIdx.y;
    const unsigned int Q_LEN = gridDim.x * BASE_SEQ_LEN;
    const unsigned int q_start = blockIdx.x * BASE_SEQ_LEN;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / WARP_SIZE;
    const unsigned int lane_id = tid % WARP_SIZE;
    const unsigned int warp_num = blockDim.x / WARP_SIZE;

    const unsigned int mem_lane_id_x = lane_id % 8;
    const unsigned int mem_lane_id_y = lane_id / 8;

    __half * smem_q[2];
    __half * smem_k[2];
    __half * smem_v[2];

    smem_q[0] = share_buffer;
    smem_q[1] = share_buffer + 1 * BASE_SEQ_LEN * TC_SIZE;
    smem_k[0] = share_buffer + 2 * BASE_SEQ_LEN * TC_SIZE;
    smem_k[1] = share_buffer + 3 * BASE_SEQ_LEN * TC_SIZE;
    smem_v[0] = share_buffer + 2 * BASE_SEQ_LEN * TC_SIZE;
    smem_v[1] = share_buffer + 3 * BASE_SEQ_LEN * TC_SIZE;

    const float *query_ptr = query + batch * Q_LEN * HEAD_DIM + q_start * HEAD_DIM;
    float *output_ptr = output + batch * Q_LEN * HEAD_DIM + q_start * HEAD_DIM;

    float thread_max_old[2] = {-cuda::std::numeric_limits<float>::infinity(),
                               -cuda::std::numeric_limits<float>::infinity()};
    float thread_sum_old[2] = {0, 0};

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, TC_SIZE, TC_SIZE, TC_SIZE, __half, nvcuda::wmma::row_major> q_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, TC_SIZE, TC_SIZE, TC_SIZE, __half, nvcuda::wmma::col_major> k_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, TC_SIZE, TC_SIZE, TC_SIZE, __half, nvcuda::wmma::row_major> qk_frag[NUM_WARPS];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, TC_SIZE, TC_SIZE, TC_SIZE, __half, nvcuda::wmma::row_major> v_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, TC_SIZE, TC_SIZE, TC_SIZE, float> kv_out_frag[
            NUM_WARPS > NUM_HEAD_WARPS ? NUM_WARPS : NUM_HEAD_WARPS];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, TC_SIZE, TC_SIZE, TC_SIZE, float> out_frag[NUM_HEAD_WARPS];

#pragma unroll
    for (int xi = 0; xi < NUM_HEAD_WARPS; xi++) {
        nvcuda::wmma::fill_fragment(out_frag[xi], 0.f);
    }

#pragma unroll
    for (int kv_start = 0; kv_start < KV_LEN; kv_start += BASE_SEQ_LEN) {
        const float *key_ptr = key + batch * KV_LEN * HEAD_DIM + kv_start * HEAD_DIM;

        float thread_max[2] = {-cuda::std::numeric_limits<float>::infinity(),
                               -cuda::std::numeric_limits<float>::infinity()};
        float thread_sum[2] = {0, 0};

#pragma unroll
        for (int xi = 0; xi < NUM_WARPS; xi++) {
            nvcuda::wmma::fill_fragment(kv_out_frag[xi], 0.f);
        }

        int k = 0, stride_warp = TC_SIZE * TC_SIZE / 2 / WARP_SIZE;
        float2 f2;
        __half2 h2;
        for (int i = 0; i < stride_warp; i++) {
            f2 = *reinterpret_cast<const float2 *>(query_ptr + warp_id * TC_SIZE * HEAD_DIM +
                                                   i * 2 * WARP_SIZE * HEAD_DIM / TC_SIZE +
                                                   mem_lane_id_y * HEAD_DIM + mem_lane_id_x * 2 + k * TC_SIZE);
            h2 = __float22half2_rn(f2);
            *(__half2 *) (smem_q[k % 2] + warp_id * TC_SIZE * TC_SIZE + i * WARP_SIZE * 2 +
                          mem_lane_id_y * TC_SIZE + mem_lane_id_x * 2) = h2;

            f2 = *reinterpret_cast<const float2 *>(key_ptr + warp_id * TC_SIZE * HEAD_DIM +
                                                   i * 2 * WARP_SIZE * HEAD_DIM / TC_SIZE +
                                                   mem_lane_id_y * HEAD_DIM + mem_lane_id_x * 2 + k * TC_SIZE);
            h2 = __float22half2_rn(f2);
            *(__half2 *) (smem_k[k % 2] + warp_id * TC_SIZE * TC_SIZE + i * WARP_SIZE * 2 +
                          mem_lane_id_y * TC_SIZE + mem_lane_id_x * 2) = h2;
        }

        for (k = 1; k < NUM_HEAD_WARPS; k++) {
            __syncthreads();
            for (int i = 0; i < stride_warp; i++) {
                f2 = *reinterpret_cast<const float2 *>(query_ptr + warp_id * TC_SIZE * HEAD_DIM +
                                                       i * 2 * WARP_SIZE * HEAD_DIM / TC_SIZE +
                                                       mem_lane_id_y * HEAD_DIM + mem_lane_id_x * 2 + k * TC_SIZE);
                h2 = __float22half2_rn(f2);
                *(__half2 *) (smem_q[k % 2] + warp_id * TC_SIZE * TC_SIZE + i * WARP_SIZE * 2 +
                              mem_lane_id_y * TC_SIZE + mem_lane_id_x * 2) = h2;

                f2 = *reinterpret_cast<const float2 *>(key_ptr + warp_id * TC_SIZE * HEAD_DIM +
                                                       i * 2 * WARP_SIZE * HEAD_DIM / TC_SIZE +
                                                       mem_lane_id_y * HEAD_DIM + mem_lane_id_x * 2 + k * TC_SIZE);
                h2 = __float22half2_rn(f2);
                *(__half2 *) (smem_k[k % 2] + warp_id * TC_SIZE * TC_SIZE + i * WARP_SIZE * 2 +
                              mem_lane_id_y * TC_SIZE + mem_lane_id_x * 2) = h2;
            }

            nvcuda::wmma::load_matrix_sync(q_frag, &(smem_q[(k - 1) % 2][warp_id * TC_SIZE * TC_SIZE]), TC_SIZE);
            for (int xi = 0; xi < NUM_WARPS; xi++) {
                nvcuda::wmma::load_matrix_sync(k_frag, &(smem_k[(k - 1) % 2][xi * TC_SIZE * TC_SIZE]), TC_SIZE);
                nvcuda::wmma::mma_sync(kv_out_frag[xi], q_frag, k_frag, kv_out_frag[xi]);
            }
        }
        __syncthreads();
        k = NUM_HEAD_WARPS - 1;
        nvcuda::wmma::load_matrix_sync(q_frag, &(smem_q[k % 2][warp_id * TC_SIZE * TC_SIZE]), TC_SIZE);
        for (int xi = 0; xi < NUM_WARPS; xi++) {
            nvcuda::wmma::load_matrix_sync(k_frag, &(smem_k[k % 2][xi * TC_SIZE * TC_SIZE]), TC_SIZE);
            nvcuda::wmma::mma_sync(kv_out_frag[xi], q_frag, k_frag, kv_out_frag[xi]);
        }

#pragma unroll
        for (int xi = 0; xi < NUM_WARPS; xi++) {
#pragma unroll
            for (int tc_yi = 0; tc_yi < 2; tc_yi++) {
#pragma unroll
                for (int tc_xi = 0; tc_xi < 2; tc_xi++) {
                    kv_out_frag[xi].x[tc_xi * 4 + tc_yi * 2 + 0] *= sqrt_d;
                    kv_out_frag[xi].x[tc_xi * 4 + tc_yi * 2 + 1] *= sqrt_d;
                    thread_max[tc_yi] = max(thread_max[tc_yi], max(kv_out_frag[xi].x[tc_xi * 4 + tc_yi * 2 + 0],
                                                                   kv_out_frag[xi].x[tc_xi * 4 + tc_yi * 2 + 1]));
                }
            }
        }

#pragma unroll
        for (int s = 2; s > 0; s >>= 1) {
#pragma unroll
            for (int tc_yi = 0; tc_yi < 2; tc_yi++) {
                thread_max[tc_yi] = max(thread_max[tc_yi], __shfl_xor_sync(0xffffffff, thread_max[tc_yi], s, 4));
            }
        }

#pragma unroll
        for (int xi = 0; xi < NUM_WARPS; xi++) {
#pragma unroll
            for (int tc_yi = 0; tc_yi < 2; tc_yi++) {
#pragma unroll
                for (int tc_xi = 0; tc_xi < 2; tc_xi++) {
                    kv_out_frag[xi].x[tc_xi * 4 + tc_yi * 2 + 0] = __expf(
                            kv_out_frag[xi].x[tc_xi * 4 + tc_yi * 2 + 0] - thread_max[tc_yi]);
                    kv_out_frag[xi].x[tc_xi * 4 + tc_yi * 2 + 1] = __expf(
                            kv_out_frag[xi].x[tc_xi * 4 + tc_yi * 2 + 1] - thread_max[tc_yi]);
                    thread_sum[tc_yi] += (kv_out_frag[xi].x[tc_xi * 4 + tc_yi * 2 + 0] +
                                          kv_out_frag[xi].x[tc_xi * 4 + tc_yi * 2 + 1]);
                }
            }
        }

#pragma unroll
        for (int s = 2; s > 0; s >>= 1) {
#pragma unroll
            for (int tc_yi = 0; tc_yi < 2; tc_yi++) {
                thread_sum[tc_yi] += __shfl_xor_sync(0xffffffff, thread_sum[tc_yi], s, 4);
            }
        }

        __syncthreads();
#pragma unroll
        for (int xi = 0; xi < NUM_WARPS; xi++) {
            nvcuda::wmma::store_matrix_sync(qk_buffer_f + warp_id * TC_SIZE * TC_SIZE, kv_out_frag[xi], TC_SIZE,
                                            nvcuda::wmma::mem_row_major);
            for (int i = 0; i < TC_SIZE * TC_SIZE / WARP_SIZE / 2; i++) {
                float t1 = qk_buffer_f[warp_id * TC_SIZE * TC_SIZE + i * WARP_SIZE * 2 + lane_id * 2];
                float t2 = qk_buffer_f[warp_id * TC_SIZE * TC_SIZE + i * WARP_SIZE * 2 + lane_id * 2 + 1];
                __half2 t3 = __floats2half2_rn(t1, t2);
                qk_buffer_h[warp_id * TC_SIZE * TC_SIZE / 2 + i * WARP_SIZE + lane_id] = t3;
            }
            nvcuda::wmma::load_matrix_sync(qk_frag[xi], reinterpret_cast<__half *>(qk_buffer_h +
                                                                                   warp_id * TC_SIZE * TC_SIZE / 2),
                                           TC_SIZE);
        }
        __syncthreads();

#pragma unroll
        for (int xi = 0; xi < NUM_HEAD_WARPS; xi++) {
            nvcuda::wmma::fill_fragment(kv_out_frag[xi], 0.f);
        }

        stride_warp = TC_SIZE / warp_num;
        const float *value_ptr = value + batch * KV_LEN * HEAD_DIM + (kv_start + stride_warp * warp_id) * HEAD_DIM;
        k = 0;
        for (int i = 0; i < stride_warp * HEAD_DIM / 2 / WARP_SIZE; i++) {
            f2 = *reinterpret_cast<const float2 *>(value_ptr + k * TC_SIZE * HEAD_DIM + i * WARP_SIZE * 2 +
                                                   lane_id * 2);
            h2 = __float22half2_rn(f2);
            *(__half2 *) (smem_v[k % 2] + warp_id * stride_warp * HEAD_DIM + i * WARP_SIZE * 2 + lane_id * 2) = h2;
        }

        for (k = 1; k < NUM_WARPS; k++) {
            __syncthreads();
            for (int i = 0; i < stride_warp * HEAD_DIM / 2 / WARP_SIZE; i++) {
                f2 = *reinterpret_cast<const float2 *>(value_ptr + k * TC_SIZE * HEAD_DIM + i * WARP_SIZE * 2 +
                                                       lane_id * 2);
                h2 = __float22half2_rn(f2);
                *(__half2 *) (smem_v[k % 2] + warp_id * stride_warp * HEAD_DIM + i * WARP_SIZE * 2 +
                              lane_id * 2) = h2;
            }

            for (int xi = 0; xi < NUM_HEAD_WARPS; xi++) {
                nvcuda::wmma::load_matrix_sync(v_frag, &(smem_v[(k - 1) % 2][xi * TC_SIZE]), HEAD_DIM);
                nvcuda::wmma::mma_sync(kv_out_frag[xi], qk_frag[k - 1], v_frag, kv_out_frag[xi]);
            }
        }
        __syncthreads();
        k = NUM_WARPS - 1;
        for (int xi = 0; xi < NUM_HEAD_WARPS; xi++) {
            nvcuda::wmma::load_matrix_sync(v_frag, &(smem_v[k % 2][xi * TC_SIZE]), HEAD_DIM);
            nvcuda::wmma::mma_sync(kv_out_frag[xi], qk_frag[k], v_frag, kv_out_frag[xi]);
        }

#pragma unroll
        for (int tc_yi = 0; tc_yi < 2; tc_yi++) {
            float thread_max_new = max(thread_max_old[tc_yi], thread_max[tc_yi]);
            float exp_max_old = __expf(thread_max_old[tc_yi] - thread_max_new);
            float exp_max = __expf(thread_max[tc_yi] - thread_max_new);
            float thread_sum_new = exp_max_old * thread_sum_old[tc_yi] + exp_max * thread_sum[tc_yi];
#pragma unroll
            for (int xi = 0; xi < NUM_HEAD_WARPS; xi++) {
#pragma unroll
                for (int tc_xi = 0; tc_xi < 2; tc_xi++) {
                    out_frag[xi].x[tc_xi * 4 + tc_yi * 2 + 0] = __frcp_rn(thread_sum_new) *
                                                                (thread_sum_old[tc_yi] * exp_max_old *
                                                                 out_frag[xi].x[tc_xi * 4 + tc_yi * 2 + 0] +
                                                                 exp_max *
                                                                 (kv_out_frag[xi].x[tc_xi * 4 + tc_yi * 2 + 0]));
                    out_frag[xi].x[tc_xi * 4 + tc_yi * 2 + 1] = __frcp_rn(thread_sum_new) *
                                                                (thread_sum_old[tc_yi] * exp_max_old *
                                                                 out_frag[xi].x[tc_xi * 4 + tc_yi * 2 + 1] +
                                                                 exp_max *
                                                                 (kv_out_frag[xi].x[tc_xi * 4 + tc_yi * 2 + 1]));
                }
            }
            thread_sum_old[tc_yi] = thread_sum_new;
            thread_max_old[tc_yi] = thread_max_new;
        }
    }

    for (int xi = 0; xi < NUM_HEAD_WARPS; xi++) {
        nvcuda::wmma::store_matrix_sync(output_ptr + warp_id * TC_SIZE * HEAD_DIM + xi * TC_SIZE, out_frag[xi],
                                        HEAD_DIM, nvcuda::wmma::mem_row_major);
    }
}




template <int HEAD_DIM, int BASE_SEQ_LEN>
__global__ void FMHAInferKernel(const __half * __restrict__ query, const __half * __restrict__ key, const __half * __restrict__ value, __half sqrt_d, __half * output, const int KV_LEN) {
    static_assert(BASE_SEQ_LEN % TC_SIZE == 0 && HEAD_DIM % TC_SIZE == 0 && BASE_SEQ_LEN >= HEAD_DIM, "");
    const int NUM_WARPS = BASE_SEQ_LEN / TC_SIZE;
    const int NUM_HEAD_WARPS = HEAD_DIM / TC_SIZE;

    __shared__ __half share_buffer[BASE_SEQ_LEN*TC_SIZE*4];
    auto qk_buffer_fh = share_buffer + BASE_SEQ_LEN*TC_SIZE*2;

    const unsigned int batch = blockIdx.y;
    const unsigned int Q_LEN = gridDim.x * BASE_SEQ_LEN;
    const unsigned int q_start = blockIdx.x * BASE_SEQ_LEN;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / WARP_SIZE;
    const unsigned int lane_id = tid % WARP_SIZE;
    const unsigned int warp_num = blockDim.x / WARP_SIZE;

    const unsigned int mem_lane_id_x = lane_id % 8;
    const unsigned int mem_lane_id_y = lane_id / 8;

    __half * smem_q[2];
    __half * smem_k[2];
    __half * smem_v[2];

    smem_q[0] = share_buffer;
    smem_q[1] = share_buffer + 1 * BASE_SEQ_LEN * TC_SIZE;
    smem_k[0] = share_buffer + 2 * BASE_SEQ_LEN * TC_SIZE;
    smem_k[1] = share_buffer + 3 * BASE_SEQ_LEN * TC_SIZE;
    smem_v[0] = share_buffer + 2 * BASE_SEQ_LEN * TC_SIZE;
    smem_v[1] = share_buffer + 3 * BASE_SEQ_LEN * TC_SIZE;

    const __half *query_ptr = query + batch * Q_LEN * HEAD_DIM + q_start * HEAD_DIM;
    __half * output_ptr = output + batch * Q_LEN * HEAD_DIM + q_start * HEAD_DIM;

    __half thread_max_old[2] = {-cuda::std::numeric_limits<__half>::infinity(),
                                -cuda::std::numeric_limits<__half>::infinity()};
    __half thread_sum_old[2] = {0, 0};

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, TC_SIZE, TC_SIZE, TC_SIZE, __half, nvcuda::wmma::row_major> q_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, TC_SIZE, TC_SIZE, TC_SIZE, __half, nvcuda::wmma::col_major> k_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, TC_SIZE, TC_SIZE, TC_SIZE, __half, nvcuda::wmma::row_major> qk_frag[NUM_WARPS];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, TC_SIZE, TC_SIZE, TC_SIZE, __half, nvcuda::wmma::row_major> v_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, TC_SIZE, TC_SIZE, TC_SIZE, __half> kv_out_frag[
            NUM_WARPS > NUM_HEAD_WARPS ? NUM_WARPS : NUM_HEAD_WARPS];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, TC_SIZE, TC_SIZE, TC_SIZE, __half> out_frag[NUM_HEAD_WARPS];

#pragma unroll
    for (int xi = 0; xi < NUM_HEAD_WARPS; xi++) {
        nvcuda::wmma::fill_fragment(out_frag[xi], 0.f);
    }

#pragma unroll
    for (int kv_start = 0; kv_start < KV_LEN; kv_start += BASE_SEQ_LEN) {
        const __half *key_ptr = key + batch * KV_LEN * HEAD_DIM + kv_start * HEAD_DIM;

        __half thread_max[2] = {-cuda::std::numeric_limits<__half>::infinity(),
                                -cuda::std::numeric_limits<__half>::infinity()};
        __half thread_sum[2] = {0, 0};

#pragma unroll
        for (int xi = 0; xi < NUM_WARPS; xi++) {
            nvcuda::wmma::fill_fragment(kv_out_frag[xi], 0.f);
        }

        int k = 0, stride_warp = TC_SIZE * TC_SIZE / 2 / WARP_SIZE;
        __half2 h2;
        for (int i = 0; i < stride_warp; i++) {
            h2 = *reinterpret_cast<const __half2 *>(query_ptr + warp_id * TC_SIZE * HEAD_DIM +
                                                    i * 2 * WARP_SIZE * HEAD_DIM / TC_SIZE +
                                                    mem_lane_id_y * HEAD_DIM + mem_lane_id_x * 2 + k * TC_SIZE);
            *(__half2 *) (smem_q[k % 2] + warp_id * TC_SIZE * TC_SIZE + i * WARP_SIZE * 2 +
                          mem_lane_id_y * TC_SIZE + mem_lane_id_x * 2) = h2;

            h2 = *reinterpret_cast<const __half2 *>(key_ptr + warp_id * TC_SIZE * HEAD_DIM +
                                                    i * 2 * WARP_SIZE * HEAD_DIM / TC_SIZE +
                                                    mem_lane_id_y * HEAD_DIM + mem_lane_id_x * 2 + k * TC_SIZE);
            *(__half2 *) (smem_k[k % 2] + warp_id * TC_SIZE * TC_SIZE + i * WARP_SIZE * 2 +
                          mem_lane_id_y * TC_SIZE + mem_lane_id_x * 2) = h2;
        }

        for (k = 1; k < NUM_HEAD_WARPS; k++) {
            __syncthreads();
            for (int i = 0; i < stride_warp; i++) {
                h2 = *reinterpret_cast<const __half2 *>(query_ptr + warp_id * TC_SIZE * HEAD_DIM +
                                                        i * 2 * WARP_SIZE * HEAD_DIM / TC_SIZE +
                                                        mem_lane_id_y * HEAD_DIM + mem_lane_id_x * 2 + k * TC_SIZE);
                *(__half2 *) (smem_q[k % 2] + warp_id * TC_SIZE * TC_SIZE + i * WARP_SIZE * 2 +
                              mem_lane_id_y * TC_SIZE + mem_lane_id_x * 2) = h2;

                h2 = *reinterpret_cast<const __half2 *>(key_ptr + warp_id * TC_SIZE * HEAD_DIM +
                                                        i * 2 * WARP_SIZE * HEAD_DIM / TC_SIZE +
                                                        mem_lane_id_y * HEAD_DIM + mem_lane_id_x * 2 + k * TC_SIZE);
                *(__half2 *) (smem_k[k % 2] + warp_id * TC_SIZE * TC_SIZE + i * WARP_SIZE * 2 +
                              mem_lane_id_y * TC_SIZE + mem_lane_id_x * 2) = h2;
            }

            nvcuda::wmma::load_matrix_sync(q_frag, &(smem_q[(k - 1) % 2][warp_id * TC_SIZE * TC_SIZE]), TC_SIZE);
            for (int xi = 0; xi < NUM_WARPS; xi++) {
                nvcuda::wmma::load_matrix_sync(k_frag, &(smem_k[(k - 1) % 2][xi * TC_SIZE * TC_SIZE]), TC_SIZE);
                nvcuda::wmma::mma_sync(kv_out_frag[xi], q_frag, k_frag, kv_out_frag[xi]);
            }
        }
        __syncthreads();
        k = NUM_HEAD_WARPS - 1;
        nvcuda::wmma::load_matrix_sync(q_frag, &(smem_q[k % 2][warp_id * TC_SIZE * TC_SIZE]), TC_SIZE);
        for (int xi = 0; xi < NUM_WARPS; xi++) {
            nvcuda::wmma::load_matrix_sync(k_frag, &(smem_k[k % 2][xi * TC_SIZE * TC_SIZE]), TC_SIZE);
            nvcuda::wmma::mma_sync(kv_out_frag[xi], q_frag, k_frag, kv_out_frag[xi]);
        }

#pragma unroll
        for (int xi = 0; xi < NUM_WARPS; xi++) {
#pragma unroll
            for (int tc_yi = 0; tc_yi < 2; tc_yi++) {
#pragma unroll
                for (int tc_xi = 0; tc_xi < 2; tc_xi++) {
                    kv_out_frag[xi].x[tc_xi * 4 + tc_yi * 2 + 0] = __hmul(
                            kv_out_frag[xi].x[tc_xi * 4 + tc_yi * 2 + 0], sqrt_d);
                    kv_out_frag[xi].x[tc_xi * 4 + tc_yi * 2 + 1] = __hmul(
                            kv_out_frag[xi].x[tc_xi * 4 + tc_yi * 2 + 1], sqrt_d);
                    thread_max[tc_yi] = hmax(thread_max[tc_yi], hmax(kv_out_frag[xi].x[tc_xi * 4 + tc_yi * 2 + 0],
                                                                     kv_out_frag[xi].x[tc_xi * 4 + tc_yi * 2 + 1]));
                }
            }
        }

#pragma unroll
        for (int s = 2; s > 0; s >>= 1) {
#pragma unroll
            for (int tc_yi = 0; tc_yi < 2; tc_yi++) {
                thread_max[tc_yi] = hmax(thread_max[tc_yi], __shfl_xor_sync(0xffffffff, thread_max[tc_yi], s, 4));
            }
        }

#pragma unroll
        for (int xi = 0; xi < NUM_WARPS; xi++) {
#pragma unroll
            for (int tc_yi = 0; tc_yi < 2; tc_yi++) {
#pragma unroll
                for (int tc_xi = 0; tc_xi < 2; tc_xi++) {
                    kv_out_frag[xi].x[tc_xi * 4 + tc_yi * 2 + 0] = hexp(
                            kv_out_frag[xi].x[tc_xi * 4 + tc_yi * 2 + 0] - thread_max[tc_yi]);
                    kv_out_frag[xi].x[tc_xi * 4 + tc_yi * 2 + 1] = hexp(
                            kv_out_frag[xi].x[tc_xi * 4 + tc_yi * 2 + 1] - thread_max[tc_yi]);
                    thread_sum[tc_yi] += (kv_out_frag[xi].x[tc_xi * 4 + tc_yi * 2 + 0] +
                                          kv_out_frag[xi].x[tc_xi * 4 + tc_yi * 2 + 1]);
                }
            }
        }

#pragma unroll
        for (int s = 2; s > 0; s >>= 1) {
#pragma unroll
            for (int tc_yi = 0; tc_yi < 2; tc_yi++) {
                thread_sum[tc_yi] += __shfl_xor_sync(0xffffffff, thread_sum[tc_yi], s, 4);
            }
        }

        __syncthreads();
#pragma unroll
        for (int xi = 0; xi < NUM_WARPS; xi++) {
            nvcuda::wmma::store_matrix_sync(qk_buffer_fh + warp_id * TC_SIZE * TC_SIZE, kv_out_frag[xi], TC_SIZE,
                                            nvcuda::wmma::mem_row_major);
            nvcuda::wmma::load_matrix_sync(qk_frag[xi], qk_buffer_fh + warp_id * TC_SIZE * TC_SIZE, TC_SIZE);
        }
        __syncthreads();

#pragma unroll
        for (int xi = 0; xi < NUM_HEAD_WARPS; xi++) {
            nvcuda::wmma::fill_fragment(kv_out_frag[xi], 0.f);
        }

        stride_warp = TC_SIZE / warp_num;
        const __half *value_ptr = value + batch * KV_LEN * HEAD_DIM + (kv_start + stride_warp * warp_id) * HEAD_DIM;
        k = 0;
        for (int i = 0; i < stride_warp * HEAD_DIM / 2 / WARP_SIZE; i++) {
            h2 = *reinterpret_cast<const __half2 *>(value_ptr + k * TC_SIZE * HEAD_DIM + i * WARP_SIZE * 2 +
                                                    lane_id * 2);
            *(__half2 *) (smem_v[k % 2] + warp_id * stride_warp * HEAD_DIM + i * WARP_SIZE * 2 + lane_id * 2) = h2;
        }

        for (k = 1; k < NUM_WARPS; k++) {
            __syncthreads();
            for (int i = 0; i < stride_warp * HEAD_DIM / 2 / WARP_SIZE; i++) {
                h2 = *reinterpret_cast<const __half2 *>(value_ptr + k * TC_SIZE * HEAD_DIM + i * WARP_SIZE * 2 +
                                                        lane_id * 2);
                *(__half2 *) (smem_v[k % 2] + warp_id * stride_warp * HEAD_DIM + i * WARP_SIZE * 2 +
                              lane_id * 2) = h2;
            }

            for (int xi = 0; xi < NUM_HEAD_WARPS; xi++) {
                nvcuda::wmma::load_matrix_sync(v_frag, &(smem_v[(k - 1) % 2][xi * TC_SIZE]), HEAD_DIM);
                nvcuda::wmma::mma_sync(kv_out_frag[xi], qk_frag[k - 1], v_frag, kv_out_frag[xi]);
            }
        }
        __syncthreads();
        k = NUM_WARPS - 1;
        for (int xi = 0; xi < NUM_HEAD_WARPS; xi++) {
            nvcuda::wmma::load_matrix_sync(v_frag, &(smem_v[k % 2][xi * TC_SIZE]), HEAD_DIM);
            nvcuda::wmma::mma_sync(kv_out_frag[xi], qk_frag[k], v_frag, kv_out_frag[xi]);
        }

#pragma unroll
        for (int tc_yi = 0; tc_yi < 2; tc_yi++) {
            __half thread_max_new = hmax(thread_max_old[tc_yi], thread_max[tc_yi]);
            __half exp_max_old = hexp(thread_max_old[tc_yi] - thread_max_new);
            __half exp_max = hexp(thread_max[tc_yi] - thread_max_new);
            __half thread_sum_new = exp_max_old * thread_sum_old[tc_yi] + exp_max * thread_sum[tc_yi];
#pragma unroll
            for (int xi = 0; xi < NUM_HEAD_WARPS; xi++) {
#pragma unroll
                for (int tc_xi = 0; tc_xi < 2; tc_xi++) {
                    out_frag[xi].x[tc_xi * 4 + tc_yi * 2 + 0] = __hdiv(
                            thread_sum_old[tc_yi] * exp_max_old * out_frag[xi].x[tc_xi * 4 + tc_yi * 2 + 0] +
                            exp_max * (kv_out_frag[xi].x[tc_xi * 4 + tc_yi * 2 + 0]), thread_sum_new);
                    out_frag[xi].x[tc_xi * 4 + tc_yi * 2 + 1] = __hdiv(
                            thread_sum_old[tc_yi] * exp_max_old * out_frag[xi].x[tc_xi * 4 + tc_yi * 2 + 1] +
                            exp_max * (kv_out_frag[xi].x[tc_xi * 4 + tc_yi * 2 + 1]), thread_sum_new);
                }
            }
            thread_sum_old[tc_yi] = thread_sum_new;
            thread_max_old[tc_yi] = thread_max_new;
        }
    }

    for (int xi = 0; xi < NUM_HEAD_WARPS; xi++) {
        nvcuda::wmma::store_matrix_sync(output_ptr + warp_id * TC_SIZE * HEAD_DIM + xi * TC_SIZE, out_frag[xi],
                                        HEAD_DIM, nvcuda::wmma::mem_row_major);
    }
}

template <int HEAD_DIM, int BASE_SEQ_LEN>
__global__ void FMHAInferKernel(const __half2 * __restrict__ query, const __half2 * __restrict__ key, const __half2 * __restrict__ value, __half2 sqrt_d, __half2 * output, const int KV_LEN) {
    static_assert(BASE_SEQ_LEN % TC_SIZE == 0 && HEAD_DIM % TC_SIZE == 0 && BASE_SEQ_LEN >= HEAD_DIM, "");
    const int NUM_WARPS = BASE_SEQ_LEN / TC_SIZE;
    const int NUM_HEAD_WARPS = HEAD_DIM / TC_SIZE;

    __shared__ __half2 share_buffer[BASE_SEQ_LEN*TC_SIZE*2];
    auto qk_buffer_fh = reinterpret_cast<__half*>(share_buffer + BASE_SEQ_LEN*TC_SIZE);

    const unsigned int batch = blockIdx.y;
    const unsigned int Q_LEN = gridDim.x * BASE_SEQ_LEN;
    const unsigned int q_start = blockIdx.x * BASE_SEQ_LEN;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / WARP_SIZE;
    const unsigned int lane_id = tid % WARP_SIZE;
    const unsigned int warp_num = blockDim.x / WARP_SIZE;

    const unsigned int mem_lane_id_x = lane_id % 8;
    const unsigned int mem_lane_id_y = lane_id / 8;

    __half2 * smem_q[2];
    __half2 * smem_k[2];
    __half2 * smem_v[2];

    smem_q[0] = share_buffer;
    smem_q[1] = share_buffer + 1 * BASE_SEQ_LEN * TC_SIZE / 2;
    smem_k[0] = share_buffer + 2 * BASE_SEQ_LEN * TC_SIZE / 2;
    smem_k[1] = share_buffer + 3 * BASE_SEQ_LEN * TC_SIZE / 2;
    smem_v[0] = share_buffer + 2 * BASE_SEQ_LEN * TC_SIZE / 2;
    smem_v[1] = share_buffer + 3 * BASE_SEQ_LEN * TC_SIZE / 2;

    const __half2* query_ptr = query + batch * Q_LEN * HEAD_DIM / 2 + q_start * HEAD_DIM / 2;
    __half2 * output_ptr = output + batch * Q_LEN * HEAD_DIM / 2 + q_start * HEAD_DIM / 2;

    __half2 thread_max_old = __half2half2(-cuda::std::numeric_limits<__half>::infinity());
    __half2 thread_sum_old = __float2half2_rn(0.f);

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, TC_SIZE, TC_SIZE, TC_SIZE, __half, nvcuda::wmma::row_major> q_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, TC_SIZE, TC_SIZE, TC_SIZE, __half, nvcuda::wmma::col_major> k_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, TC_SIZE, TC_SIZE, TC_SIZE, __half, nvcuda::wmma::row_major> qk_frag[NUM_WARPS];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, TC_SIZE, TC_SIZE, TC_SIZE, __half, nvcuda::wmma::row_major> v_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, TC_SIZE, TC_SIZE, TC_SIZE, __half> kv_out_frag[NUM_WARPS > NUM_HEAD_WARPS ? NUM_WARPS : NUM_HEAD_WARPS];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, TC_SIZE, TC_SIZE, TC_SIZE, __half> out_frag[NUM_HEAD_WARPS];

#pragma unroll
    for(int xi=0; xi<NUM_HEAD_WARPS; xi++){
        nvcuda::wmma::fill_fragment(out_frag[xi], 0.f);
    }

#pragma unroll
    for (int kv_start=0; kv_start < KV_LEN; kv_start+=BASE_SEQ_LEN) {
        const __half2* key_ptr = key + batch * KV_LEN * HEAD_DIM / 2 + kv_start * HEAD_DIM / 2;

        __half2 thread_max = __half2half2(-cuda::std::numeric_limits<__half>::infinity());
        __half2 thread_sum = __float2half2_rn(0.f);

#pragma unroll
        for(int xi=0; xi<NUM_WARPS; xi++){
            nvcuda::wmma::fill_fragment(kv_out_frag[xi], 0.f);
        }

        int k = 0, stride_warp = TC_SIZE*TC_SIZE/2/WARP_SIZE;
        __half2 h2;
        for (int i=0; i<stride_warp; i++) {
            h2 = *(query_ptr + warp_id * TC_SIZE * HEAD_DIM / 2 + i * WARP_SIZE * HEAD_DIM / TC_SIZE + mem_lane_id_y * HEAD_DIM / 2 + mem_lane_id_x + k * TC_SIZE / 2);
            *(smem_q[k%2] + warp_id * TC_SIZE * TC_SIZE / 2 + i * WARP_SIZE + mem_lane_id_y * TC_SIZE / 2 + mem_lane_id_x) = h2;

            h2 = *(key_ptr + warp_id * TC_SIZE * HEAD_DIM / 2 + i * WARP_SIZE * HEAD_DIM / TC_SIZE + mem_lane_id_y * HEAD_DIM / 2 + mem_lane_id_x + k * TC_SIZE / 2);
            *(smem_k[k%2] + warp_id * TC_SIZE * TC_SIZE / 2 + i * WARP_SIZE + mem_lane_id_y * TC_SIZE / 2 + mem_lane_id_x) = h2;
        }

        for (k=1; k < NUM_HEAD_WARPS; k++) {
            __syncthreads();
            for (int i=0; i<stride_warp; i++) {
                h2 = *(query_ptr + warp_id * TC_SIZE * HEAD_DIM / 2 + i * WARP_SIZE * HEAD_DIM / TC_SIZE + mem_lane_id_y * HEAD_DIM / 2 + mem_lane_id_x + k * TC_SIZE / 2);
                *(smem_q[k%2] + warp_id * TC_SIZE * TC_SIZE / 2 + i * WARP_SIZE + mem_lane_id_y * TC_SIZE / 2 + mem_lane_id_x) = h2;

                h2 = *(key_ptr + warp_id * TC_SIZE * HEAD_DIM / 2 + i * WARP_SIZE * HEAD_DIM / TC_SIZE + mem_lane_id_y * HEAD_DIM / 2 + mem_lane_id_x + k * TC_SIZE / 2);
                *(smem_k[k%2] + warp_id * TC_SIZE * TC_SIZE / 2 + i * WARP_SIZE + mem_lane_id_y * TC_SIZE / 2 + mem_lane_id_x) = h2;
            }

            nvcuda::wmma::load_matrix_sync(q_frag, reinterpret_cast<__half*>(&(smem_q[(k-1)%2][warp_id*TC_SIZE*TC_SIZE/2])), TC_SIZE);
            for (int xi=0; xi < NUM_WARPS; xi++) {
                nvcuda::wmma::load_matrix_sync(k_frag, reinterpret_cast<__half*>(&(smem_k[(k-1)%2][xi*TC_SIZE*TC_SIZE/2])), TC_SIZE);
                nvcuda::wmma::mma_sync(kv_out_frag[xi], q_frag, k_frag, kv_out_frag[xi]);
            }
        }
        __syncthreads();
        k = NUM_HEAD_WARPS - 1;
        nvcuda::wmma::load_matrix_sync(q_frag, reinterpret_cast<__half*>(&(smem_q[k%2][warp_id*TC_SIZE*TC_SIZE/2])), TC_SIZE);
        for (int xi=0; xi < NUM_WARPS; xi++) {
            nvcuda::wmma::load_matrix_sync(k_frag, reinterpret_cast<__half*>(&(smem_k[k%2][xi*TC_SIZE*TC_SIZE/2])), TC_SIZE);
            nvcuda::wmma::mma_sync(kv_out_frag[xi], q_frag, k_frag, kv_out_frag[xi]);
        }

#pragma unroll
        for(int xi=0; xi<NUM_WARPS; xi++){
#pragma unroll
            for(int tc_xi=0; tc_xi<2; tc_xi++){
                __half2 temp1 = __halves2half2(kv_out_frag[xi].x[tc_xi*4], kv_out_frag[xi].x[tc_xi*4+2]);
                __half2 temp2 = __halves2half2(kv_out_frag[xi].x[tc_xi*4+1], kv_out_frag[xi].x[tc_xi*4+3]);
                temp1 = __hmul2(temp1, sqrt_d);
                temp2 = __hmul2(temp2, sqrt_d);

                kv_out_frag[xi].x[tc_xi*4] = __low2half(temp1);
                kv_out_frag[xi].x[tc_xi*4+2] = __high2half(temp1);
                kv_out_frag[xi].x[tc_xi*4+1] = __low2half(temp2);
                kv_out_frag[xi].x[tc_xi*4+3] = __high2half(temp2);
                thread_max = hmax(thread_max, hmax(temp1, temp2));
            }
        }

#pragma unroll
        for (int s = 2; s > 0; s >>= 1){
            thread_max = hmax(thread_max, __shfl_xor_sync(0xffffffff, thread_max, s, 4));
        }

#pragma unroll
        for(int xi=0; xi<NUM_WARPS; xi++){
#pragma unroll
            for(int tc_xi=0; tc_xi<2; tc_xi++){
                __half2 temp1 = __halves2half2(kv_out_frag[xi].x[tc_xi*4], kv_out_frag[xi].x[tc_xi*4+2]);
                __half2 temp2 = __halves2half2(kv_out_frag[xi].x[tc_xi*4+1], kv_out_frag[xi].x[tc_xi*4+3]);
                temp1 = h2exp(__hsub2(temp1, thread_max));
                temp2 = h2exp(__hsub2(temp2, thread_max));
                kv_out_frag[xi].x[tc_xi*4] = __low2half(temp1);
                kv_out_frag[xi].x[tc_xi*4+2] = __high2half(temp1);
                kv_out_frag[xi].x[tc_xi*4+1] = __low2half(temp2);
                kv_out_frag[xi].x[tc_xi*4+3] = __high2half(temp2);
                thread_sum = __hadd2(thread_sum, __hadd2(temp1, temp2));
            }
        }

#pragma unroll
        for (int s = 2; s > 0; s >>= 1){
            thread_sum = __hadd2(thread_sum, __shfl_xor_sync(0xffffffff, thread_sum, s, 4));
        }

        __syncthreads();
#pragma unroll
        for(int xi=0; xi<NUM_WARPS; xi++){
            nvcuda::wmma::store_matrix_sync(qk_buffer_fh + warp_id * TC_SIZE*TC_SIZE, kv_out_frag[xi], TC_SIZE, nvcuda::wmma::mem_row_major);
            nvcuda::wmma::load_matrix_sync(qk_frag[xi], qk_buffer_fh+warp_id*TC_SIZE*TC_SIZE, TC_SIZE);
        }
        __syncthreads();

#pragma unroll
        for(int xi=0; xi<NUM_HEAD_WARPS; xi++){
            nvcuda::wmma::fill_fragment(kv_out_frag[xi], 0.f);
        }

        stride_warp = TC_SIZE / warp_num;
        const __half2* value_ptr = value + batch * KV_LEN * HEAD_DIM / 2 + (kv_start + stride_warp * warp_id) * HEAD_DIM / 2;
        k = 0;
        for (int i=0; i < stride_warp * HEAD_DIM / 2 / WARP_SIZE; i++) {
            h2 = *(value_ptr + k * TC_SIZE * HEAD_DIM / 2 + i * WARP_SIZE + lane_id);
            *(smem_v[k%2] + warp_id * stride_warp * HEAD_DIM / 2 + i * WARP_SIZE + lane_id) = h2;
        }

        for (k=1; k < NUM_WARPS; k++) {
            __syncthreads();
            for (int i=0; i < stride_warp * HEAD_DIM / 2 / WARP_SIZE; i++) {
                h2 = *(value_ptr + k * TC_SIZE * HEAD_DIM / 2 + i * WARP_SIZE + lane_id);
                *(smem_v[k%2] + warp_id * stride_warp * HEAD_DIM / 2 + i * WARP_SIZE + lane_id) = h2;
            }

            for (int xi=0; xi < NUM_HEAD_WARPS; xi++) {
                nvcuda::wmma::load_matrix_sync(v_frag, reinterpret_cast<__half*>(&(smem_v[(k-1)%2][xi*TC_SIZE/2])), HEAD_DIM);
                nvcuda::wmma::mma_sync(kv_out_frag[xi], qk_frag[k-1], v_frag, kv_out_frag[xi]);
            }
        }
        __syncthreads();
        k = NUM_WARPS - 1;
        for (int xi=0; xi < NUM_HEAD_WARPS; xi++) {
            nvcuda::wmma::load_matrix_sync(v_frag, reinterpret_cast<__half*>(&(smem_v[k%2][xi*TC_SIZE/2])), HEAD_DIM);
            nvcuda::wmma::mma_sync(kv_out_frag[xi], qk_frag[k], v_frag, kv_out_frag[xi]);
        }

        __half2 thread_max_new = hmax(thread_max_old, thread_max);
        __half2 exp_max_old = h2exp(__hsub2(thread_max_old, thread_max_new));
        __half2 exp_max = h2exp(__hsub2(thread_max, thread_max_new));
        __half2 thread_sum_new = __hadd2(__hmul2(exp_max_old, thread_sum_old), __hmul2(exp_max, thread_sum));
#pragma unroll
        for(int xi=0; xi<NUM_HEAD_WARPS; xi++){
#pragma unroll
            for(int tc_xi=0; tc_xi<2; tc_xi++){
                __half2 temp1 = __halves2half2(out_frag[xi].x[tc_xi*4], out_frag[xi].x[tc_xi*4+2]);
                __half2 temp2 = __halves2half2(out_frag[xi].x[tc_xi*4+1], out_frag[xi].x[tc_xi*4+3]);
                __half2 temp1_kv = __halves2half2(kv_out_frag[xi].x[tc_xi*4], kv_out_frag[xi].x[tc_xi*4+2]);
                __half2 temp2_kv = __halves2half2(kv_out_frag[xi].x[tc_xi*4+1], kv_out_frag[xi].x[tc_xi*4+3]);

                temp1 = __h2div(__hfma2(exp_max, temp1_kv, __hmul2(__hmul2(thread_sum_old, exp_max_old), temp1)), thread_sum_new);
                temp2 = __h2div(__hfma2(exp_max, temp2_kv, __hmul2(__hmul2(thread_sum_old, exp_max_old), temp2)), thread_sum_new);

                out_frag[xi].x[tc_xi*4] = __low2half(temp1);
                out_frag[xi].x[tc_xi*4+2] = __high2half(temp1);
                out_frag[xi].x[tc_xi*4+1] = __low2half(temp2);
                out_frag[xi].x[tc_xi*4+3] = __high2half(temp2);
            }
        }
        thread_sum_old = thread_sum_new;
        thread_max_old = thread_max_new;
    }

    for (int xi=0; xi < NUM_HEAD_WARPS; xi++) {
        nvcuda::wmma::store_matrix_sync(reinterpret_cast<__half*>(output_ptr + warp_id * TC_SIZE * HEAD_DIM / 2 + xi * TC_SIZE / 2), out_frag[xi], HEAD_DIM, nvcuda::wmma::mem_row_major);
    }
}

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


template <int HEAD_DIM, int BASE_SEQ_LEN>
__global__ void FMHAInferInt8Kernel(const int8_4 * __restrict__ query, const float scale_q, const int8_4 * __restrict__ key, const float scale_k, const int8_4 * __restrict__ value, const float scale_v, const float sqrt_d, int8_4 * output, const float scale_o, const int KV_LEN) {
    static_assert(BASE_SEQ_LEN % TC_SIZE == 0 && HEAD_DIM % TC_SIZE == 0 && BASE_SEQ_LEN >= HEAD_DIM, "");
    const int NUM_WARPS = BASE_SEQ_LEN / TC_SIZE;
    const int NUM_HEAD_WARPS = HEAD_DIM / TC_SIZE;

    __shared__ int share_buffer[BASE_SEQ_LEN*TC_SIZE*5/4];
    auto qk_buffer_h = reinterpret_cast<__half*>(share_buffer + BASE_SEQ_LEN*TC_SIZE/2);
    auto qk_buffer_i84 = reinterpret_cast<int8_4*>(share_buffer + BASE_SEQ_LEN*TC_SIZE);
    auto out_buffer_h = reinterpret_cast<__half*>(share_buffer);

    const unsigned int batch = blockIdx.y;
    const unsigned int Q_LEN = gridDim.x * BASE_SEQ_LEN;
    const unsigned int q_start = blockIdx.x * BASE_SEQ_LEN;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / WARP_SIZE;
    const unsigned int lane_id = tid % WARP_SIZE;
    const unsigned int warp_num = blockDim.x / WARP_SIZE;

    const unsigned int mem_lane_id_x = lane_id % 4;
    const unsigned int mem_lane_id_y = lane_id / 4;

    const float scale_qkv = scale_v * __frcp_rn(127.f);
    const __half2 scale_softmax_re = __float2half2_rn(127.f);
    const __half2 scale_o_re = __float2half2_rn(__frcp_rn(scale_o));

    int8_4 * smem_q[2];
    int8_4 * smem_k[2];
    int8_4 * smem_v[2];

    smem_q[0] = reinterpret_cast<int8_4*>(share_buffer);
    smem_q[1] = reinterpret_cast<int8_4*>(share_buffer + 1 * BASE_SEQ_LEN * TC_SIZE / 4);
    smem_k[0] = reinterpret_cast<int8_4*>(share_buffer + 2 * BASE_SEQ_LEN * TC_SIZE / 4);
    smem_k[1] = reinterpret_cast<int8_4*>(share_buffer + 3 * BASE_SEQ_LEN * TC_SIZE / 4);
    smem_v[0] = reinterpret_cast<int8_4*>(share_buffer + 2 * BASE_SEQ_LEN * TC_SIZE / 4);
    smem_v[1] = reinterpret_cast<int8_4*>(share_buffer + 3 * BASE_SEQ_LEN * TC_SIZE / 4);

    const int8_4 *query_ptr = query + batch * Q_LEN * HEAD_DIM / 4 + q_start * HEAD_DIM / 4;
    int8_4 *output_ptr = output + batch * Q_LEN * HEAD_DIM / 4 + q_start * HEAD_DIM / 4;

    __half2 thread_max_old = __half2half2(-cuda::std::numeric_limits<__half>::infinity());
    __half2 thread_sum_old = __float2half2_rn(0.f);

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, TC_SIZE, TC_SIZE, TC_SIZE, int8_t, nvcuda::wmma::row_major> q_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, TC_SIZE, TC_SIZE, TC_SIZE, int8_t, nvcuda::wmma::col_major> k_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, TC_SIZE, TC_SIZE, TC_SIZE, int8_t, nvcuda::wmma::row_major> qk_frag[NUM_WARPS];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, TC_SIZE, TC_SIZE, TC_SIZE, int8_t, nvcuda::wmma::row_major> v_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, TC_SIZE, TC_SIZE, TC_SIZE, int32_t> qk_out_frag_int32[
            NUM_WARPS > NUM_HEAD_WARPS ? NUM_WARPS : NUM_HEAD_WARPS];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, TC_SIZE, TC_SIZE, TC_SIZE, __half> qk_out_frag_half[NUM_WARPS];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, TC_SIZE, TC_SIZE, TC_SIZE, __half> out_frag[NUM_HEAD_WARPS];

#pragma unroll
    for (int xi = 0; xi < NUM_HEAD_WARPS; xi++) {
        nvcuda::wmma::fill_fragment(out_frag[xi], 0);
    }

#pragma unroll
    for (int kv_start = 0; kv_start < KV_LEN; kv_start += BASE_SEQ_LEN) {
        const int8_4 *key_ptr = key + batch * KV_LEN * HEAD_DIM / 4 + kv_start * HEAD_DIM / 4;

        __half2 thread_max = __half2half2(-cuda::std::numeric_limits<__half>::infinity());
        __half2 thread_sum = __float2half2_rn(0.f);

#pragma unroll
        for (int xi = 0; xi < NUM_WARPS; xi++) {
            nvcuda::wmma::fill_fragment(qk_out_frag_int32[xi], 0);
        }

        int k = 0, stride_warp = TC_SIZE * TC_SIZE / 4 / WARP_SIZE;
        int8_4 temp_int8_4;
#pragma unroll
        for (int i = 0; i < stride_warp; i++) {
            temp_int8_4 = *(query_ptr + warp_id * TC_SIZE * HEAD_DIM / 4 + i * WARP_SIZE * HEAD_DIM / TC_SIZE +
                            mem_lane_id_y * HEAD_DIM / 4 + mem_lane_id_x + k * TC_SIZE / 4);
            *(smem_q[k % 2] + warp_id * TC_SIZE * TC_SIZE / 4 + i * WARP_SIZE + mem_lane_id_y * TC_SIZE / 4 +
              mem_lane_id_x) = temp_int8_4;

            temp_int8_4 = *(key_ptr + warp_id * TC_SIZE * HEAD_DIM / 4 + i * WARP_SIZE * HEAD_DIM / TC_SIZE +
                            mem_lane_id_y * HEAD_DIM / 4 + mem_lane_id_x + k * TC_SIZE / 4);
            *(smem_k[k % 2] + warp_id * TC_SIZE * TC_SIZE / 4 + i * WARP_SIZE + mem_lane_id_y * TC_SIZE / 4 +
              mem_lane_id_x) = temp_int8_4;
        }

#pragma unroll
        for (k = 1; k < NUM_HEAD_WARPS; k++) {
            __syncthreads();
#pragma unroll
            for (int i = 0; i < stride_warp; i++) {
                temp_int8_4 = *(query_ptr + warp_id * TC_SIZE * HEAD_DIM / 4 + i * WARP_SIZE * HEAD_DIM / TC_SIZE +
                                mem_lane_id_y * HEAD_DIM / 4 + mem_lane_id_x + k * TC_SIZE / 4);
                *(smem_q[k % 2] + warp_id * TC_SIZE * TC_SIZE / 4 + i * WARP_SIZE + mem_lane_id_y * TC_SIZE / 4 +
                  mem_lane_id_x) = temp_int8_4;

                temp_int8_4 = *(key_ptr + warp_id * TC_SIZE * HEAD_DIM / 4 + i * WARP_SIZE * HEAD_DIM / TC_SIZE +
                                mem_lane_id_y * HEAD_DIM / 4 + mem_lane_id_x + k * TC_SIZE / 4);
                *(smem_k[k % 2] + warp_id * TC_SIZE * TC_SIZE / 4 + i * WARP_SIZE + mem_lane_id_y * TC_SIZE / 4 +
                  mem_lane_id_x) = temp_int8_4;
            }

            nvcuda::wmma::load_matrix_sync(q_frag,
                                           reinterpret_cast<int8_t *>(&(smem_q[(k - 1) % 2][warp_id * TC_SIZE *
                                                                                            TC_SIZE / 4])),
                                           TC_SIZE);
#pragma unroll
            for (int xi = 0; xi < NUM_WARPS; xi++) {
                nvcuda::wmma::load_matrix_sync(k_frag,
                                               reinterpret_cast<int8_t *>(&(smem_k[(k - 1) % 2][xi * TC_SIZE *
                                                                                                TC_SIZE / 4])),
                                               TC_SIZE);
                nvcuda::wmma::mma_sync(qk_out_frag_int32[xi], q_frag, k_frag, qk_out_frag_int32[xi]);
            }
        }
        __syncthreads();
        k = NUM_HEAD_WARPS - 1;
        nvcuda::wmma::load_matrix_sync(q_frag,
                                       reinterpret_cast<int8_t *>(&(smem_q[k % 2][warp_id * TC_SIZE * TC_SIZE /
                                                                                  4])), TC_SIZE);
#pragma unroll
        for (int xi = 0; xi < NUM_WARPS; xi++) {
            nvcuda::wmma::load_matrix_sync(k_frag,
                                           reinterpret_cast<int8_t *>(&(smem_k[k % 2][xi * TC_SIZE * TC_SIZE / 4])),
                                           TC_SIZE);
            nvcuda::wmma::mma_sync(qk_out_frag_int32[xi], q_frag, k_frag, qk_out_frag_int32[xi]);
        }

#pragma unroll
        for (int xi = 0; xi < NUM_WARPS; xi++) {
#pragma unroll
            for (int tc_xi = 0; tc_xi < 2; tc_xi++) {
                __half2 temp1 = __floats2half2_rn(qk_out_frag_int32[xi].x[tc_xi * 4] * sqrt_d,
                                                  qk_out_frag_int32[xi].x[tc_xi * 4 + 2] * sqrt_d);
                __half2 temp2 = __floats2half2_rn(qk_out_frag_int32[xi].x[tc_xi * 4 + 1] * sqrt_d,
                                                  qk_out_frag_int32[xi].x[tc_xi * 4 + 3] * sqrt_d);

                qk_out_frag_half[xi].x[tc_xi * 4] = __low2half(temp1);
                qk_out_frag_half[xi].x[tc_xi * 4 + 2] = __high2half(temp1);
                qk_out_frag_half[xi].x[tc_xi * 4 + 1] = __low2half(temp2);
                qk_out_frag_half[xi].x[tc_xi * 4 + 3] = __high2half(temp2);
                thread_max = hmax(thread_max, hmax(temp1, temp2));
            }
        }

#pragma unroll
        for (int s = 2; s > 0; s >>= 1) {
            thread_max = hmax(thread_max, __shfl_xor_sync(0xffffffff, thread_max, s, 4));
        }

#pragma unroll
        for (int xi = 0; xi < NUM_WARPS; xi++) {
#pragma unroll
            for (int tc_xi = 0; tc_xi < 2; tc_xi++) {
                __half2 temp1 = __halves2half2(qk_out_frag_half[xi].x[tc_xi * 4],
                                               qk_out_frag_half[xi].x[tc_xi * 4 + 2]);
                __half2 temp2 = __halves2half2(qk_out_frag_half[xi].x[tc_xi * 4 + 1],
                                               qk_out_frag_half[xi].x[tc_xi * 4 + 3]);
                temp1 = h2exp(__hsub2(temp1, thread_max));
                temp2 = h2exp(__hsub2(temp2, thread_max));
                qk_out_frag_half[xi].x[tc_xi * 4] = __low2half(temp1);
                qk_out_frag_half[xi].x[tc_xi * 4 + 2] = __high2half(temp1);
                qk_out_frag_half[xi].x[tc_xi * 4 + 1] = __low2half(temp2);
                qk_out_frag_half[xi].x[tc_xi * 4 + 3] = __high2half(temp2);
                thread_sum = __hadd2(thread_sum, __hadd2(temp1, temp2));
            }
        }

#pragma unroll
        for (int s = 2; s > 0; s >>= 1) {
            thread_sum = __hadd2(thread_sum, __shfl_xor_sync(0xffffffff, thread_sum, s, 4));
        }

        __syncthreads();
#pragma unroll
        for (int xi = 0; xi < NUM_WARPS; xi++) {
            nvcuda::wmma::store_matrix_sync(qk_buffer_h + warp_id * TC_SIZE * TC_SIZE, qk_out_frag_half[xi],
                                            TC_SIZE, nvcuda::wmma::mem_row_major);
            for (int i = 0; i < TC_SIZE * TC_SIZE / WARP_SIZE / 4; i++) {
                __half2 t1 = *reinterpret_cast<__half2 *>(qk_buffer_h + warp_id * TC_SIZE * TC_SIZE +
                                                          i * WARP_SIZE * 4 + lane_id * 4);
                __half2 t2 = *reinterpret_cast<__half2 *>(qk_buffer_h + warp_id * TC_SIZE * TC_SIZE +
                                                          i * WARP_SIZE * 4 + lane_id * 4 + 2);

                t1 = __hmul2(t1, scale_softmax_re);
                t2 = __hmul2(t2, scale_softmax_re);
                int8_4 t3 = int8_4(T2int8(__low2half(t1)), T2int8(__high2half(t1)), T2int8(__low2half(t2)),
                                   T2int8(__high2half(t2)));
                qk_buffer_i84[warp_id * TC_SIZE * TC_SIZE / 4 + i * WARP_SIZE + lane_id] = t3;
            }
            nvcuda::wmma::load_matrix_sync(qk_frag[xi], reinterpret_cast<int8_t *>(qk_buffer_i84 +
                                                                                   warp_id * TC_SIZE * TC_SIZE / 4),
                                           TC_SIZE);
        }
        __syncthreads();

#pragma unroll
        for (int xi = 0; xi < NUM_HEAD_WARPS; xi++) {
            nvcuda::wmma::fill_fragment(qk_out_frag_int32[xi], 0);
        }

        stride_warp = TC_SIZE / warp_num;
        const int8_4 *value_ptr =
                value + batch * KV_LEN * HEAD_DIM / 4 + (kv_start + stride_warp * warp_id) * HEAD_DIM / 4;
        k = 0;
#pragma unroll
        for (int i = 0; i < stride_warp * HEAD_DIM / 4 / WARP_SIZE; i++) {
            temp_int8_4 = *(value_ptr + k * TC_SIZE * HEAD_DIM / 4 + i * WARP_SIZE + lane_id);
            *(smem_v[k % 2] + warp_id * stride_warp * HEAD_DIM / 4 + i * WARP_SIZE + lane_id) = temp_int8_4;
        }
#pragma unroll
        for (k = 1; k < NUM_WARPS; k++) {
            __syncthreads();
            for (int i = 0; i < stride_warp * HEAD_DIM / 4 / WARP_SIZE; i++) {
                temp_int8_4 = *(value_ptr + k * TC_SIZE * HEAD_DIM / 4 + i * WARP_SIZE + lane_id);
                *(smem_v[k % 2] + warp_id * stride_warp * HEAD_DIM / 4 + i * WARP_SIZE + lane_id) = temp_int8_4;
            }
#pragma unroll
            for (int xi = 0; xi < NUM_HEAD_WARPS; xi++) {
                nvcuda::wmma::load_matrix_sync(v_frag,
                                               reinterpret_cast<int8_t *>(&(smem_v[(k - 1) % 2][xi * TC_SIZE / 4])),
                                               HEAD_DIM);
                nvcuda::wmma::mma_sync(qk_out_frag_int32[xi], qk_frag[k - 1], v_frag, qk_out_frag_int32[xi]);
            }
        }
        __syncthreads();
        k = NUM_WARPS - 1;
#pragma unroll
        for (int xi = 0; xi < NUM_HEAD_WARPS; xi++) {
            nvcuda::wmma::load_matrix_sync(v_frag, reinterpret_cast<int8_t *>(&(smem_v[k % 2][xi * TC_SIZE / 4])),
                                           HEAD_DIM);
            nvcuda::wmma::mma_sync(qk_out_frag_int32[xi], qk_frag[k], v_frag, qk_out_frag_int32[xi]);
        }

        __half2 thread_max_new = hmax(thread_max_old, thread_max);
        __half2 exp_max_old = h2exp(__hsub2(thread_max_old, thread_max_new));
        __half2 exp_max = h2exp(__hsub2(thread_max, thread_max_new));
        __half2 thread_sum_new = __hadd2(__hmul2(exp_max_old, thread_sum_old), __hmul2(exp_max, thread_sum));

#pragma unroll
        for (int xi = 0; xi < NUM_HEAD_WARPS; xi++) {
#pragma unroll
            for (int tc_xi = 0; tc_xi < 2; tc_xi++) {
                __half2 temp1 = __halves2half2(out_frag[xi].x[tc_xi * 4], out_frag[xi].x[tc_xi * 4 + 2]);
                __half2 temp2 = __halves2half2(out_frag[xi].x[tc_xi * 4 + 1], out_frag[xi].x[tc_xi * 4 + 3]);
                __half2 temp1_kv = __floats2half2_rn(qk_out_frag_int32[xi].x[tc_xi * 4] * scale_qkv,
                                                     qk_out_frag_int32[xi].x[tc_xi * 4 + 2] * scale_qkv);
                __half2 temp2_kv = __floats2half2_rn(qk_out_frag_int32[xi].x[tc_xi * 4 + 1] * scale_qkv,
                                                     qk_out_frag_int32[xi].x[tc_xi * 4 + 3] * scale_qkv);

                temp1 = __h2div(__hfma2(exp_max, temp1_kv, __hmul2(__hmul2(thread_sum_old, exp_max_old), temp1)),
                                thread_sum_new);
                temp2 = __h2div(__hfma2(exp_max, temp2_kv, __hmul2(__hmul2(thread_sum_old, exp_max_old), temp2)),
                                thread_sum_new);

                out_frag[xi].x[tc_xi * 4] = __low2half(temp1);
                out_frag[xi].x[tc_xi * 4 + 2] = __high2half(temp1);
                out_frag[xi].x[tc_xi * 4 + 1] = __low2half(temp2);
                out_frag[xi].x[tc_xi * 4 + 3] = __high2half(temp2);
            }
        }
        thread_sum_old = thread_sum_new;
        thread_max_old = thread_max_new;
    }

#pragma unroll
    for (int xi = 0; xi < NUM_HEAD_WARPS; xi++) {
        nvcuda::wmma::store_matrix_sync(out_buffer_h + warp_id * TC_SIZE * TC_SIZE, out_frag[xi], TC_SIZE,
                                        nvcuda::wmma::mem_row_major);
#pragma unroll
        for (int i = 0; i < TC_SIZE * TC_SIZE / WARP_SIZE / 4; i++) {
            __half2 t1 = *reinterpret_cast<__half2 *>(out_buffer_h + warp_id * TC_SIZE * TC_SIZE +
                                                      i * WARP_SIZE * 4 + lane_id * 4);
            __half2 t2 = *reinterpret_cast<__half2 *>(out_buffer_h + warp_id * TC_SIZE * TC_SIZE +
                                                      i * WARP_SIZE * 4 + lane_id * 4 + 2);

            t1 = __hmul2(t1, scale_o_re);
            t2 = __hmul2(t2, scale_o_re);

            int8_4 t3 = int8_4(T2int8(__low2half(t1)), T2int8(__high2half(t1)), T2int8(__low2half(t2)),
                               T2int8(__high2half(t2)));
            *(output_ptr + warp_id * TC_SIZE * HEAD_DIM / 4 + xi * TC_SIZE / 4 +
              i * HEAD_DIM * WARP_SIZE / TC_SIZE + mem_lane_id_y * HEAD_DIM / 4 + mem_lane_id_x) = t3;
        }
    }
}

template <typename T>
int qkv_flash(const T * query, const T * key, const T *value, T *output, const int &batch, const int &q_len, const int &kv_len, const int &embed_dim, cudaStream_t stream) {
    const T sqrt_d = 1.f / std::sqrt((float)embed_dim);

#define HEAD_DEFINE(BASE_LEN) \
    if (embed_dim == 32) { \
        FMHAInferKernel<32, BASE_LEN><<<dim3(q_len/BASE_LEN, batch), BASE_LEN*2, 0, stream>>>(query, key, value, sqrt_d, output, kv_len); \
    } else if (embed_dim == 64) { \
        FMHAInferKernel<64, BASE_LEN><<<dim3(q_len/BASE_LEN, batch), BASE_LEN*2, 0, stream>>>(query, key, value, sqrt_d, output, kv_len); \
    } else { \
        printf("Do not support head_dim=%d\n", embed_dim); \
        exit(1);    \
    }

    if (q_len % 128 == 0 && kv_len % 128 == 0) {
        HEAD_DEFINE(128)
    } else if (q_len % 64 == 0 && kv_len % 64 == 0) {
        HEAD_DEFINE(64)
    } else {
        printf("Do not support q_len=%d, kv_len=%d\n", q_len, kv_len);
        exit(1);
    }
    cudaCheckError();
    return 0;
}

template <>
int qkv_flash(const __half2 * query, const __half2 * key, const __half2 *value, __half2 *output, const int &batch, const int &q_len, const int &kv_len, const int &embed_dim, cudaStream_t stream) {
    const __half2 sqrt_d = __float2half2_rn(1.f / std::sqrt((float)embed_dim));

#define HEAD_DEFINE(BASE_LEN) \
    if (embed_dim == 32) { \
        FMHAInferKernel<32, BASE_LEN><<<dim3(q_len/BASE_LEN, batch), BASE_LEN*2, 0, stream>>>(query, key, value, sqrt_d, output, kv_len); \
    } else if (embed_dim == 64) { \
        FMHAInferKernel<64, BASE_LEN><<<dim3(q_len/BASE_LEN, batch), BASE_LEN*2, 0, stream>>>(query, key, value, sqrt_d, output, kv_len); \
    } else { \
        printf("Do not support head_dim=%d\n", embed_dim); \
        exit(1);    \
    }

    if (q_len % 128 == 0 && kv_len % 128 == 0) {
        HEAD_DEFINE(128)
    } else if (q_len % 64 == 0 && kv_len % 64 == 0) {
        HEAD_DEFINE(64)
    } else {
        printf("Do not support q_len=%d, kv_len=%d\n", q_len, kv_len);
        exit(1);
    }
    cudaCheckError();
    return 0;
}


int qkv_flash_int8(const int8_4 * query, const float&scale_q, const int8_4 * key, const float&scale_k, const int8_4 *value, const float&scale_v, int8_4 *output, const float&scale_o, const int &batch, const int &q_len, const int &kv_len, const int &embed_dim, cudaStream_t stream) {
    const float sqrt_d = scale_q * scale_k / std::sqrt((float)embed_dim);

#define HEAD_DEFINE_INT8(BASE_LEN) \
    if (embed_dim == 32) { \
        FMHAInferInt8Kernel<32, BASE_LEN><<<dim3(q_len/BASE_LEN, batch), BASE_LEN*2, 0, stream>>>(query, scale_q, key, scale_k, value, scale_v, sqrt_d, output, scale_o, kv_len); \
    } else if (embed_dim == 64) { \
        FMHAInferInt8Kernel<64, BASE_LEN><<<dim3(q_len/BASE_LEN, batch), BASE_LEN*2, 0, stream>>>(query, scale_q, key, scale_k, value, scale_v, sqrt_d, output, scale_o, kv_len); \
    } else { \
        printf("Do not support head_dim=%d\n", embed_dim); \
        exit(1);    \
    }
    
    if (q_len % 128 == 0 && kv_len % 128 == 0) {
        HEAD_DEFINE_INT8(128)
    } else if (q_len % 64 == 0 && kv_len % 64 == 0) {
        HEAD_DEFINE_INT8(64)
    } else {
        printf("Do not support q_len=%d, kv_len=%d\n", q_len, kv_len);
        exit(1);
    }
    cudaCheckError();
    return 0;
}


template int qkv_flash(const float *query, const float *key, const float *value, float *output, const int &batch, const int &q_len, const int &kv_len, const int &embed_dim, cudaStream_t stream);
template int qkv_flash(const __half *query, const __half *key, const __half *value, __half *output, const int &batch, const int &q_len, const int &kv_len, const int &embed_dim, cudaStream_t stream);
template int qkv_flash(const __half2 *query, const __half2 *key, const __half2 *value, __half2 *output, const int &batch, const int &q_len, const int &kv_len, const int &embed_dim, cudaStream_t stream);
