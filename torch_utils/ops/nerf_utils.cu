// Copyright (c) Facebook, Inc. and its affiliates.All Rights Reserved


#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/torch.h>
#include <torch/extension.h>

#include "utils.h"


template <typename T>
__host__ __device__ T div_round_up(T val, T divisor) {
	return (val + divisor - 1) / divisor;
}


template <uint32_t S>
__global__ void kernel_topp_masking(
    const int * __restrict__ sorted_indices,
    const float * __restrict__ sorted_weights, 
    bool *output_mask, 
    const float p, const uint32_t B, 
    const uint32_t N, const uint32_t D) {

    const uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= N) return;

    const uint32_t batch_id = blockIdx.y;
    
    // locate
    sorted_weights += (b + batch_id * N) * D;
    sorted_indices += (b + batch_id * N) * D;
    output_mask += (b + batch_id * N) * D;
    
    float w_sum = 0;

    #pragma unroll
    for (uint32_t d = 0; d < S; d++){
        if (d >= D) break;
        w_sum += sorted_weights[d];
        output_mask[sorted_indices[d]] = true;
        if (w_sum >= p) break;
    }
    }

void topp_masking_cuda(
    const int *sorted_indices, 
    const float *sorted_weights, bool *output_mask, 
    const float p, const uint32_t B, const uint32_t N, const uint32_t D) {
    static constexpr uint32_t N_THREAD = 512;
    const dim3 blocks = {div_round_up(N, N_THREAD), B, 1};
    if (D < 8)        kernel_topp_masking<8><<<  blocks, N_THREAD>>>(sorted_indices, sorted_weights, output_mask, p, B, N, D);
    else if (D < 16)  kernel_topp_masking<16><<< blocks, N_THREAD>>>(sorted_indices, sorted_weights, output_mask, p, B, N, D);
    else if (D < 32)  kernel_topp_masking<32><<< blocks, N_THREAD>>>(sorted_indices, sorted_weights, output_mask, p, B, N, D);
    else if (D < 64)  kernel_topp_masking<64><<< blocks, N_THREAD>>>(sorted_indices, sorted_weights, output_mask, p, B, N, D);
    else if (D < 128) kernel_topp_masking<128><<<blocks, N_THREAD>>>(sorted_indices, sorted_weights, output_mask, p, B, N, D);
    else if (D < 256) kernel_topp_masking<256><<<blocks, N_THREAD>>>(sorted_indices, sorted_weights, output_mask, p, B, N, D);
    else throw std::runtime_error{"# of sampled points should not exceed 256"};

}

void topp_masking(
    at::Tensor sorted_indices, at::Tensor sorted_weights, at::Tensor output_mask, 
    const float p, const uint32_t B, const uint32_t N, const uint32_t D) {
    CHECK_CUDA(sorted_indices);
    CHECK_CUDA(sorted_weights);
    CHECK_CUDA(output_mask);
  
    CHECK_CONTIGUOUS(sorted_indices);
    CHECK_CONTIGUOUS(sorted_weights);
    CHECK_CONTIGUOUS(output_mask);

    CHECK_IS_FLOAT(sorted_weights);
    CHECK_IS_INT(sorted_indices);
    
    topp_masking_cuda(sorted_indices.data_ptr<int>(), sorted_weights.data_ptr<float>(), output_mask.data_ptr<bool>(), p, B, N, D);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("topp_masking", &topp_masking, "topp masking");
}
