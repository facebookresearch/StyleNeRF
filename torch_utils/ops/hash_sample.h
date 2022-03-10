// Copyright (c) Facebook, Inc. and its affiliates.All Rights Reserved


// Please refer to original code: https://github.com/NVlabs/instant-ngp
// and the pytorch wrapper from https://github.com/ashawkey/torch-ngp

#ifndef _HASH_SAMPLE_H
#define _HASH_SAMPLE_H

#include <stdint.h>
#include <torch/torch.h>
#include <torch/extension.h>

// inputs: [B, N, D], float, in [0, 1]
// embeddings: [B, sO, C], float
// offsets: [L + 1], uint32_t
// outputs: [B, N, L * C], float
// H: base resolution
void hash_encode_forward(at::Tensor inputs, at::Tensor embeddings, at::Tensor offsets, at::Tensor outputs, const float beta, const uint32_t B, const uint32_t N, const uint32_t D, const uint32_t C, const uint32_t L, const uint32_t H, const bool calc_grad_inputs, at::Tensor dy_dx, const uint32_t mode);
void hash_encode_backward(at::Tensor grad, at::Tensor inputs, at::Tensor embeddings, at::Tensor offsets, at::Tensor grad_embeddings, const float beta, const uint32_t B,  const uint32_t N, const uint32_t D, const uint32_t C, const uint32_t L, const uint32_t H, const bool calc_grad_inputs, at::Tensor dy_dx, at::Tensor grad_inputs, const uint32_t mode);
void hash_encode_forward_cuda(const float *inputs, const float *embeddings, const int *offsets, float *outputs,  const float beta, const uint32_t B,  const uint32_t N, const uint32_t D, const uint32_t C, const uint32_t L, const uint32_t H, const bool calc_grad_inputs, float *dy_dx, const uint32_t mode);
void hash_encode_backward_cuda(const float *grad, const float *inputs, const float *embeddings, const int *offsets, float *grad_embeddings,  const float beta, const uint32_t B,  const uint32_t N, const uint32_t D, const uint32_t C, const uint32_t L, const uint32_t H, const bool calc_grad_inputs, float *dy_dx, float *grad_inputs, const uint32_t mode);
#endif