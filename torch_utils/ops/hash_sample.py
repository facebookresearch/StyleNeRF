# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# Please refer to original code: https://github.com/NVlabs/instant-ngp
# and the pytorch wrapper from https://github.com/ashawkey/torch-ngp

import os
import torch

from .. import custom_ops
from torch.cuda.amp import custom_bwd, custom_fwd

_plugin = None
_null_tensor = torch.empty([0])

def _init():
    global _plugin
    if _plugin is None:
        _plugin = custom_ops.get_plugin(
            module_name='hash_sample_plugin',
            sources=['hash_sample.cpp', 'hash_sample.cu'],
            headers=['hash_sample.h', 'utils.h'],
            source_dir=os.path.dirname(__file__),
            extra_cuda_cflags=['--use_fast_math'],
        )
    return True


def hash_sample(x, h, offsets, beta=2, base_res=16, calc_grad=True, mode='fast_hash'):
    """Hash-table look up and d-linear interpolation
       x: B x N x D       coordinates
       h: B x L x T x C   hash-tables
       offsets: L resolutions
    """
    assert x.device.type == 'cuda'
    assert (x.size(-1) == 3) or (x.size(-1) == 2), "currently only 2D/3D is implemented"
    _init()    
    return _hash_sample_cuda(mode).apply(x, h, offsets, beta, base_res, calc_grad)

    
_hash_sample_cuda_cache = dict()

def _hash_sample_cuda(mode='fast_hash'):
    """CUDA implementation of hash-table look-up
    """
    if mode in _hash_sample_cuda_cache:
        return _hash_sample_cuda_cache[mode]

    if mode == 'fast_hash':
        h_mode = 0
    elif mode == 'grid_hash':
        h_mode = 1
    else:
        raise NotImplementedError('only two types are supported now.')

    class HashSampleCuda(torch.autograd.Function):
        @staticmethod
        @custom_fwd(cast_inputs=torch.half)
        def forward(ctx, inputs, embeddings, offsets, beta, base_resolution, calc_grad_inputs=False):
            # inputs:     [B, N, D], float in [0, 1]
            # embeddings: [B, sO, C], float
            # offsets:    [L + 1], int
            # RETURN:     [B, N, F], float

            inputs = inputs.contiguous()
            embeddings = embeddings.contiguous()
            offsets = offsets.contiguous().to(inputs.device)

            B, N, D = inputs.shape     # batch size, # of samples, coord dim
            L = offsets.shape[0] - 1   # level
            C = embeddings.shape[-1]   # embedding dim for each level
            H = base_resolution        # base resolution
            
            outputs = torch.zeros(B, N, L * C, device=inputs.device, dtype=inputs.dtype)

            if calc_grad_inputs:
                dy_dx = torch.zeros(B, N, L * D * C).to(inputs.device, dtype=inputs.dtype)
            else:
                dy_dx = torch.zeros(1).to(inputs.device, dtype=inputs.dtype)
   
            _plugin.hash_encode_forward(inputs, embeddings, offsets, outputs, beta, B, N, D, C, L, H, calc_grad_inputs, dy_dx, h_mode)

            ctx.save_for_backward(inputs, embeddings, offsets, dy_dx)
            ctx.dims = [B, N, D, C, L, H, beta]
            ctx.calc_grad_inputs = calc_grad_inputs

            return outputs
        
        @staticmethod
        @custom_bwd
        def backward(ctx, grad):
            # grad: [B, L * C]

            grad = grad.contiguous()

            inputs, embeddings, offsets, dy_dx = ctx.saved_tensors
            B, N, D, C, L, H, beta = ctx.dims
            calc_grad_inputs = ctx.calc_grad_inputs

            grad_embeddings = torch.zeros_like(embeddings)

            if calc_grad_inputs:
                grad_inputs = torch.zeros_like(inputs)
            else:
                grad_inputs = torch.zeros(1).to(inputs.device, dtype=inputs.dtype)

            _plugin.hash_encode_backward(grad, inputs, embeddings, offsets, grad_embeddings, beta, B, N, D, C, L, H, calc_grad_inputs, dy_dx, grad_inputs, h_mode)

            if calc_grad_inputs:
                return grad_inputs, grad_embeddings, None, None, None, None
            else:
                return None, grad_embeddings, None, None, None, None


    # Add to cache.
    _hash_sample_cuda_cache[mode] = HashSampleCuda
    return HashSampleCuda