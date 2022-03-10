# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Custom replacement for `torch.nn.functional.grid_sample` that
supports arbitrarily high order gradients between the input and output.
Only works on 2D images and assumes
`mode='bilinear'`, `padding_mode='zeros'`, `align_corners=False`."""

import torch

# pylint: disable=redefined-builtin
# pylint: disable=arguments-differ
# pylint: disable=protected-access

#----------------------------------------------------------------------------

enabled = True  # Enable the custom op by setting this to true.

#----------------------------------------------------------------------------

def grid_sample(input, grid):
    if _should_use_custom_op():
        return _GridSampleForward.apply(input, grid)
    return torch.nn.functional.grid_sample(input=input, grid=grid, mode='bilinear', padding_mode='zeros', align_corners=False)

#----------------------------------------------------------------------------

def _should_use_custom_op():
    return enabled

#----------------------------------------------------------------------------

class _GridSampleForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, grid):
        assert input.ndim == 4 or input.ndim == 5
        assert grid.ndim == 4 or input.ndim == 5
        output = torch.nn.functional.grid_sample(input=input, grid=grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        ctx.save_for_backward(input, grid)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, grid = ctx.saved_tensors
        grad_input, grad_grid = _GridSampleBackward.apply(grad_output, input, grid)
        return grad_input, grad_grid

#----------------------------------------------------------------------------

class _GridSampleBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grad_output, input, grid):
        if input.ndim == 4:
            op = torch._C._jit_get_operation('aten::grid_sampler_2d_backward')
        else:
            op = torch._C._jit_get_operation('aten::grid_sampler_3d_backward')
        grad_input, grad_grid = op(grad_output, input, grid, 0, 0, False)
        ctx.save_for_backward(grid)
        return grad_input, grad_grid

    @staticmethod
    def backward(ctx, grad2_grad_input, grad2_grad_grid):
        _ = grad2_grad_grid # unused
        grid, = ctx.saved_tensors
        grad2_grad_output = None
        grad2_input = None
        grad2_grid = None

        if ctx.needs_input_grad[0]:
            grad2_grad_output = _GridSampleForward.apply(grad2_grad_input, grid)

        assert not ctx.needs_input_grad[2]
        return grad2_grad_output, grad2_input, grad2_grid

#----------------------------------------------------------------------------
