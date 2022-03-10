# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import os
import torch
from .. import custom_ops


_plugin = None

def _init():
    global _plugin
    if _plugin is None:
        _plugin = custom_ops.get_plugin(
            module_name='nerf_utils_plugin',
            sources=['nerf_utils.cu'],
            headers=['utils.h'],
            source_dir=os.path.dirname(__file__),
            extra_cuda_cflags=['--use_fast_math'],
        )

    return True

def topp_masking(w, p=0.99):
    """
    w: B x N x S  normalized (S number of samples)
    p: top-P used
    """ 
    # _init()
    w_sorted, w_indices = w.sort(dim=-1, descending=True)
    
    w_mask = w_sorted.cumsum(-1).lt(p)
    w_mask = torch.cat([torch.ones_like(w_mask[...,:1]), w_mask[..., :-1]], -1)
    w_mask = w_mask.scatter(-1, w_indices, w_mask)
    
    # w_mask = torch.zeros_like(w).bool()
    # _plugin.topp_masking(w_indices.int(), w_sorted, w_mask, p, w.size(0), w.size(1), w.size(2))
    return w_mask