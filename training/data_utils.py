# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import PIL.Image
import torch
import cv2, albumentations
import numpy as np


def save_image(img, filename):
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(filename)


def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape(gh, gw, C, H, W)
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape(gh * H, gw * W, C)

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)


def resize_image(img_pytorch, curr_res):
    img = img_pytorch.permute(0,2,3,1).cpu().numpy()
    img = [albumentations.geometric.functional.resize(
        img[i], height=curr_res, width=curr_res, interpolation=cv2.INTER_LANCZOS4)
        for i in range(img.shape[0])]
    img = torch.from_numpy(np.stack(img, axis=0)).permute(0,3,1,2).to(img_pytorch.device)
    return img