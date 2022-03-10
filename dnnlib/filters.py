# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import math

import torch
from torch import nn
from torch.nn import functional as F


def kaiser_attenuation(n_taps, f_h, sr):
    df = (2 * f_h) / (sr / 2)
    return 2.285 * (n_taps - 1) * math.pi * df + 7.95


def kaiser_beta(n_taps, f_h, sr):
    atten = kaiser_attenuation(n_taps, f_h, sr)

    if atten > 50:
        return 0.1102 * (atten - 8.7)

    elif 50 >= atten >= 21:
        return 0.5842 * (atten - 21) ** 0.4 + 0.07886 * (atten - 21)

    else:
        return 0.0

def sinc(x, eps=1e-10):
    y = torch.sin(math.pi * x) / (math.pi * x + eps)
    y = y.masked_fill(x.eq(0), 1.0)
    return y


def kaiser_window(n_taps, f_h, sr):
    beta = kaiser_beta(n_taps, f_h, sr)
    ind = torch.arange(n_taps) - (n_taps - 1) / 2
    return torch.i0(beta * torch.sqrt(1 - ((2 * ind) / (n_taps - 1)) ** 2)) / torch.i0(
        torch.tensor(beta)
    )


def lowpass_filter(n_taps, cutoff, band_half, sr):
    window = kaiser_window(n_taps, band_half, sr)
    ind = torch.arange(n_taps) - (n_taps - 1) / 2
    lowpass = 2 * cutoff / sr * sinc(2 * cutoff / sr * ind) * window
    return lowpass


def filter_parameters(
    n_layer,
    n_critical,
    sr_max,
    cutoff_0,
    cutoff_n,
    stopband_0,
    stopband_n
):
    cutoffs = []
    stopbands = []
    srs = []
    band_halfs = []

    for i in range(n_layer):
        f_c = cutoff_0 * (cutoff_n / cutoff_0) ** min(i / (n_layer - n_critical), 1)
        f_t = stopband_0 * (stopband_n / stopband_0) ** min(
            i / (n_layer - n_critical), 1
        )
        s_i = 2 ** math.ceil(math.log(min(2 * f_t, sr_max), 2))
        f_h = max(f_t, s_i / 2) - f_c

        cutoffs.append(f_c)
        stopbands.append(f_t)
        srs.append(s_i)
        band_halfs.append(f_h)

    return {
        "cutoffs": cutoffs,
        "stopbands": stopbands,
        "srs": srs,
        "band_halfs": band_halfs,
    }
