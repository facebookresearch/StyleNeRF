# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from pickle import NONE
from re import X
from sndhdr import whathdr
import numpy as np
import math
import scipy.signal
import scipy.optimize

from numpy import core
from numpy.lib.arraysetops import isin

import torch
import torch.nn.functional as F
from torch.overrides import is_tensor_method_or_property
from einops import repeat
from dnnlib import camera, util, geometry
from torch_utils import misc
from torch_utils import persistence
from torch_utils.ops import conv2d_resample
from torch_utils.ops import upfirdn2d
from torch_utils.ops import bias_act
from torch_utils.ops import fma
from torch_utils.ops import filtered_lrelu

#----------------------------------------------------------------------------

@misc.profiled_function
def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()


@misc.profiled_function
def conv3d(x, w, up=1, down=1, padding=0, groups=1):
    if up > 1:
        x = F.interpolate(x, scale_factor=up, mode='trilinear', align_corners=True)
    x = F.conv3d(x, w, padding=padding, groups=groups)
    if down > 1:
        x = F.interpolate(x, scale_factor=1./float(down), mode='trilinear', align_corners=True)
    return x

#----------------------------------------------------------------------------

@misc.profiled_function
def modulated_conv2d(
    x,                          # Input tensor of shape [batch_size, in_channels, in_height, in_width].
    weight,                     # Weight tensor of shape [out_channels, in_channels, kernel_height, kernel_width].
    styles,                     # Modulation coefficients of shape [batch_size, in_channels].
    noise           = None,     # Optional noise tensor to add to the output activations.
    up              = 1,        # Integer upsampling factor.
    down            = 1,        # Integer downsampling factor.
    padding         = 0,        # Padding with respect to the upsampled image.
    resample_filter = None,     # Low-pass filter to apply when resampling activations. Must be prepared beforehand by calling upfirdn2d.setup_filter().
    demodulate      = True,     # Apply weight demodulation?
    flip_weight     = True,     # False = convolution, True = correlation (matches torch.nn.functional.conv2d) ????????
    fused_modconv   = True,     # Perform modulation, convolution, and demodulation as a single fused operation?
    mode            = '2d',     # modulated 2d/3d conv or MLP
    **unused,
):
    batch_size = x.shape[0]
    if mode == '3d':
        _, in_channels, kd, kh, kw = weight.shape
    else:
        _, in_channels, kh, kw = weight.shape

    # Pre-normalize inputs to avoid FP16 overflow.
    if x.dtype == torch.float16 and demodulate:
        weight_sizes = in_channels * kh * kw if mode != '3d' else in_channels * kd * kh * kw
        weight = weight * (1 / np.sqrt(weight_sizes) / weight.norm(float('inf'), dim=[1,2,3], keepdim=True)) # max_Ikk
        styles = styles / styles.norm(float('inf'), dim=1, keepdim=True) # max_I

    # Calculate per-sample weights and demodulation coefficients.
    w = None
    dcoefs = None
    if mode != '3d':
        rsizes, ssizes = [-1, 1, 1], [2, 3, 4]
    else:
        rsizes, ssizes = [-1, 1, 1, 1], [2, 3, 4, 5]

    if demodulate or fused_modconv:  # if not fused, skip
        w =  weight.unsqueeze(0) * styles.reshape(batch_size, 1, *rsizes)
    if demodulate:
        dcoefs = (w.square().sum(dim=ssizes) + 1e-8).rsqrt() # [NO]

    if demodulate and fused_modconv:
        w = w * dcoefs.reshape(batch_size, *rsizes, 1) # [NOIkk]  (batch_size, out_channels, in_channels, kernel_size, kernel_size)

    # Execute by scaling the activations before and after the convolution.
    if not fused_modconv:
        x = x * styles.to(x.dtype).reshape(batch_size, *rsizes)
        if mode == '2d':
            x = conv2d_resample.conv2d_resample(x=x, w=weight.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, flip_weight=flip_weight)
        elif mode == '3d':
            x = conv3d(x=x, w=weight.to(x.dtype), up=up, down=down, padding=padding)
        else:
            raise NotImplementedError

        if demodulate and noise is not None:
            x = fma.fma(x, dcoefs.to(x.dtype).reshape(batch_size, *rsizes), noise.to(x.dtype))  # fused multiply add
        elif demodulate:
            x = x * dcoefs.to(x.dtype).reshape(batch_size, *rsizes)
        elif noise is not None:
            x = x.add_(noise.to(x.dtype))
        return x

    # Execute as one fused op using grouped convolution.
    with misc.suppress_tracer_warnings(): # this value will be treated as a constant
        batch_size = int(batch_size)

    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, *w.shape[2:])
    if mode == '2d':
        x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, groups=batch_size, flip_weight=flip_weight)
    elif mode == '3d':
        x = conv3d(x=x, w=w.to(x.dtype), up=up, down=down, padding=padding, groups=batch_size)
    x = x.reshape(batch_size, -1, *x.shape[2:])
    
    if noise is not None:
        x = x.add_(noise)
    return x


#----------------------------------------------------------------------------

@persistence.persistent_class
class FullyConnectedLayer(torch.nn.Module):
    def __init__(self,
        in_features,                # Number of input features.
        out_features,               # Number of output features.
        bias            = True,     # Apply additive bias before the activation function?
        activation      = 'linear', # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 1,        # Learning rate multiplier.
        bias_init       = 0,        # Initial value for the additive bias.
    ):
        super().__init__()
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = torch.nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class Conv2dLayer(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        kernel_size,                    # Width and height of the convolution kernel.
        bias            = True,         # Apply additive bias before the activation function?
        activation      = 'linear',     # Activation function: 'relu', 'lrelu', etc.
        up              = 1,            # Integer upsampling factor.
        down            = 1,            # Integer downsampling factor.
        resample_filter = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp      = None,         # Clamp the output to +-X, None = disable clamping.
        channels_last   = False,        # Expect the input to have memory_format=channels_last?
        trainable       = True,         # Update the weights of this layer during training?
        mode            = '2d',
        **unused
    ):
        super().__init__()
        self.activation = activation
        self.up = up
        self.down = down
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))
        self.act_gain = bias_act.activation_funcs[activation].def_gain
        self.mode = mode
        weight_shape = [out_channels, in_channels, kernel_size, kernel_size]
        if mode == '3d':
            weight_shape += [kernel_size]

        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        weight = torch.randn(weight_shape).to(memory_format=memory_format)
        bias = torch.zeros([out_channels]) if bias else None
        if trainable:
            self.weight = torch.nn.Parameter(weight)
            self.bias = torch.nn.Parameter(bias) if bias is not None else None
        else:
            self.register_buffer('weight', weight)
            if bias is not None:
                self.register_buffer('bias', bias)
            else:
                self.bias = None

    def forward(self, x, gain=1):
        w = self.weight * self.weight_gain
        b = self.bias.to(x.dtype) if self.bias is not None else None
        flip_weight = (self.up == 1)  # slightly faster

        if self.mode == '2d':
            x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=self.resample_filter, up=self.up, down=self.down, padding=self.padding, flip_weight=flip_weight)
        elif self.mode == '3d':
            x = conv3d(x=x, w=w.to(x.dtype), up=self.up, down=self.down, padding=self.padding)

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act.bias_act(x, b, act=self.activation, gain=act_gain, clamp=act_clamp)
        return x

# ---------------------------------------------------------------------------

@persistence.persistent_class
class Blur(torch.nn.Module):
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer('f', f)

    def forward(self, x):
        from kornia.filters import filter2d
        f = self.f
        f = f[None, None, :] * f [None, :, None]
        return filter2d(x, f, normalized=True)

#----------------------------------------------------------------------------

@persistence.persistent_class
class MappingNetwork(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality, 0 = no latent.
        c_dim,                      # Conditioning label (C) dimensionality, 0 = no label.
        w_dim,                      # Intermediate latent (W) dimensionality.
        num_ws,                     # Number of intermediate latents to output, None = do not broadcast.
        num_layers      = 8,        # Number of mapping layers.
        embed_features  = None,     # Label embedding dimensionality, None = same as w_dim.
        layer_features  = None,     # Number of intermediate features in the mapping layers, None = same as w_dim.
        activation      = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 0.01,     # Learning rate multiplier for the mapping layers.
        w_avg_beta      = 0.995,    # Decay for tracking the moving average of W during training, None = do not track.
        **unused,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        if embed_features is None:
            embed_features = w_dim
        if c_dim == 0:
            embed_features = 0
        if layer_features is None:
            layer_features = w_dim
        features_list = [z_dim + embed_features] + [layer_features] * (num_layers - 1) + [w_dim]

        if c_dim > 0:   # project label condition
            self.embed = FullyConnectedLayer(c_dim, embed_features)
        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = FullyConnectedLayer(in_features, out_features, activation=activation, lr_multiplier=lr_multiplier)
            setattr(self, f'fc{idx}', layer)

        if num_ws is not None and w_avg_beta is not None:
            self.register_buffer('w_avg', torch.zeros([w_dim]))

    def forward(self, z=None, c=None, truncation_psi=1, truncation_cutoff=None, skip_w_avg_update=False, styles=None, **unused_kwargs):
        if styles is not None:
            return styles

        # Embed, normalize, and concat inputs.
        x = None
        with torch.autograd.profiler.record_function('input'):
            if self.z_dim > 0:
                misc.assert_shape(z, [None, self.z_dim])
                x = normalize_2nd_moment(z.to(torch.float32))   # normalize z to shpere
            if self.c_dim > 0:
                misc.assert_shape(c, [None, self.c_dim])
                y = normalize_2nd_moment(self.embed(c.to(torch.float32)))
                x = torch.cat([x, y], dim=1) if x is not None else y
        
        # Main layers.
        for idx in range(self.num_layers):
            layer = getattr(self, f'fc{idx}')
            x = layer(x)

        # Update moving average of W.
        if self.w_avg_beta is not None and self.training and not skip_w_avg_update:
            with torch.autograd.profiler.record_function('update_w_avg'):
                self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

        # Broadcast.
        if self.num_ws is not None:
            with torch.autograd.profiler.record_function('broadcast'):
                x = x.unsqueeze(1).repeat([1, self.num_ws, 1])

        # Apply truncation.
        if truncation_psi != 1:
            with torch.autograd.profiler.record_function('truncate'):
                assert self.w_avg_beta is not None
                if self.num_ws is None or truncation_cutoff is None:
                    x = self.w_avg.lerp(x, truncation_psi)
                else:
                    x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisLayer(torch.nn.Module):
    def __init__(self,
        in_channels,                     # Number of input channels.
        out_channels,                    # Number of output channels.
        w_dim,                           # Intermediate latent (W) dimensionality.
        resolution,                      # Resolution of this layer.
        kernel_size        = 3,            # Convolution kernel size.
        up                 = 1,            # Integer upsampling factor.
        use_noise          = True,         # Enable noise input?
        activation         = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
        resample_filter    = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp         = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        channels_last      = False,        # Use channels_last format for the weights?
        upsample_mode      = 'default',    # [default, bilinear, ray_comm, ray_attn, ray_penc]
        use_group          = False,
        magnitude_ema_beta = -1,           # -1 means not using magnitude ema
        mode               = '2d',         # choose from 1d, 2d or 3d
        **unused_kwargs
    ):
        super().__init__()
        self.resolution = resolution
        self.up = up
        self.use_noise = use_noise
        self.activation = activation
        self.conv_clamp = conv_clamp
        self.upsample_mode = upsample_mode
        self.mode = mode

        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        if up == 2:
            if 'pixelshuffle' in upsample_mode:
                self.adapter = torch.nn.Sequential(
                    Conv2dLayer(out_channels, out_channels // 4, kernel_size=1, activation=activation),
                    Conv2dLayer(out_channels // 4, out_channels * 4, kernel_size=1, activation='linear'),
                )
            elif upsample_mode == 'liif':
                from dnnlib.geometry import get_grids, local_ensemble
                pi = get_grids(self.resolution//2, self.resolution//2, 'cpu', align=False).transpose(0,1)
                po = get_grids(self.resolution, self.resolution, 'cpu', align=False).transpose(0,1)
                diffs, coords, coeffs = local_ensemble(pi, po, self.resolution)

                self.diffs   = torch.nn.Parameter(diffs, requires_grad=False)
                self.coords  = torch.nn.Parameter(coords.float(), requires_grad=False)
                self.coeffs  = torch.nn.Parameter(coeffs, requires_grad=False)
                add_dim      = 2
                self.adapter = torch.nn.Sequential(
                    Conv2dLayer(out_channels + add_dim, out_channels // 2, kernel_size=1, activation=activation),
                    Conv2dLayer(out_channels // 2, out_channels, kernel_size=1, activation='linear'),
                )
            elif 'nn_cat' in upsample_mode:
                self.adapter = torch.nn.Sequential(
                    Conv2dLayer(out_channels * 2, out_channels // 4, kernel_size=1, activation=activation),
                    Conv2dLayer(out_channels // 4, out_channels, kernel_size=1, activation='linear'),
                ) 
            elif 'ada' in upsample_mode:
                self.adapter = torch.nn.Sequential(
                    Conv2dLayer(out_channels, 8, kernel_size=1, activation=activation),
                    Conv2dLayer(8, out_channels, kernel_size=1, activation='linear')
                )
                self.adapter[1].weight.data.zero_()
                if 'blur' in upsample_mode:
                    self.blur = Blur()

        self.padding = kernel_size // 2
        self.groups = 2 if use_group else 1
        self.act_gain = bias_act.activation_funcs[activation].def_gain
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)

        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        weight_sizes = [out_channels // self.groups, in_channels, kernel_size, kernel_size]
        if self.mode == '3d':
            weight_sizes += [kernel_size]
        weight = torch.randn(weight_sizes).to(memory_format=memory_format)
        self.weight = torch.nn.Parameter(weight)
        
        if use_noise:
            if self.mode == '2d':
                noise_sizes = [resolution, resolution]
            elif self.mode == '3d': 
                noise_sizes = [resolution, resolution, resolution]
            else:
                raise NotImplementedError('not support for MLP')
            self.register_buffer('noise_const', torch.randn(noise_sizes))  # HACK: for safety reasons
            self.noise_strength = torch.nn.Parameter(torch.zeros([]))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))

        self.magnitude_ema_beta = magnitude_ema_beta
        if magnitude_ema_beta > 0:
            self.register_buffer('w_avg', torch.ones([]))  # TODO: name for compitibality

    def forward(self, x, w, noise_mode='random', fused_modconv=True, gain=1, skip_up=False, input_noise=None, **unused_kwargs):
        assert noise_mode in ['random', 'const', 'none']
        batch_size = x.size(0)
        
        if (self.magnitude_ema_beta > 0):
            if self.training:  # updating EMA.
                with torch.autograd.profiler.record_function('update_magnitude_ema'):
                    magnitude_cur = x.detach().to(torch.float32).square().mean()
                    self.w_avg.copy_(magnitude_cur.lerp(self.w_avg, self.magnitude_ema_beta))
            input_gain = self.w_avg.rsqrt()
            x = x * input_gain

        styles          = self.affine(w)      # Batch x style_dim
        if styles.size(0) < x.size(0):        # for repeating
            assert (x.size(0) // styles.size(0) * styles.size(0) == x.size(0))
            styles = repeat(styles, 'b c -> (b s) c', s=x.size(0) // styles.size(0))
        up              = self.up if not skip_up else 1
        use_default     = (self.upsample_mode == 'default')
        noise           = None
        resample_filter = None
        if use_default and (up > 1):
            resample_filter = self.resample_filter

        if self.use_noise:
            if input_noise is not None:
                noise = input_noise * self.noise_strength
            elif noise_mode == 'random':
                noise_sizes = [x.shape[0], 1, up * x.shape[2], up * x.shape[3]]
                if self.mode == '3d':
                    noise_sizes += [up * x.shape[4]]
                noise = torch.randn(noise_sizes, device=x.device) * self.noise_strength
            elif noise_mode == 'const':
                noise = self.noise_const * self.noise_strength
                if noise.shape[-1] < (up * x.shape[3]):
                    noise = repeat(noise, 'h w -> h (s w)', s=up*x.shape[3]//noise.shape[-1])

        flip_weight = (up == 1)  # slightly faster
        x = modulated_conv2d(
            x=x, weight=self.weight, styles=styles, 
            noise=noise if (use_default and not skip_up) else None, 
            up=up if use_default else 1,
            padding=self.padding, 
            resample_filter=resample_filter, 
            flip_weight=flip_weight, 
            fused_modconv=fused_modconv,
            groups=self.groups,
            mode=self.mode
        )
        
        if (up == 2) and (not use_default):
            resolution = x.size(-1) * 2
            if 'bilinear' in self.upsample_mode:
                x = F.interpolate(x, size=(resolution, resolution), mode='bilinear', align_corners=True)
            elif 'nearest' in self.upsample_mode:
                x = F.interpolate(x, size=(resolution, resolution), mode='nearest')
                x = upfirdn2d.filter2d(x, self.resample_filter)
            elif 'bicubic' in self.upsample_mode:
                x = F.interpolate(x, size=(resolution, resolution), mode='bicubic',  align_corners=True)
            elif 'pixelshuffle' in self.upsample_mode:  # does not have rotation invariance
                x = F.interpolate(x, size=(resolution, resolution), mode='nearest') + torch.pixel_shuffle(self.adapter(x), 2)
                if not 'noblur' in self.upsample_mode:           
                   x = upfirdn2d.filter2d(x, self.resample_filter)
            elif 'nn_cat' in self.upsample_mode:
                x_pad = x.new_zeros(*x.size()[:2], x.size(-2)+2, x.size(-1)+2)
                x_pad[...,1:-1,1:-1] = x
                xl, xu, xd, xr = x_pad[..., 1:-1, :-2], x_pad[..., :-2, 1:-1], x_pad[..., 2:, 1:-1], x_pad[..., 1:-1, 2:]
                x1, x2, x3, x4 = xl + xu, xu + xr, xl + xd, xr + xd
                xb = torch.stack([x1, x2, x3, x4], 2) / 2
                xb = torch.pixel_shuffle(xb.view(xb.size(0), -1, xb.size(-2), xb.size(-1)), 2)
                xa = F.interpolate(x, size=(resolution, resolution), mode='nearest')
                x = xa + self.adapter(torch.cat([xa, xb], 1))
                if not 'noblur' in self.upsample_mode:
                    x = upfirdn2d.filter2d(x, self.resample_filter)
            elif self.upsample_mode == 'liif':   # this is an old version
                x = torch.stack([x[..., self.coords[j,:,:,0].long(), self.coords[j,:,:,1].long()] for j in range(4)], 0)
                d = self.diffs[:, None].type_as(x).repeat(1,batch_size,1,1,1).permute(0,1,4,2,3)
                x = self.adapter(torch.cat([x, d.type_as(x)], 2).reshape(batch_size*4,-1,*x.size()[-2:]))
                x = (x.reshape(4,batch_size,*x.size()[-3:]) * self.coeffs[:,None,None].type_as(x)).sum(0)
            else:
                raise NotImplementedError

        if up == 2:
            if 'ada' in self.upsample_mode:
                x = x + self.adapter(x)
                if 'blur' in self.upsample_mode:
                    x = self.blur(x)

        if (noise is not None) and (not use_default) and (not skip_up):
            x = x.add_(noise.type_as(x))
        
        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act.bias_act(x, self.bias.to(x.dtype), act=self.activation, gain=act_gain, clamp=act_clamp)

        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisLayer3(torch.nn.Module):
    """copy from the stylegan3 codebase with minor changes"""
    def __init__(self,
        w_dim,                          # Intermediate latent (W) dimensionality.
        is_torgb,                       # Is this the final ToRGB layer?
        is_critically_sampled,          # Does this layer use critical sampling?
        use_fp16,                       # Does this layer use FP16?

        # Input & output specifications.
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        in_size,                        # Input spatial size: int or [width, height].
        out_size,                       # Output spatial size: int or [width, height].
        in_sampling_rate,               # Input sampling rate (s).
        out_sampling_rate,              # Output sampling rate (s).
        in_cutoff,                      # Input cutoff frequency (f_c).
        out_cutoff,                     # Output cutoff frequency (f_c).
        in_half_width,                  # Input transition band half-width (f_h).
        out_half_width,                 # Output Transition band half-width (f_h).

        # Hyperparameters.
        kernel_size         = 3,        # Convolution kernel size. Ignored for final the ToRGB layer.
        filter_size         = 6,        # Low-pass filter size relative to the lower resolution when up/downsampling.
        lrelu_upsampling    = 2,        # Relative sampling rate for leaky ReLU. Ignored for final the ToRGB layer.
        use_radial_filters  = False,    # Use radially symmetric downsampling filter? Ignored for critically sampled layers.
        conv_clamp          = 256,      # Clamp the output to [-X, +X], None = disable clamping.
        magnitude_ema_beta  = 0.999,    # Decay rate for the moving average of input magnitudes.

        **unused_kwargs,
    ):
        super().__init__()
        self.w_dim = w_dim
        self.is_torgb = is_torgb
        self.is_critically_sampled = is_critically_sampled
        self.use_fp16 = use_fp16
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_size = np.broadcast_to(np.asarray(in_size), [2])
        self.out_size = np.broadcast_to(np.asarray(out_size), [2])
        self.in_sampling_rate = in_sampling_rate
        self.out_sampling_rate = out_sampling_rate
        self.tmp_sampling_rate = max(in_sampling_rate, out_sampling_rate) * (1 if is_torgb else lrelu_upsampling)
        self.in_cutoff = in_cutoff
        self.out_cutoff = out_cutoff
        self.in_half_width = in_half_width
        self.out_half_width = out_half_width
        self.conv_kernel = 1 if is_torgb else kernel_size
        self.conv_clamp = conv_clamp
        self.magnitude_ema_beta = magnitude_ema_beta

        # Setup parameters and buffers.
        self.affine = FullyConnectedLayer(self.w_dim, self.in_channels, bias_init=1)
        self.weight = torch.nn.Parameter(torch.randn([self.out_channels, self.in_channels, self.conv_kernel, self.conv_kernel]))
        self.bias = torch.nn.Parameter(torch.zeros([self.out_channels]))
        if magnitude_ema_beta > 0:
            self.register_buffer('w_avg', torch.ones([]))

        # Design upsampling filter.
        self.up_factor = int(np.rint(self.tmp_sampling_rate / self.in_sampling_rate))
        assert self.in_sampling_rate * self.up_factor == self.tmp_sampling_rate
        self.up_taps = filter_size * self.up_factor if self.up_factor > 1 and not self.is_torgb else 1
        self.register_buffer('up_filter', self.design_lowpass_filter(
            numtaps=self.up_taps, cutoff=self.in_cutoff, width=self.in_half_width*2, fs=self.tmp_sampling_rate))

        # Design downsampling filter.
        self.down_factor = int(np.rint(self.tmp_sampling_rate / self.out_sampling_rate))
        assert self.out_sampling_rate * self.down_factor == self.tmp_sampling_rate
        self.down_taps = filter_size * self.down_factor if self.down_factor > 1 and not self.is_torgb else 1
        self.down_radial = use_radial_filters and not self.is_critically_sampled
        self.register_buffer('down_filter', self.design_lowpass_filter(
            numtaps=self.down_taps, cutoff=self.out_cutoff, width=self.out_half_width*2, fs=self.tmp_sampling_rate, radial=self.down_radial))

        # Compute padding.
        pad_total = (self.out_size - 1) * self.down_factor + 1 # Desired output size before downsampling.
        pad_total -= (self.in_size + self.conv_kernel - 1) * self.up_factor # Input size after upsampling.
        pad_total += self.up_taps + self.down_taps - 2 # Size reduction caused by the filters.
        pad_lo = (pad_total + self.up_factor) // 2 # Shift sample locations according to the symmetric interpretation (Appendix C.3).
        pad_hi = pad_total - pad_lo
        self.padding = [int(pad_lo[0]), int(pad_hi[0]), int(pad_lo[1]), int(pad_hi[1])]

    def forward(self, x, w, noise_mode='random', force_fp32=False, **unused_kwargs):
        assert noise_mode in ['random', 'const', 'none'] # unused
        misc.assert_shape(x, [None, self.in_channels, int(self.in_size[1]), int(self.in_size[0])])
        misc.assert_shape(w, [x.shape[0], self.w_dim])

        # Track input magnitude.
        if (self.magnitude_ema_beta > 0):
            if self.training:  # updating EMA.
                with torch.autograd.profiler.record_function('update_magnitude_ema'):
                    magnitude_cur = x.detach().to(torch.float32).square().mean()
                    self.w_avg.copy_(magnitude_cur.lerp(self.w_avg, self.magnitude_ema_beta))
            input_gain = self.w_avg.rsqrt()
            x = x * input_gain

        # Execute affine layer.
        styles = self.affine(w)
        if self.is_torgb:
            weight_gain = 1 / np.sqrt(self.in_channels * (self.conv_kernel ** 2))
            styles = styles * weight_gain

        # Execute modulated conv2d.
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32
        x = modulated_conv2d(x=x.to(dtype), weight=self.weight, styles=styles, padding=self.conv_kernel-1, up=1, fused_modconv=True)

        # Execute bias, filtered leaky ReLU, and clamping.
        gain = 1 if self.is_torgb else np.sqrt(2)
        slope = 1 if self.is_torgb else 0.2
        x = filtered_lrelu.filtered_lrelu(x=x, fu=self.up_filter, fd=self.down_filter, b=self.bias.to(x.dtype),
            up=self.up_factor, down=self.down_factor, padding=self.padding, gain=gain, slope=slope, clamp=self.conv_clamp)
        
        # Ensure correct shape and dtype.
        misc.assert_shape(x, [None, self.out_channels, int(self.out_size[1]), int(self.out_size[0])])
        assert x.dtype == dtype
        return x

    @staticmethod
    def design_lowpass_filter(numtaps, cutoff, width, fs, radial=False):
        assert numtaps >= 1

        # Identity filter.
        if numtaps == 1:
            return None

        # Separable Kaiser low-pass filter.
        if not radial:
            f = scipy.signal.firwin(numtaps=numtaps, cutoff=cutoff, width=width, fs=fs)
            return torch.as_tensor(f, dtype=torch.float32)

        # Radially symmetric jinc-based filter.
        x = (np.arange(numtaps) - (numtaps - 1) / 2) / fs
        r = np.hypot(*np.meshgrid(x, x))
        f = scipy.special.j1(2 * cutoff * (np.pi * r)) / (np.pi * r)
        beta = scipy.signal.kaiser_beta(scipy.signal.kaiser_atten(numtaps, width / (fs / 2)))
        w = np.kaiser(numtaps, beta)
        f *= np.outer(w, w)
        f /= np.sum(f)
        return torch.as_tensor(f, dtype=torch.float32)

    def extra_repr(self):
        return '\n'.join([
            f'w_dim={self.w_dim:d}, is_torgb={self.is_torgb},',
            f'is_critically_sampled={self.is_critically_sampled}, use_fp16={self.use_fp16},',
            f'in_sampling_rate={self.in_sampling_rate:g}, out_sampling_rate={self.out_sampling_rate:g},',
            f'in_cutoff={self.in_cutoff:g}, out_cutoff={self.out_cutoff:g},',
            f'in_half_width={self.in_half_width:g}, out_half_width={self.out_half_width:g},',
            f'in_size={list(self.in_size)}, out_size={list(self.out_size)},',
            f'in_channels={self.in_channels:d}, out_channels={self.out_channels:d}'])

#----------------------------------------------------------------------------

@persistence.persistent_class
class ToRGBLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, w_dim=0, kernel_size=1, conv_clamp=None, channels_last=False, mode='2d', **unused):
        super().__init__()
        self.conv_clamp = conv_clamp
        self.mode = mode
        weight_shape = [out_channels, in_channels, kernel_size, kernel_size]
        if mode == '3d':
            weight_shape += [kernel_size]

        if w_dim > 0:
            self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
            memory_format = torch.channels_last if channels_last else torch.contiguous_format
            self.weight = torch.nn.Parameter(torch.randn(weight_shape).to(memory_format=memory_format))
            self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
            self.weight_gain = 1 / np.sqrt(np.prod(weight_shape[1:]))
        
        else:
            assert kernel_size == 1, "does not support larger kernel sizes for now. used in NeRF"
            assert mode != '3d', "does not support 3D convolution for now"

            self.weight = torch.nn.Parameter(torch.Tensor(out_channels, in_channels))
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
            self.weight_gain = 1.

            # initialization
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, w=None, fused_modconv=True):
        if w is not None:
            styles = self.affine(w) * self.weight_gain
            if x.size(0) > styles.size(0):
                assert (x.size(0) // styles.size(0) * styles.size(0) == x.size(0))
                styles = repeat(styles, 'b c -> (b s) c', s=x.size(0) // styles.size(0))
            x = modulated_conv2d(x=x, weight=self.weight, styles=styles, demodulate=False, fused_modconv=fused_modconv, mode=self.mode)
            x = bias_act.bias_act(x, self.bias.to(x.dtype), clamp=self.conv_clamp)
        else:
            if x.ndim == 2:
                x = F.linear(x, self.weight, self.bias)
            else:
                x = F.conv2d(x, self.weight[:,:,None,None], self.bias)
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisBlock(torch.nn.Module):
    def __init__(self,
        in_channels,                        # Number of input channels, 0 = first block.
        out_channels,                       # Number of output channels.
        w_dim,                              # Intermediate latent (W) dimensionality.
        resolution,                         # Resolution of this block.
        img_channels,                       # Number of output color channels.
        is_last,                            # Is this the last block?
        architecture        = 'skip',       # Architecture: 'orig', 'skip', 'resnet'.
        resample_filter     = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp          = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16            = False,        # Use FP16 for this block?
        fp16_channels_last  = False,        # Use channels-last memory format with FP16?
        use_single_layer    = False,        # use only one instead of two synthesis layer
        disable_upsample    = False,
        **layer_kwargs,                     # Arguments for SynthesisLayer.
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.is_last = is_last
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.num_conv = 0
        self.num_torgb = 0

        self.groups = 1
        self.use_single_layer = use_single_layer
        self.margin = layer_kwargs.get('margin', 0)
        self.upsample_mode = layer_kwargs.get('upsample_mode', 'default')
        self.disable_upsample = disable_upsample
        self.mode = layer_kwargs.get('mode', '2d')

        if in_channels == 0:
            const_sizes = [out_channels, resolution, resolution]
            if self.mode == '3d':
                const_sizes = const_sizes + [resolution]
            self.const = torch.nn.Parameter(torch.randn(const_sizes))
        
        if in_channels != 0:
            self.conv0 = util.construct_class_by_name(
                class_name=layer_kwargs.get('layer_name', "training.networks.SynthesisLayer"),
                in_channels=in_channels, out_channels=out_channels, 
                w_dim=w_dim, resolution=resolution, 
                up=2 if (not disable_upsample) else 1,
                resample_filter=resample_filter, conv_clamp=conv_clamp, 
                channels_last=self.channels_last, **layer_kwargs)
            self.num_conv += 1

        if not self.use_single_layer:
            self.conv1 = util.construct_class_by_name(
                    class_name=layer_kwargs.get('layer_name', "training.networks.SynthesisLayer"),
                    in_channels=out_channels, out_channels=out_channels, 
                    w_dim=w_dim, resolution=resolution,
                    conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs)
            self.num_conv += 1

        if is_last or architecture == 'skip':
            self.torgb = ToRGBLayer(
                out_channels, img_channels, w_dim=w_dim,
                conv_clamp=conv_clamp, channels_last=self.channels_last,
                groups=self.groups, mode=self.mode)
            self.num_torgb += 1
            
        if in_channels != 0 and architecture == 'resnet':
            self.skip = Conv2dLayer(
                in_channels, out_channels, kernel_size=1, bias=False, up=2,
                resample_filter=resample_filter, 
                channels_last=self.channels_last,
                mode=self.mode)

    def forward(self, x, img, ws, force_fp32=False, fused_modconv=None, add_on=None, block_noise=None, disable_rgb=False, **layer_kwargs):
        misc.assert_shape(ws, [None, self.num_conv + self.num_torgb, self.w_dim])
        w_iter = iter(ws.unbind(dim=1))
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format
        if fused_modconv is None:
            with misc.suppress_tracer_warnings(): # this value will be treated as a constant
                fused_modconv = (not self.training) and (dtype == torch.float32 or int(x.shape[0]) == 1)

        # Input.
        if self.in_channels == 0:
            x = self.const.to(dtype=dtype, memory_format=memory_format)
            x = x.unsqueeze(0).expand(ws.shape[0], *x.size())
        else:
            x = x.to(dtype=dtype, memory_format=memory_format)

        # Main layers.
        if add_on is not None:
            add_on = add_on.to(dtype=dtype, memory_format=memory_format)

        if self.in_channels == 0:
            if not self.use_single_layer:
                layer_kwargs['input_noise'] = block_noise[:,1:2] if block_noise is not None else None
                x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
        
        elif self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            layer_kwargs['input_noise'] = block_noise[:,0:1] if block_noise is not None else None
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            if not self.use_single_layer:
                layer_kwargs['input_noise'] = block_noise[:,1:2] if block_noise is not None else None
                x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, gain=np.sqrt(0.5), **layer_kwargs)
            x = y.add_(x)
        else:
            layer_kwargs['input_noise'] = block_noise[:,0:1] if block_noise is not None else None
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            if not self.use_single_layer:
                layer_kwargs['input_noise'] = block_noise[:,1:2] if block_noise is not None else None
                x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)

        # ToRGB.
        if img is not None:
            if img.size(-1) * 2 == x.size(-1):
                if (self.upsample_mode == 'bilinear_all') or (self.upsample_mode == 'bilinear_ada'):
                    img = F.interpolate(img, scale_factor=2, mode='bilinear', align_corners=True)
                else:
                    img = upfirdn2d.upsample2d(img, self.resample_filter)   # this is upsampling. Not sure about details and why they do this..
            elif img.size(-1) == x.size(-1):
                pass
            else:
                raise NotImplementedError

        if self.is_last or self.architecture == 'skip':
            if not disable_rgb:
                y = x if add_on is None else x + add_on
                y = self.torgb(y, next(w_iter), fused_modconv=fused_modconv)
                y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
                img = img.add_(y) if img is not None else y
            else:
                img = None

        assert x.dtype == dtype
        assert img is None or img.dtype == torch.float32
        return x, img

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisBlock3(torch.nn.Module):
    def __init__(self,
        in_channels,                        # Number of input channels, 0 = first block.
        out_channels,                       # Number of output channels.
        w_dim,                              # Intermediate latent (W) dimensionality.
        resolution,                         # Resolution of this block.
        img_channels,                       # Number of output color channels.
        block_id,
        stylegan3_hyperam,
        use_fp16            = False,        # Use FP16 for this block?
        **layer_kwargs,                     # Arguments for SynthesisLayer.
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.num_conv = 0
        self.num_torgb = 0
        self.use_fp16 = use_fp16

        is_critically_sampled = block_id == (len(stylegan3_hyperam['sampling_rates'][:-1]) // 2 - 1)
        sizes, sampling_rates, cutoffs, half_widths = \
            stylegan3_hyperam['sizes'], stylegan3_hyperam['sampling_rates'], \
            stylegan3_hyperam['cutoffs'], stylegan3_hyperam['half_widths']

        # each block has two layer
        prev = max(block_id * 2 - 1, 0)
        curr = block_id * 2 
        self.conv0 = util.construct_class_by_name(
                class_name=layer_kwargs.get('layer_name', "training.networks.SynthesisLayer3"),
                w_dim=self.w_dim, 
                is_torgb=False, 
                is_critically_sampled=is_critically_sampled, 
                use_fp16=use_fp16,
                in_channels=in_channels, 
                out_channels=out_channels,
                in_size=int(sizes[prev]), 
                out_size=int(sizes[curr]),
                in_sampling_rate=int(sampling_rates[prev]), 
                out_sampling_rate=int(sampling_rates[curr]),
                in_cutoff=cutoffs[prev], 
                out_cutoff=cutoffs[curr],
                in_half_width=half_widths[prev], 
                out_half_width=half_widths[curr],
                use_radial_filters=True,
                **layer_kwargs)
        self.num_conv += 1

        prev = block_id * 2
        curr = block_id * 2 + 1 
        self.conv1 = util.construct_class_by_name(
                class_name=layer_kwargs.get('layer_name', "training.networks.SynthesisLayer3"),
                w_dim=self.w_dim, 
                is_torgb=False, 
                is_critically_sampled=is_critically_sampled, 
                use_fp16=use_fp16,
                in_channels=out_channels, 
                out_channels=out_channels,
                in_size=int(sizes[prev]), 
                out_size=int(sizes[curr]),
                in_sampling_rate=int(sampling_rates[prev]), 
                out_sampling_rate=int(sampling_rates[curr]),
                in_cutoff=cutoffs[prev], 
                out_cutoff=cutoffs[curr],
                in_half_width=half_widths[prev], 
                out_half_width=half_widths[curr],
                use_radial_filters=True,
                **layer_kwargs)
        self.num_conv += 1

        # toRGB layer (used for progressive growing)
        self.torgb = ToRGBLayer(out_channels, img_channels, w_dim=w_dim)
        self.num_torgb += 1

    def forward(self, x, img, ws,  force_fp32=False, add_on=None, disable_rgb=False, **layer_kwargs):
        w_iter = iter(ws.unbind(dim=1))
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.contiguous_format
        
        # Main layers.
        x = x.to(dtype=dtype, memory_format=memory_format)
        if add_on is not None:
            add_on = add_on.to(dtype=dtype, memory_format=memory_format)

        x = self.conv0(x, next(w_iter), **layer_kwargs)
        x = self.conv1(x, next(w_iter), **layer_kwargs)

        assert img is None, "currently not support."
        if not disable_rgb:
            y = x if add_on is None else x + add_on
            y = self.torgb(y, next(w_iter), fused_modconv=True)
            y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            img = y
        
        assert x.dtype == dtype
        assert img is None or img.dtype == torch.float32
        return x, img

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisNetwork(torch.nn.Module):
    def __init__(self,
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output image resolution.
        img_channels,               # Number of color channels.
        channel_base    = 1,        # Overall multiplier for the number of channels.
        channel_max     = 512,      # Maximum number of channels in any layer.
        num_fp16_res    = 0,        # Use FP16 for the N highest resolutions.
        **block_kwargs,             # Arguments for SynthesisBlock.
    ):
        assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0
        super().__init__()
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(2, self.img_resolution_log2 + 1)]

        channel_base = int(channel_base * 32768)
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)
        self.channels_dict = channels_dict

        self.num_ws = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res // 2] if res > 4 else 0
            out_channels = channels_dict[res]
            use_fp16 = (res >= fp16_resolution)
            is_last = (res == self.img_resolution)
            block = util.construct_class_by_name(
                class_name=block_kwargs.get('block_name', "training.networks.SynthesisBlock"),
                in_channels=in_channels, out_channels=out_channels, w_dim=w_dim, resolution=res,
                img_channels=img_channels, is_last=is_last, use_fp16=use_fp16, **block_kwargs)

            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb
            setattr(self, f'b{res}', block)

    def forward(self, ws, **block_kwargs):
        block_ws = []

        # this part is to slice the style matrices (W) to each layer (conv/RGB)
        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
            ws = ws.to(torch.float32)
            w_idx = 0
            for res in self.block_resolutions:
                block = getattr(self, f'b{res}')
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv

        x = img = None
        for res, cur_ws in zip(self.block_resolutions, block_ws):
            block = getattr(self, f'b{res}')
            x, img = block(x, img, cur_ws, **block_kwargs)
        return img

    def get_current_resolution(self):
        return [self.img_resolution]   # For compitibility

#----------------------------------------------------------------------------

@persistence.persistent_class
class Generator(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        synthesis_kwargs    = {},   # Arguments for SynthesisNetwork.
        encoder_kwargs      = {},   # Arguments for Encoder (optional)
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.synthesis = util.construct_class_by_name(
            class_name=synthesis_kwargs.get('module_name', "training.networks.SynthesisNetwork"),
            w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels, **synthesis_kwargs)
        self.num_ws  = self.synthesis.num_ws
        self.mapping = None
        self.encoder = None

        if len(mapping_kwargs) > 0:   # Use mapping network
            self.mapping = util.construct_class_by_name(
                class_name=mapping_kwargs.get('module_name', "training.networks.MappingNetwork"),
                z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)
        
        if len(encoder_kwargs) > 0:   # Use Image-Encoder
            encoder_kwargs['model_kwargs'].update({'num_ws': self.num_ws, 'w_dim': self.w_dim})
            self.encoder = util.construct_class_by_name(
               img_resolution=img_resolution, 
               img_channels=img_channels,
               **encoder_kwargs) 
        
    def forward(self, z=None, c=None, styles=None, truncation_psi=1, truncation_cutoff=None, img=None, **synthesis_kwargs):
        if styles is None:
            assert z is not None
            if (self.encoder is not None) and (img is not None):  #TODO: debug
                outputs = self.encoder(img)
                ws = outputs['ws']
                if ('camera' in outputs) and ('camera_mode' not in synthesis_kwargs):
                    synthesis_kwargs['camera_RT'] = outputs['camera']
            else:
                ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, **synthesis_kwargs)
        else:
            ws = styles

        img = self.synthesis(ws, **synthesis_kwargs)
        return img

    def get_final_output(self, *args, **kwargs):
        img = self.forward(*args, **kwargs)
        if isinstance(img, list):
            return img[-1]
        elif isinstance(img, dict):
            return img['img']
        return img

#----------------------------------------------------------------------------

@persistence.persistent_class
class DiscriminatorBlock(torch.nn.Module):
    def __init__(self,
        in_channels,                        # Number of input channels, 0 = first block.
        tmp_channels,                       # Number of intermediate channels.
        out_channels,                       # Number of output channels.
        resolution,                         # Resolution of this block.
        img_channels,                       # Number of input color channels.
        first_layer_idx,                    # Index of the first layer.
        architecture        = 'resnet',     # Architecture: 'orig', 'skip', 'resnet'.
        activation          = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
        resample_filter     = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp          = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16            = False,        # Use FP16 for this block?
        fp16_channels_last  = False,        # Use channels-last memory format with FP16?
        freeze_layers       = 0,            # Freeze-D: Number of layers to freeze.
    ):
        assert in_channels in [0, tmp_channels]
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.resolution = resolution
        self.img_channels = img_channels
        self.first_layer_idx = first_layer_idx
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))

        self.num_layers = 0
        def trainable_gen():
            while True:
                layer_idx = self.first_layer_idx + self.num_layers
                trainable = (layer_idx >= freeze_layers)
                self.num_layers += 1
                yield trainable
        trainable_iter = trainable_gen()

        if in_channels == 0 or architecture == 'skip':
            self.fromrgb = Conv2dLayer(img_channels, tmp_channels, kernel_size=1, activation=activation,
                trainable=next(trainable_iter), conv_clamp=conv_clamp, channels_last=self.channels_last)

        self.conv0 = Conv2dLayer(tmp_channels, tmp_channels, kernel_size=3, activation=activation,
            trainable=next(trainable_iter), conv_clamp=conv_clamp, channels_last=self.channels_last)

        self.conv1 = Conv2dLayer(tmp_channels, out_channels, kernel_size=3, activation=activation, down=2,
            trainable=next(trainable_iter), resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=self.channels_last)

        if architecture == 'resnet':
            self.skip = Conv2dLayer(tmp_channels, out_channels, kernel_size=1, bias=False, down=2,
                trainable=next(trainable_iter), resample_filter=resample_filter, channels_last=self.channels_last)

    def forward(self, x, img, force_fp32=False, downsampler=None):
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format

        # Input.
        if x is not None:
            misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution])
            x = x.to(dtype=dtype, memory_format=memory_format)

        # FromRGB.
        if self.in_channels == 0 or self.architecture == 'skip':
            misc.assert_shape(img, [None, self.img_channels, self.resolution, self.resolution])
            img = img.to(dtype=dtype, memory_format=memory_format)
            y = self.fromrgb(img)
            x = x + y if x is not None else y
            if self.architecture != 'skip':
                img = None
            elif downsampler is not None:
                img = downsampler(img, 2)
            else:
                img = upfirdn2d.downsample2d(img, self.resample_filter)

        # Main layers.
        if self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x)
            x = self.conv1(x, gain=np.sqrt(0.5))
            x = y.add_(x)
        else:
            x = self.conv0(x)
            x = self.conv1(x)

        assert x.dtype == dtype
        return x, img

#----------------------------------------------------------------------------

@persistence.persistent_class
class MinibatchStdLayer(torch.nn.Module):
    def __init__(self, group_size, num_channels=1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x):
        N, C, H, W = x.shape
        with misc.suppress_tracer_warnings(): # as_tensor results are registered as constants
            G = torch.min(torch.as_tensor(self.group_size), torch.as_tensor(N)) if self.group_size is not None else N
        F = self.num_channels
        c = C // F

        y = x.reshape(G, -1, F, c, H, W)    # [GnFcHW] Split minibatch N into n groups of size G, and channels C into F groups of size c.
        y = y - y.mean(dim=0)               # [GnFcHW] Subtract mean over group.
        y = y.square().mean(dim=0)          # [nFcHW]  Calc variance over group.
        y = (y + 1e-8).sqrt()               # [nFcHW]  Calc stddev over group.
        y = y.mean(dim=[2,3,4])             # [nF]     Take average over channels and pixels.
        y = y.reshape(-1, F, 1, 1)          # [nF11]   Add missing dimensions.
        y = y.repeat(G, 1, H, W)            # [NFHW]   Replicate over group and pixels.
        x = torch.cat([x, y], dim=1)        # [NCHW]   Append to input as new channels.
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class DiscriminatorEpilogue(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        cmap_dim,                       # Dimensionality of mapped conditioning label, 0 = no label.
        resolution,                     # Resolution of this block.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        mbstd_group_size    = 4,        # Group size for the minibatch standard deviation layer, None = entire minibatch.
        mbstd_num_channels  = 1,        # Number of features for the minibatch standard deviation layer, 0 = disable.
        activation          = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        conv_clamp          = None,     # Clamp the output of convolution layers to +-X, None = disable clamping.
        final_channels      = 1,        # for classification it is always 1.
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.final_channels = final_channels
        self.cmap_dim = cmap_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.architecture = architecture

        if architecture == 'skip':
            self.fromrgb = Conv2dLayer(img_channels, in_channels, kernel_size=1, activation=activation)
        self.mbstd = MinibatchStdLayer(group_size=mbstd_group_size, num_channels=mbstd_num_channels) if mbstd_num_channels > 0 else None
        self.conv = Conv2dLayer(in_channels + mbstd_num_channels, in_channels, kernel_size=3, activation=activation, conv_clamp=conv_clamp)
        self.fc = FullyConnectedLayer(in_channels * (resolution ** 2), in_channels, activation=activation)
        self.out = FullyConnectedLayer(in_channels, final_channels if cmap_dim == 0 else cmap_dim)

    def forward(self, x, img, cmap, force_fp32=False):
        misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution]) # [NCHW]
        _ = force_fp32 # unused
        dtype = torch.float32
        memory_format = torch.contiguous_format

        # FromRGB.
        x = x.to(dtype=dtype, memory_format=memory_format)
        if self.architecture == 'skip':
            misc.assert_shape(img, [None, self.img_channels, self.resolution, self.resolution])
            img = img.to(dtype=dtype, memory_format=memory_format)
            x = x + self.fromrgb(img)

        # Main layers.
        if self.mbstd is not None:
            x = self.mbstd(x)
        x = self.conv(x)
        x = self.fc(x.flatten(1))
        x = self.out(x)

        # Conditioning.
        if self.cmap_dim > 0:
            misc.assert_shape(cmap, [None, self.cmap_dim])
            x = (x * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))

        assert x.dtype == dtype
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class Discriminator(torch.nn.Module):   # The original StyleGAN2 discriminator
    def __init__(self,
        c_dim,                          # Conditioning label (C) dimensionality.
        img_resolution,                 # Input resolution.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        channel_base        = 32768,    # Overall multiplier for the number of channels.
        channel_max         = 512,      # Maximum number of channels in any layer.
        num_fp16_res        = 0,        # Use FP16 for the N highest resolutions.
        conv_clamp          = None,     # Clamp the output of convolution layers to +-X, None = disable clamping.
        cmap_dim            = None,     # Dimensionality of mapped conditioning label, None = default.
        block_kwargs        = {},       # Arguments for DiscriminatorBlock.
        mapping_kwargs      = {},       # Arguments for MappingNetwork.
        epilogue_kwargs     = {},       # Arguments for DiscriminatorEpilogue.
    ):
        super().__init__()
        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        if cmap_dim is None:
            cmap_dim = channels_dict[4]
        if c_dim == 0:
            cmap_dim = 0

        common_kwargs = dict(img_channels=img_channels, architecture=architecture, conv_clamp=conv_clamp)
        cur_layer_idx = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res] if res < img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            use_fp16 = (res >= fp16_resolution)
            block = DiscriminatorBlock(in_channels, tmp_channels, out_channels, resolution=res,
                first_layer_idx=cur_layer_idx, use_fp16=use_fp16, **block_kwargs, **common_kwargs)
            setattr(self, f'b{res}', block)
            cur_layer_idx += block.num_layers
        if c_dim > 0:
            self.mapping = MappingNetwork(z_dim=0, c_dim=c_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None, **mapping_kwargs)
        self.b4 = DiscriminatorEpilogue(channels_dict[4], cmap_dim=cmap_dim, resolution=4, **epilogue_kwargs, **common_kwargs)

    def forward(self, img, c, **block_kwargs):
        x = None
        if isinstance(img, dict):
            img = img['img']
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            x, img = block(x, img, **block_kwargs)

        cmap = None
        if self.c_dim > 0:
            cmap = self.mapping(None, c)
        x = self.b4(x, img, cmap)
        return x

#----------------------------------------------------------------------------
# encoders maybe used for inversion (not cleaned)

@persistence.persistent_class
class EncoderResBlock(torch.nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.conv1 = Conv2dLayer(in_channel, in_channel, 3, activation='lrelu')
        self.conv2 = Conv2dLayer(in_channel, out_channel, 3, down=2, activation='lrelu')
        self.skip  = Conv2dLayer(in_channel, out_channel, 1, down=2, activation='linear', bias=False)

    def forward(self, input):
        out  = self.conv1(input)
        out  = self.conv2(out)
        skip = self.skip(input)
        out  = (out + skip) / math.sqrt(2)
        return out


@persistence.persistent_class
class EqualConv2d(torch.nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()
        new_scale   = 1.0
        self.weight = torch.nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size) * new_scale
        )
        self.scale   = 1 / math.sqrt(in_channel * kernel_size ** 2)
        self.stride  = stride
        self.padding = padding
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_channel))
        else:
            self.bias = None

    def forward(self, input):
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )
        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
            f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
        )


@persistence.persistent_class
class Encoder(torch.nn.Module):
    def __init__(self, size, n_latents, w_dim=512, add_dim=0, **unused):
        super().__init__()
        
        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256,
            128: 128,
            256: 64,
            512: 32,
            1024: 16
        }        
        
        self.w_dim = w_dim
        self.add_dim = add_dim
        log_size = int(math.log(size, 2))
        
        self.n_latents = n_latents
        convs = [Conv2dLayer(3, channels[size], 1)]

        in_channel = channels[size]
        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            convs.append(EncoderResBlock(in_channel, out_channel))
            in_channel = out_channel
   
        self.convs = torch.nn.Sequential(*convs)
        self.projector = EqualConv2d(in_channel, self.n_latents*self.w_dim + add_dim, 4, padding=0, bias=False)

    def forward(self, input):
        out = self.convs(input)
        out = self.projector(out)
        pws, pcm = out[:, :-2], out[:, -2:]
        pws = pws.view(len(input), self.n_latents, self.w_dim)
        pcm = pcm.view(len(input), self.add_dim)
        return pws, pcm


@persistence.persistent_class
class ResNetEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        import torchvision
        resnet_net = torchvision.models.resnet18(pretrained=True)
        modules = list(resnet_net.children())[:-1]
        self.convs = torch.nn.Sequential(*modules)
        self.requires_grad_(True)
        self.train()

    def preprocess_tensor(self, x):
        x = F.interpolate(x, size=(224, 224), mode='bicubic', align_corners=False)
        return x

    def forward(self, input):
        out = self.convs(self.preprocess_tensor(input))
        return out[:, :, 0, 0]


@persistence.persistent_class
class CLIPEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        import clip
        clip_net, _ = clip.load('ViT-B/32', device='cpu', jit=False)
        self.encoder = clip_net.visual
        for p in self.encoder.parameters():
            p.requires_grad_(True)

    def preprocess_tensor(self, x):
        import PIL.Image
        import torchvision.transforms.functional as TF
        x = x * 0.5 + 0.5  # mapping to 0~1
        x = TF.resize(x, size=224, interpolation=PIL.Image.BICUBIC)
        x = TF.normalize(x, (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        return x

    def forward(self, input):
        out = self.encoder(self.preprocess_tensor(input))
        return out


# --------------------------------------------------------------------------------------------------- #
# VolumeGAN thanks https://gist.github.com/justimyhxu/a96f5ac25480d733f3151adb8142d706

@persistence.persistent_class
class InstanceNormLayer3d(torch.nn.Module):
    """Implements instance normalization layer."""
    def __init__(self, num_features, epsilon=1e-8, affine=False):
        super().__init__()
        self.eps = epsilon
        self.affine = affine
        if self.affine:
            self.weight = torch.nn.Parameter(torch.Tensor(1, num_features,1,1,1))
            self.bias = torch.nn.Parameter(torch.Tensor(1, num_features,1,1,1))
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def forward(self, x, weight=None, bias=None):
        x = x - torch.mean(x, dim=[2, 3, 4], keepdim=True)
        norm = torch.sqrt(
            torch.mean(x**2, dim=[2, 3, 4], keepdim=True) + self.eps)
        x = x / norm
        isnot_input_none = weight is not None and bias is not None
        assert (isnot_input_none and not self.affine) or (not isnot_input_none and self.affine)
        if self.affine:
            x = x*self.weight + self.bias
        else:
            x = x*weight + bias
        return x

@persistence.persistent_class      
class FeatureVolume(torch.nn.Module):
    def __init__(
        self,
        feat_res=32,
        init_res=4,
        base_channels=256,
        output_channels=32,
        z_dim=256,
        use_mapping=True,
        **kwargs
    ):
        super().__init__()
        self.num_stages = int(np.log2(feat_res // init_res)) + 1
        self.use_mapping = use_mapping

        self.const = nn.Parameter(
            torch.ones(1, base_channels, init_res, init_res, init_res))
        inplanes = base_channels
        outplanes = base_channels

        self.stage_channels = []
        for i in range(self.num_stages):
            conv = nn.Conv3d(inplanes,
                             outplanes,
                             kernel_size=(3, 3, 3),
                             padding=(1, 1, 1))
            self.stage_channels.append(outplanes)
            self.add_module(f'layer{i}', conv)
            instance_norm = InstanceNormLayer3d(num_features=outplanes, affine=not use_mapping)

            self.add_module(f'instance_norm{i}', instance_norm)
            inplanes = outplanes
            outplanes = max(outplanes // 2, output_channels)
            if i == self.num_stages - 1:
                outplanes = output_channels

        if self.use_mapping:
            self.mapping_network = CustomMappingNetwork(
                z_dim, 256,
                sum(self.stage_channels) * 2)
        self.upsample = UpsamplingLayer()
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, z, **kwargs):
        if self.use_mapping:
            scales, shifts, style = self.mapping_network(z)

        x = self.const.repeat(z.shape[0], 1, 1, 1, 1)
        for idx in range(self.num_stages):
            if idx != 0:
                x = self.upsample(x)
            conv_layer = self.__getattr__(f'layer{idx}')
            x = conv_layer(x)
            instance_norm = self.__getattr__(f'instance_norm{idx}')
            if self.use_mapping:
                scale = scales[:, sum(self.stage_channels[:idx]):sum(self.stage_channels[:idx + 1])]
                shift = shifts[:, sum(self.stage_channels[:idx]):sum(self.stage_channels[:idx + 1])]
                scale = scale.view(scale.shape + (1, 1, 1))
                shift = shift.view(shift.shape + (1, 1, 1))
            else:
                scale, shift = None, None
            x = instance_norm(x, weight=scale, bias=shift)
            x = self.lrelu(x)

        return x