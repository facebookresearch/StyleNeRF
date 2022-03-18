# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from bdb import set_trace
import copy
from email import generator
import imp
import math
from platform import architecture


import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import grad
from training.networks import *
from dnnlib.camera import *
from dnnlib.geometry import (
    positional_encoding, upsample, downsample
)
from dnnlib.util import dividable, hash_func, EasyDict
from torch_utils.ops.hash_sample import hash_sample
from torch_utils.ops.grid_sample_gradfix import grid_sample
from torch_utils.ops.nerf_utils import topp_masking
from einops import repeat, rearrange


# --------------------------------- basic modules ------------------------------------------- #
@persistence.persistent_class
class Style2Layer(nn.Module):
    def __init__(self, 
        in_channels, 
        out_channels, 
        w_dim, 
        activation='lrelu', 
        resample_filter=[1,3,3,1],
        magnitude_ema_beta = -1,           # -1 means not using magnitude ema
        **unused_kwargs):

        # simplified version of SynthesisLayer 
        # no noise, kernel size forced to be 1x1, used in NeRF block
        super().__init__()
        self.activation = activation
        self.conv_clamp = None
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = 0
        self.act_gain = bias_act.activation_funcs[activation].def_gain
        self.w_dim = w_dim
        self.in_features = in_channels
        self.out_features = out_channels
        memory_format = torch.contiguous_format

        if w_dim > 0:
            self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
            self.weight = torch.nn.Parameter(
               torch.randn([out_channels, in_channels, 1, 1]).to(memory_format=memory_format))
            self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        
        else:
            self.weight = torch.nn.Parameter(torch.Tensor(out_channels, in_channels))
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
            self.weight_gain = 1.

            # initialization
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

        self.magnitude_ema_beta = magnitude_ema_beta
        if magnitude_ema_beta > 0:
            self.register_buffer('w_avg', torch.ones([]))

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, style={}'.format(
            self.in_features, self.out_features, self.w_dim
        )

    def forward(self, x, w=None, fused_modconv=None, gain=1, up=1, **unused_kwargs):
        flip_weight = True # (up == 1) # slightly faster HACK
        act = self.activation

        if (self.magnitude_ema_beta > 0):
            if self.training:  # updating EMA.
                with torch.autograd.profiler.record_function('update_magnitude_ema'):
                    magnitude_cur = x.detach().to(torch.float32).square().mean()
                    self.w_avg.copy_(magnitude_cur.lerp(self.w_avg, self.magnitude_ema_beta))
            input_gain = self.w_avg.rsqrt()
            x = x * input_gain

        if fused_modconv is None:
            with misc.suppress_tracer_warnings(): # this value will be treated as a constant
                fused_modconv = not self.training
        
        if self.w_dim > 0:           # modulated convolution
            assert x.ndim == 4,  "currently not support modulated MLP"
            styles = self.affine(w)      # Batch x style_dim
            if x.size(0) > styles.size(0):
                styles = repeat(styles, 'b c -> (b s) c', s=x.size(0) // styles.size(0))
            
            x = modulated_conv2d(x=x, weight=self.weight, styles=styles, noise=None, up=up,
                padding=self.padding, resample_filter=self.resample_filter, 
                flip_weight=flip_weight, fused_modconv=fused_modconv)
            act_gain = self.act_gain * gain
            act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
            x = bias_act.bias_act(x, self.bias.to(x.dtype), act=act, gain=act_gain, clamp=act_clamp)
        
        else:
            if x.ndim == 2:  # MLP mode
                x = F.relu(F.linear(x, self.weight, self.bias.to(x.dtype)))
            else:
                x = F.relu(F.conv2d(x, self.weight[:,:,None, None], self.bias))
                # x = bias_act.bias_act(x, self.bias.to(x.dtype), act='relu')
        return x


@persistence.persistent_class
class SDFDensityLaplace(nn.Module):  # alpha * Laplace(loc=0, scale=beta).cdf(-sdf)
    def __init__(self, params_init={}, noise_std=0.0, beta_min=0.001, exp_beta=False):
        super().__init__()
        self.noise_std = noise_std
        for p in params_init:
            param = nn.Parameter(torch.tensor(params_init[p]))
            setattr(self, p, param)
        self.beta_min = beta_min
        self.exp_beta = exp_beta
        if (exp_beta == 'upper') or exp_beta:
            self.register_buffer("steps", torch.scalar_tensor(0).float())

    def density_func(self, sdf, beta=None):
        if beta is None:
            beta = self.get_beta()
        alpha = 1 / beta
        return alpha * (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() / beta))  # TODO: need abs maybe, not sure

    def get_beta(self):
        if self.exp_beta == 'upper':
            beta_upper = 0.12 * torch.exp(-0.003 * (self.steps / 1e3))
            beta = min(self.beta.abs(), beta_upper) + self.beta_min
        elif self.exp_beta:
            if self.steps < 500000:
                beta = self.beta.abs() + self.beta_min
            else:
                beta = self.beta.abs().detach() + self.beta_min
        else:
            beta = self.beta.abs() + self.beta_min
        return beta

    def set_steps(self, steps):
        if hasattr(self, "steps"):
            self.steps = self.steps * 0 + steps

# ------------------------------------------------------------------------------------------- #

@persistence.persistent_class
class NeRFBlock(nn.Module):
    ''' 
    Predicts volume density and color from 3D location, viewing
    direction, and latent code z.
    '''
    # dimensions
    input_dim            = 3
    w_dim                = 512   # style latent
    z_dim                = 0     # input latent
    rgb_out_dim          = 128
    hidden_size          = 128
    n_blocks             = 8
    img_channels         = 3
    magnitude_ema_beta   = -1
    disable_latents      = False
    max_batch_size       = 2 ** 18
    shuffle_factor       = 1
    implementation       = 'batch_reshape'  # option: [flatten_2d, batch_reshape]

    # architecture settings
    activation           = 'lrelu'
    use_skip             = False 
    use_viewdirs         = False
    add_rgb              = False
    predict_rgb          = False
    inverse_sphere       = False
    merge_sigma_feat     = False   # use one MLP for sigma and features
    no_sigma             = False   # do not predict sigma, only output features
    
    tcnn_backend         = False
    use_style            = None 
    use_normal           = False
    use_sdf              = None
    volsdf_exp_beta      = False
    normalized_feat      = False
    final_sigmoid_act    = False

    # positional encoding inpuut
    use_pos              = False
    n_freq_posenc        = 10
    n_freq_posenc_views  = 4
    downscale_p_by       = 1
    gauss_dim_pos        = 20 
    gauss_dim_view       = 4 
    gauss_std            = 10.
    positional_encoding  = "normal"

    def __init__(self, nerf_kwargs):
        super().__init__()
        for key in nerf_kwargs:
            if hasattr(self, key):
                setattr(self, key, nerf_kwargs[key])

        self.sdf_mode = self.use_sdf
        self.use_sdf  = self.use_sdf is not None
        if self.use_sdf == 'volsdf':
            self.density_transform = SDFDensityLaplace(
                params_init={'beta': 0.1}, 
                beta_min=0.0001, 
                exp_beta=self.volsdf_exp_beta)

        # ----------- input module -------------------------
        D = self.input_dim if not self.inverse_sphere else self.input_dim + 1
        if self.positional_encoding == 'gauss':
            rng = np.random.RandomState(2021)
            B_pos  = self.gauss_std * torch.from_numpy(rng.randn(D, self.gauss_dim_pos * D)).float()
            B_view = self.gauss_std * torch.from_numpy(rng.randn(3, self.gauss_dim_view * 3)).float()
            self.register_buffer("B_pos", B_pos)
            self.register_buffer("B_view", B_view)
            dim_embed = D * self.gauss_dim_pos * 2
            dim_embed_view = 3 * self.gauss_dim_view * 2
        elif self.positional_encoding == 'normal':
            dim_embed = D * self.n_freq_posenc * 2
            dim_embed_view = 3 * self.n_freq_posenc_views * 2
        else:  # not using positional encoding
            dim_embed, dim_embed_view = D, 3

        if self.use_pos:
            dim_embed, dim_embed_view = dim_embed + D, dim_embed_view + 3

        self.dim_embed = dim_embed
        self.dim_embed_view = dim_embed_view

        # ------------ Layers --------------------------
        assert not (self.add_rgb and self.predict_rgb), "only one could be achieved"
        assert not ((self.use_viewdirs or self.use_normal) and (self.merge_sigma_feat or self.no_sigma)), \
            "merged MLP does not support."
        
        if self.disable_latents:
            w_dim = 0
        elif self.z_dim > 0:  # if input global latents, disable using style vectors
            w_dim, dim_embed, dim_embed_view = 0, dim_embed + self.z_dim, dim_embed_view + self.z_dim
        else:
            w_dim = self.w_dim

        final_in_dim = self.hidden_size
        if self.use_normal:
            final_in_dim += D

        final_out_dim = self.rgb_out_dim * self.shuffle_factor
        if self.merge_sigma_feat:
            final_out_dim += self.shuffle_factor  # predicting sigma
        if self.add_rgb:
            final_out_dim += self.img_channels

        # start building the model
        if self.tcnn_backend:
            try:
                import tinycudann as tcnn
            except ImportError:
                raise ImportError("This sample requires the tiny-cuda-nn extension for PyTorch.")

            assert self.merge_sigma_feat and (not self.predict_rgb) and (not self.add_rgb)
            assert w_dim == 0, "do not use any modulating inputs"
            
            tcnn_config  = {"otype": "FullyFusedMLP", "activation": "ReLU", "output_activation": "None", "n_neurons": 64, "n_hidden_layers": 1}
            self.network = tcnn.Network(dim_embed, final_out_dim, tcnn_config)
            self.num_ws  = 0
            
        else:
            self.fc_in  = Style2Layer(dim_embed, self.hidden_size, w_dim, activation=self.activation)
            self.num_ws = 1
            self.skip_layer = self.n_blocks // 2 - 1 if self.use_skip else None      
            if self.n_blocks > 1:
                self.blocks = nn.ModuleList([
                    Style2Layer(
                        self.hidden_size if i != self.skip_layer else self.hidden_size + dim_embed, 
                        self.hidden_size, 
                        w_dim, activation=self.activation,
                        magnitude_ema_beta=self.magnitude_ema_beta)
                    for i in range(self.n_blocks - 1)])
                self.num_ws += (self.n_blocks - 1)

            if not (self.merge_sigma_feat or self.no_sigma):
                self.sigma_out = ToRGBLayer(self.hidden_size, self.shuffle_factor, w_dim, kernel_size=1)
                self.num_ws += 1
            self.feat_out = ToRGBLayer(final_in_dim, final_out_dim, w_dim, kernel_size=1)
            if (self.z_dim == 0 and (not self.disable_latents)):
                self.num_ws += 1
            else:
                self.num_ws = 0        
            
            if self.use_viewdirs:
                assert self.predict_rgb, "only works when predicting RGB"
                self.from_ray = Conv2dLayer(dim_embed_view, final_out_dim, kernel_size=1, activation='linear')
            
            if self.predict_rgb:   # predict RGB over features
                self.to_rgb = Conv2dLayer(final_out_dim, self.img_channels * self.shuffle_factor, kernel_size=1, activation='linear')
        
    def set_steps(self, steps):
        if hasattr(self, "steps"):
            self.steps.fill_(steps)
        
    def transform_points(self, p, views=False):
        p = p / self.downscale_p_by
        if self.positional_encoding == 'gauss':
            B = self.B_view if views else self.B_pos
            p_transformed = positional_encoding(p, B, 'gauss', self.use_pos)
        elif self.positional_encoding == 'normal':
            L = self.n_freq_posenc_views if views else self.n_freq_posenc
            p_transformed = positional_encoding(p, L, 'normal', self.use_pos)
        else:
            p_transformed = p
        return p_transformed

    def forward(self, p_in, ray_d, z_shape=None, z_app=None, ws=None, shape=None, requires_grad=False, impl=None):
        with torch.set_grad_enabled(self.training or self.use_sdf or requires_grad):
            impl = 'mlp' if self.tcnn_backend else impl
            option, p_in = self.forward_inputs(p_in, shape=shape, impl=impl)
            if self.tcnn_backend:
                with torch.cuda.amp.autocast():
                    p = p_in.squeeze(-1).squeeze(-1)
                    o = self.network(p)
                    sigma_raw, feat = o[:, :self.shuffle_factor], o[:, self.shuffle_factor:]
                sigma_raw = rearrange(sigma_raw, '(b s) d -> b s d', s=option[2]).to(p_in.dtype)
                feat = rearrange(feat,  '(b s) d -> b s d', s=option[2]).to(p_in.dtype)
            else:
                feat, sigma_raw = self.forward_nerf(option, p_in, ray_d,  ws=ws, z_shape=z_shape, z_app=z_app)
        return feat, sigma_raw

    def forward_inputs(self, p_in, shape=None, impl=None):
        # prepare the inputs
        impl = impl if impl is not None else self.implementation
        if (shape is not None) and (impl == 'batch_reshape'):
            height, width, n_steps = shape[1:]
        elif impl == 'flatten_2d':
            (height, width), n_steps = dividable(p_in.shape[1]), 1
        elif impl == 'mlp':
            height, width, n_steps = 1, 1, p_in.shape[1]
        else:
            raise NotImplementedError("looking for more efficient implementation.")        
        p_in = rearrange(p_in, 'b (h w s) d -> (b s) d h w', h=height, w=width, s=n_steps)
        use_normal = self.use_normal or self.use_sdf
        if use_normal:
            p_in.requires_grad_(True)
        return (height, width, n_steps, use_normal), p_in
    
    def forward_nerf(self, option, p_in, ray_d=None, ws=None, z_shape=None, z_app=None):
        height, width, n_steps, use_normal = option
        
        # forward nerf feature networks
        p = self.transform_points(p_in.permute(0,2,3,1))
        if (self.z_dim > 0) and (not self.disable_latents):
            assert (z_shape is not None) and (ws is None)
            z_shape = repeat(z_shape, 'b c -> (b s) h w c', h=height, w=width, s=n_steps)
            p = torch.cat([p, z_shape], -1)
        p = p.permute(0,3,1,2)    # BS x C x H x W

        if height == width == 1:  # MLP
            p = p.squeeze(-1).squeeze(-1)
            
        net = self.fc_in(p, ws[:, 0] if ws is not None else None)
        if self.n_blocks > 1:
            for idx, layer in enumerate(self.blocks):
                ws_i = ws[:, idx + 1] if ws is not None else None
                if (self.skip_layer is not None) and (idx == self.skip_layer):
                    net = torch.cat([net, p], 1)
                net = layer(net, ws_i, up=1)

        # forward to get the final results
        w_idx = self.n_blocks  # fc_in, self.blocks
                
        feat_inputs = [net]
        if not (self.merge_sigma_feat or self.no_sigma):
            ws_i      = ws[:, w_idx] if ws is not None else None
            sigma_out = self.sigma_out(net, ws_i)
            if use_normal:
                gradients, = grad(
                    outputs=sigma_out, inputs=p_in, 
                    grad_outputs=torch.ones_like(sigma_out, requires_grad=False), 
                    retain_graph=True, create_graph=True, only_inputs=True)
                feat_inputs.append(gradients)
    
        ws_i = ws[:, -1] if ws is not None else None
        net = torch.cat(feat_inputs, 1) if len(feat_inputs) > 1 else net
        feat_out = self.feat_out(net, ws_i)  # this is used for lowres output

        if self.merge_sigma_feat:  # split sigma from the feature
            sigma_out, feat_out = feat_out[:, :self.shuffle_factor], feat_out[:, self.shuffle_factor:]
        elif self.no_sigma:
            sigma_out = None
                
        if self.predict_rgb:
            if self.use_viewdirs and ray_d is not None:
                ray_d = ray_d / torch.norm(ray_d, dim=-1, keepdim=True)
                ray_d = self.transform_points(ray_d, views=True)
                if self.z_dim > 0:
                    ray_d = torch.cat([ray_d, repeat(z_app, 'b c -> b (h w s) c', h=height, w=width, s=n_steps)], -1)
                ray_d = rearrange(ray_d, 'b (h w s) d -> （b s) d h w', h=height, w=width, s=n_steps)
                feat_ray = self.from_ray(ray_d)
                rgb = self.to_rgb(F.leaky_relu(feat_out + feat_ray))
            else:
                rgb = self.to_rgb(feat_out)

            if self.final_sigmoid_act:
                rgb = torch.sigmoid(rgb)    
            if self.normalized_feat:
                feat_out = feat_out / (1e-7 + feat_out.norm(dim=-1, keepdim=True))
            feat_out = torch.cat([rgb, feat_out], 1)

        # transform back
        if feat_out.ndim == 2:  # mlp mode
            sigma_out = rearrange(sigma_out, '(b s) d -> b s d', s=n_steps) if sigma_out is not None else None
            feat_out  = rearrange(feat_out,  '(b s) d -> b s d', s=n_steps)
        else:
            sigma_out = rearrange(sigma_out, '(b s) d h w -> b (h w s) d', s=n_steps) if sigma_out is not None else None
            feat_out  = rearrange(feat_out,  '(b s) d h w -> b (h w s) d', s=n_steps)
        return feat_out, sigma_out


@persistence.persistent_class
class CameraGenerator(torch.nn.Module):
    def __init__(self, in_dim=2, hi_dim=128, out_dim=2):
        super().__init__()
        self.affine1 = FullyConnectedLayer(in_dim, hi_dim, activation='lrelu')
        self.affine2 = FullyConnectedLayer(hi_dim, hi_dim, activation='lrelu')
        self.proj    = FullyConnectedLayer(hi_dim, out_dim)
        
    def forward(self, x):
        cam = self.proj(self.affine2(self.affine1(x)))
        return cam


@persistence.persistent_class
class CameraRay(object):

    range_u          = (0, 0)
    range_v          = (0.25, 0.25)
    range_radius     = (2.732, 2.732)
    depth_range      = [0.5, 6.]
    gaussian_camera  = False
    angular_camera   = False
    intersect_ball   = False
    fov              = 49.13
    bg_start         = 1.0
    depth_transform  = None     # "LogWarp" or "InverseWarp"
    dists_normalized = False    # use normalized interval instead of real dists
    random_rotate    = False
    ray_align_corner = True
    
    nonparam_cameras = None

    def __init__(self, camera_kwargs, **other_kwargs):
        if len(camera_kwargs) == 0:  # for compitatbility of old checkpoints
            camera_kwargs.update(other_kwargs)        
        for key in camera_kwargs:
            if hasattr(self, key):
                setattr(self, key, camera_kwargs[key])
        self.camera_matrix = get_camera_mat(fov=self.fov)

    def prepare_pixels(self, img_res, tgt_res, vol_res, camera_matrices, theta, margin=0, **unused):
        if self.ray_align_corner:    
            all_pixels = self.get_pixel_coords(img_res, camera_matrices, theta=theta)
            all_pixels = rearrange(all_pixels, 'b (h w) c -> b c h w', h=img_res, w=img_res)
            tgt_pixels = F.interpolate(all_pixels, size=(tgt_res, tgt_res), mode='nearest') if tgt_res < img_res else all_pixels.clone()
            vol_pixels = F.interpolate(tgt_pixels, size=(vol_res, vol_res), mode='nearest') if tgt_res > vol_res else tgt_pixels.clone()
            vol_pixels = rearrange(vol_pixels, 'b c h w -> b (h w) c')
            
        else:  # coordinates not aligned!
            tgt_pixels = self.get_pixel_coords(tgt_res, camera_matrices, corner_aligned=False, theta=theta)
            vol_pixels = self.get_pixel_coords(vol_res, camera_matrices, corner_aligned=False, theta=theta, margin=margin) \
                if (tgt_res > vol_res) or (margin > 0) else tgt_pixels.clone()
            tgt_pixels = rearrange(tgt_pixels, 'b (h w) c -> b c h w', h=tgt_res, w=tgt_res)
        return vol_pixels, tgt_pixels

    def prepare_pixels_regularization(self, tgt_pixels, n_reg_samples):
        # only apply when size is bigger than voxel resolution
        pace = tgt_pixels.size(-1) // n_reg_samples
        idxs = torch.arange(0, tgt_pixels.size(-1), pace, device=tgt_pixels.device)           # n_reg_samples
        u_xy = torch.rand(tgt_pixels.size(0), 2, device=tgt_pixels.device)
        u_xy = (u_xy * pace).floor().long()    # batch_size x 2
        x_idxs, y_idxs = idxs[None,:] + u_xy[:,:1], idxs[None,:] + u_xy[:,1:]
        rand_indexs = (x_idxs[:,None,:] + y_idxs[:,:,None] * tgt_pixels.size(-1)).reshape(tgt_pixels.size(0), -1)
        tgt_pixels  = rearrange(tgt_pixels, 'b c h w -> b (h w) c')
        rand_pixels = tgt_pixels.gather(1, rand_indexs.unsqueeze(-1).repeat(1,1,2))
        return rand_pixels, rand_indexs

    def get_roll(self, ws, training=True, theta=None, **unused):
        if (self.random_rotate is not None) and training:
            theta = torch.randn(ws.size(0)).to(ws.device) * self.random_rotate / 2
            theta = theta / 180 * math.pi
        else:
            if theta is not None:
                theta = torch.ones(ws.size(0)).to(ws.device) * theta
        return theta

    def get_camera(self, batch_size, device, mode='random', fov=None, force_uniform=False):
        if fov is not None:
            camera_matrix = get_camera_mat(fov)
        else:
            camera_matrix = self.camera_matrix
        camera_mat = camera_matrix.repeat(batch_size, 1, 1).to(device)
        reg_loss = None  # TODO: useless

        if isinstance(mode, list):   
            # default camera generator, we assume input mode is linear
            if len(mode) == 3:
                val_u, val_v, val_r = mode
                r0 = self.range_radius[0]
                r1 = self.range_radius[1]
            else:
                val_u, val_v, val_r, r_s = mode
                r0 = self.range_radius[0] * r_s
                r1 = self.range_radius[1] * r_s
            
            world_mat = get_camera_pose(
                self.range_u, self.range_v, [r0, r1], 
                val_u, val_v, val_r, 
                batch_size=batch_size, 
                gaussian=False,   # input mode is by default uniform
                angular=self.angular_camera).to(device)
        
        elif isinstance(mode, torch.Tensor):    
            world_mat, mode = get_camera_pose_v2(
                self.range_u, self.range_v, self.range_radius, mode, 
                gaussian=self.gaussian_camera and (not force_uniform), 
                angular=self.angular_camera)
            world_mat = world_mat.to(device)
            mode = torch.stack(mode, 1).to(device)
        
        else:
            world_mat, mode = get_random_pose(
                self.range_u, self.range_v, 
                self.range_radius, batch_size,
                gaussian=self.gaussian_camera, 
                angular=self.angular_camera)            
            world_mat = world_mat.to(device)
            mode = torch.stack(mode, 1).to(device)
        return camera_mat.float(), world_mat.float(), mode, reg_loss

    def get_transformed_depth(self, di, reversed=False):
        depth_range = self.depth_range
        
        if (self.depth_transform is None) or (self.depth_transform == 'None'):
            g_fwd, g_inv = lambda x: x, lambda x: x
        elif self.depth_transform == 'LogWarp':
            g_fwd, g_inv = math.log, torch.exp
        elif self.depth_transform == 'InverseWarp':
            g_fwd, g_inv = lambda x: 1/x, lambda x: 1/x
        else:
            raise NotImplementedError

        if not reversed:
            return g_inv(g_fwd(depth_range[1]) * di + g_fwd(depth_range[0]) * (1 - di))
        else:
            d0 = (g_fwd(di) - g_fwd(depth_range[0])) / (g_fwd(depth_range[1]) - g_fwd(depth_range[0])) 
            return d0.clip(min=0, max=1)
    
    def get_evaluation_points(self, pixels_world=None, camera_world=None, di=None, p_i=None, no_reshape=False, transform=None):
        if p_i is None:
            batch_size = pixels_world.shape[0]
            n_steps = di.shape[-1]
            ray_i = pixels_world - camera_world
            p_i = camera_world.unsqueeze(-2).contiguous() + \
                di.unsqueeze(-1).contiguous() * ray_i.unsqueeze(-2).contiguous()
            ray_i = ray_i.unsqueeze(-2).repeat(1, 1, n_steps, 1)

        else:
            assert no_reshape, "only used to transform points to a warped space"

        if transform is None:
            transform = self.depth_transform

        if transform == 'LogWarp':
            c = torch.tensor([1., 0., 0.]).to(p_i.device)
            p_i = normalization_inverse_sqrt_dist_centered(
                p_i, c[None, None, None, :], self.depth_range[1])
        
        elif transform == 'InverseWarp':
            # https://arxiv.org/pdf/2111.12077.pdf
            p_n = p_i.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-7)
            con = p_n.ge(1).type_as(p_n)
            p_i = p_i * (1 -con) + (2 - 1 / p_n) * (p_i / p_n) * con
            
        if no_reshape:
            return p_i

        assert(p_i.shape == ray_i.shape)
        p_i = p_i.reshape(batch_size, -1, 3)
        ray_i = ray_i.reshape(batch_size, -1, 3)
        return p_i, ray_i

    def get_evaluation_points_bg(self, pixels_world, camera_world, di):
        batch_size = pixels_world.shape[0]
        n_steps    = di.shape[-1]
        n_pixels   = pixels_world.shape[1]
        ray_world  = pixels_world - camera_world
        ray_world  = ray_world / ray_world.norm(dim=-1, keepdim=True)  # normalize
        
        camera_world = camera_world.unsqueeze(-2).expand(batch_size, n_pixels, n_steps, 3)
        ray_world = ray_world.unsqueeze(-2).expand(batch_size, n_pixels, n_steps, 3)
        bg_pts, _ = depth2pts_outside(camera_world, ray_world, di)    # di: 1 ---> 0

        bg_pts    = bg_pts.reshape(batch_size, -1, 4)
        ray_world = ray_world.reshape(batch_size, -1, 3)
        return bg_pts, ray_world

    def add_noise_to_interval(self, di):
        di_mid  = .5 * (di[..., 1:] + di[..., :-1])
        di_high = torch.cat([di_mid, di[..., -1:]], dim=-1)
        di_low  = torch.cat([di[..., :1], di_mid], dim=-1)
        noise   = torch.rand_like(di_low)
        ti      = di_low + (di_high - di_low) * noise
        return ti

    def calc_volume_weights(self, sigma, z_vals=None, ray_vector=None, dists=None, last_dist=1e10):
        if dists is None:
            dists = z_vals[..., 1:] - z_vals[..., :-1]
            if ray_vector is not None:
                dists = dists * torch.norm(ray_vector, dim=-1, keepdim=True)
            dists = torch.cat([dists, torch.ones_like(dists[..., :1]) * last_dist], dim=-1)
        alpha = 1.-torch.exp(-F.relu(sigma)*dists)

        if last_dist > 0:
            alpha[..., -1] = 1
            
        # alpha = 1.-torch.exp(-sigma * dists)
        T = torch.cumprod(torch.cat([
                torch.ones_like(alpha[:, :, :1]),
                (1. - alpha + 1e-10), ], dim=-1), dim=-1)[..., :-1]
        weights = alpha * T
        return weights, T[..., -1], dists

    def get_pixel_coords(self, tgt_res, camera_matrices, corner_aligned=True, margin=0, theta=None, invert_y=True):
        device     = camera_matrices[0].device
        batch_size = camera_matrices[0].shape[0]
        # margin = self.margin if margin is None else margin
        full_pixels = arange_pixels((tgt_res, tgt_res), 
            batch_size, invert_y_axis=invert_y, margin=margin,
            corner_aligned=corner_aligned).to(device)
        if (theta is not None):
            theta = theta.unsqueeze(-1)
            x = full_pixels[..., 0] * torch.cos(theta) - full_pixels[..., 1] * torch.sin(theta)
            y = full_pixels[..., 0] * torch.sin(theta) + full_pixels[..., 1] * torch.cos(theta)
            full_pixels = torch.stack([x, y], -1)
        return full_pixels

    def get_origin_direction(self, pixels, camera_matrices):
        camera_mat, world_mat = camera_matrices[:2]
        if camera_mat.size(0) < pixels.size(0):
            camera_mat = repeat(camera_mat, 'b c d -> (b s) c d', s=pixels.size(0)//camera_mat.size(0))
        if world_mat.size(0) < pixels.size(0):
            world_mat = repeat(world_mat, 'b c d -> (b s) c d', s=pixels.size(0)//world_mat.size(0))
        pixels_world = image_points_to_world(pixels, camera_mat=camera_mat, world_mat=world_mat)
        camera_world = origin_to_world(pixels.size(1), camera_mat=camera_mat, world_mat=world_mat)
        ray_vector = pixels_world - camera_world
        return pixels_world, camera_world, ray_vector

    def set_camera_prior(self, dataset_cams):
        self.nonparam_cameras = dataset_cams


@persistence.persistent_class
class VolumeRenderer(object):

    n_ray_samples     = 14
    n_bg_samples      = 4
    n_final_samples   = None    # final nerf steps after upsampling (optional)
    sigma_type        = 'relu'  # other allowed options including, "abs", "shiftedsoftplus", "exp"
    
    hierarchical      = True
    fine_only         = False
    no_background     = False
    white_background  = False
    mask_background   = False
    pre_volume_size   = None
    
    bound             = None
    density_p_target  = 1.0
    tv_loss_weight    = 0.0     # for now only works for density-based voxels

    def __init__(self, renderer_kwargs, camera_ray, input_encoding=None, **other_kwargs):
        if len(renderer_kwargs) == 0:  # for compitatbility of old checkpoints
            renderer_kwargs.update(other_kwargs)        
        for key in renderer_kwargs:
            if hasattr(self, key):
                setattr(self, key, renderer_kwargs[key])
        self.C = camera_ray
        self.I = input_encoding

    def split_feat(self, x, img_channels, white_color=None, split_rgb=True):
        img = x[:, :img_channels]
        if split_rgb:
            x = x[:, img_channels:]
        if (white_color is not None) and self.white_background:
            img = img + white_color
        return x, img

    def get_bound(self):
        if self.bound is not None:
            return self.bound

        # when applying normalization, the points are restricted inside R=2 ball
        if self.C.depth_transform == 'InverseWarp':
            bound = 2
        else:  # TODO: this is a bit hacky as we assume object at origin
            bound = (self.C.depth_range[1] - self.C.depth_range[0])
        return bound

    def get_density(self, sigma_raw, fg_nerf, no_noise=False, training=False):
        if fg_nerf.use_sdf:
            sigma = fg_nerf.density_transform.density_func(sigma_raw)
        elif self.sigma_type == 'relu':
            if training and (not no_noise):    # adding noise to pass gradient?
                sigma_raw = sigma_raw + torch.randn_like(sigma_raw)
            sigma = F.relu(sigma_raw)
        elif self.sigma_type == 'shiftedsoftplus':  # https://arxiv.org/pdf/2111.11215.pdf
            sigma = F.softplus(sigma_raw - 1)       # 1 is the shifted bias.
        elif self.sigma_type == 'exp_truncated':    # density in the log-space
            sigma = torch.exp(5 - F.relu(5 - (sigma_raw - 1)))  # up-bound = 5, also shifted by 1
        else:
            sigma = sigma_raw
        return sigma
    
    def forward_hierarchical_sampling(self, di, weights, n_steps, det=False):
        di_mid = 0.5 * (di[..., :-1] + di[..., 1:])
        n_bins = di_mid.size(-1)
        batch_size = di.size(0)
        di_fine = sample_pdf(
            di_mid.reshape(-1, n_bins), 
            weights.reshape(-1, n_bins+1)[:, 1:-1],
            n_steps, det=det).reshape(batch_size, -1, n_steps)
        return di_fine

    def forward_rendering_with_grid(self, H, output, fg_nerf, nerf_input_cams, nerf_input_feats, latent_codes, styles):
        pixels_world, camera_world, ray_vector = nerf_input_cams
        z_shape_obj, z_app_obj = latent_codes[:2]
        height, width = dividable(H.n_points)
        fg_shape = [H.batch_size, height, width, H.n_steps]
        bound = self.get_bound()

        # sample points
        di = torch.linspace(0., 1., steps=H.n_steps).to(H.device)
        di = repeat(di, 's -> b n s', b=H.batch_size, n=H.n_points)
        if (H.training and (not H.get('disable_noise', False))) or H.get('force_noise', False):
            di = self.C.add_noise_to_interval(di)
        di_trs = self.C.get_transformed_depth(di)
        p_i, r_i = self.C.get_evaluation_points(pixels_world, camera_world, di_trs)

        # query the density grids and compute the mask and indices
        pre_sigma_raw = self.I.query_input_features(p_i, ('volume', nerf_input_feats[3]), fg_shape, bound) 
        pre_sigma     = self.get_density(rearrange(pre_sigma_raw, 'b (n s) () -> b n s', s=H.n_steps), fg_nerf, training=H.training)
        pre_weights   = self.C.calc_volume_weights(pre_sigma, di if self.C.dists_normalized else di_trs, ray_vector, last_dist=1e10)[0]
        pre_p_target  = 1.0 if H.alpha <= 0 else 1.0 - (1.0 - self.density_p_target) * H.alpha 
        
        if self.tv_loss_weight > 0:
            voxel_density = nerf_input_feats[3][:, 0]
            tv_loss = torch.mean(torch.sqrt(1e-7 +
                (voxel_density[:, :-1, :-1, 1:] - voxel_density[:, :-1, :-1, :-1]) ** 2 +
                (voxel_density[:, :-1, 1:, :-1] - voxel_density[:, :-1, :-1, :-1]) ** 2 +
                (voxel_density[:, 1:, :-1, :-1] - voxel_density[:, :-1, :-1, :-1]) ** 2))
            output.reg_loss.tv_loss = tv_loss * self.tv_loss_weight

        if pre_p_target < 1:  # use density grid to prune samples
            pre_topp_mask = rearrange(topp_masking(pre_weights, pre_p_target), 'b n s -> b (n s)')
            pre_topp_asgn = repeat(torch.arange(H.n_points * H.batch_size, device=p_i.device), 
                '(b n) -> b (n s)', b=H.batch_size, n=H.n_points, s=H.n_steps)[pre_topp_mask]
            pre_topp_lens = pre_topp_mask.sum(-1).cpu().tolist()
            pre_weights   = rearrange(pre_weights, 'b n s -> b (n s)')
            pre_topp_wgts = pre_weights[pre_topp_mask]
            pre_topp_maxl = int(np.ceil(max(pre_topp_lens) / 512) * 512)  # just for convinenet
            pre_topp_asg2 = torch.cat([
                torch.arange(pre_topp_lens[b], device=p_i.device) + 
                pre_topp_maxl * b for b in range(H.batch_size)])
            
            # prune the samples based on masks, move to 2D for style-based generation
            filtered_pi   = p_i[pre_topp_mask]
            filtered_pi_b = torch.scatter(
                filtered_pi.new_zeros(H.batch_size * pre_topp_maxl, 3), 0,
                repeat(pre_topp_asg2, 'n -> n s', s=3), filtered_pi).reshape(H.batch_size, -1, 3)
            fg_shape = [H.batch_size, pre_topp_maxl // 512, 512, 1]
            
            # forward nerf (with tri-plane features) with pruned points
            filtered_pi_b   = self.I.query_input_features(filtered_pi_b, nerf_input_feats, fg_shape, bound)
            filtered_feat_b = fg_nerf(filtered_pi_b, r_i, z_shape_obj, z_app_obj, ws=styles, shape=fg_shape, impl='mlp')[0]

            # get back to 1D
            filtered_feat = torch.gather(filtered_feat_b.reshape(H.batch_size * pre_topp_maxl, -1), 0, 
                repeat(pre_topp_asg2, 'n -> n s', s=filtered_feat_b.size(-1)))
            feat = torch.scatter_add(
                filtered_feat.new_zeros(H.batch_size * H.n_points, filtered_feat.size(-1)), 0, 
                repeat(pre_topp_asgn, 'n -> n d', d=filtered_feat.size(-1)),
                filtered_feat * pre_topp_wgts[:, None]).reshape(H.batch_size, H.n_points, -1)
            pre_weights_before = pre_weights.reshape(H.batch_size, -1, H.n_steps)
            pre_weights = torch.zeros_like(pre_weights).masked_scatter(
                pre_topp_mask, pre_topp_wgts).reshape(H.batch_size, -1, H.n_steps)
            
            # balancing and pass gradients?
            feat = feat / pre_weights.sum(dim=-1, keepdim=True) *  pre_weights_before.sum(dim=-1, keepdim=True)
            
        else:  
            # normal NeRF forward, no pruning.
            p_i  = self.I.query_input_features(p_i, nerf_input_feats, fg_shape, bound)
            feat = fg_nerf(p_i, r_i, z_shape_obj, z_app_obj, ws=styles, shape=fg_shape)[0]
            feat = rearrange(feat, 'b (n s) d -> b n s d', s=H.n_steps)
            feat = torch.sum(pre_weights.unsqueeze(-1) * feat, dim=-2)
        
        output.feat      += [feat]
        output.fg_weights = pre_weights
        output.fg_depths  = (di, di_trs)  
        return output

    def forward_sampling(self, H, output, fg_nerf, nerf_input_cams, nerf_input_feats, latent_codes, styles):
        # TODO: experimental research code. Not functional yet.
         
        pixels_world, camera_world, ray_vector = nerf_input_cams
        z_shape_obj, z_app_obj = latent_codes[:2]
        height, width = dividable(H.n_points)
        bound = self.get_bound()
        
        # just to simulate
        H.n_steps = 64
        di = torch.linspace(0., 1., steps=H.n_steps).to(H.device)
        di = repeat(di, 's -> b n s', b=H.batch_size, n=H.n_points)
        if (H.training and (not H.get('disable_noise', False))) or H.get('force_noise', False):
            di = self.C.add_noise_to_interval(di)
        di_trs = self.C.get_transformed_depth(di)
        
        fg_shape = [H.batch_size, height, width, 1]
        
        # iteration in the loop (?)
        feats, sigmas = [], []
        with torch.enable_grad():
            di_trs.requires_grad_(True)
            for s in range(di_trs.shape[-1]):
                di_s = di_trs[..., s:s+1]
                p_i, r_i = self.C.get_evaluation_points(pixels_world, camera_world, di_s)
                if nerf_input_feats is not None:
                    p_i = self.I.query_input_features(p_i, nerf_input_feats, fg_shape, bound)        
                feat, sigma_raw = fg_nerf(p_i, r_i, z_shape_obj, z_app_obj, ws=styles, shape=fg_shape, requires_grad=True)
                sigma = self.get_density(sigma_raw, fg_nerf, training=H.training)
            feats += [feat]
            sigmas += [sigma]
        feat, sigma = torch.stack(feats, 2), torch.cat(sigmas, 2)
        fg_weights, bg_lambda = self.C.calc_volume_weights(
            sigma, di if self.C.dists_normalized else di_trs,  # use real dists for computing weights
            ray_vector, last_dist=0 if not H.fg_inf_depth else 1e10)[:2]
        fg_feat = torch.sum(fg_weights.unsqueeze(-1) * feat, dim=-2)
        
        output.feat       += [fg_feat]
        output.full_out   += [feat]
        output.fg_weights  = fg_weights
        output.bg_lambda   = bg_lambda
        output.fg_depths   = (di, di_trs)
        return output
        
    def forward_rendering(self, H, output, fg_nerf, nerf_input_cams, nerf_input_feats, latent_codes, styles):
        pixels_world, camera_world, ray_vector = nerf_input_cams
        z_shape_obj, z_app_obj = latent_codes[:2]
        height, width = dividable(H.n_points)
        fg_shape = [H.batch_size, height, width, H.n_steps]
        bound = self.get_bound()

        # sample points
        di = torch.linspace(0., 1., steps=H.n_steps).to(H.device)
        di = repeat(di, 's -> b n s', b=H.batch_size, n=H.n_points)
        if (H.training and (not H.get('disable_noise', False))) or H.get('force_noise', False):
            di = self.C.add_noise_to_interval(di)
        di_trs = self.C.get_transformed_depth(di)
        p_i, r_i = self.C.get_evaluation_points(pixels_world, camera_world, di_trs)

        if nerf_input_feats is not None:
            p_i = self.I.query_input_features(p_i, nerf_input_feats, fg_shape, bound)        
        
        feat, sigma_raw = fg_nerf(p_i, r_i, z_shape_obj, z_app_obj, ws=styles, shape=fg_shape)
        feat = rearrange(feat, 'b (n s) d -> b n s d', s=H.n_steps)
        sigma_raw = rearrange(sigma_raw.squeeze(-1), 'b (n s) -> b n s', s=H.n_steps) 
        sigma = self.get_density(sigma_raw, fg_nerf, training=H.training)         
        fg_weights, bg_lambda = self.C.calc_volume_weights(
            sigma, di if self.C.dists_normalized else di_trs,  # use real dists for computing weights
            ray_vector, last_dist=0 if not H.fg_inf_depth else 1e10)[:2]

        if self.hierarchical and (not H.get('disable_hierarchical', False)):
            with torch.no_grad():
                di_fine = self.forward_hierarchical_sampling(di, fg_weights, H.n_steps, det=(not H.training))
            di_trs_fine = self.C.get_transformed_depth(di_fine)
            p_f, r_f = self.C.get_evaluation_points(pixels_world, camera_world, di_trs_fine)
            if nerf_input_feats is not None:
                p_f = self.I.query_input_features(p_f, nerf_input_feats, fg_shape, bound)

            feat_f, sigma_raw_f = fg_nerf(p_f, r_f, z_shape_obj, z_app_obj, ws=styles, shape=fg_shape)
            feat_f      = rearrange(feat_f, 'b (n s) d -> b n s d', s=H.n_steps)
            sigma_raw_f = rearrange(sigma_raw_f.squeeze(-1), 'b (n s) -> b n s', s=H.n_steps)
            sigma_f     = self.get_density(sigma_raw_f, fg_nerf, training=H.training)
            
            feat      = torch.cat([feat_f, feat], 2)
            sigma     = torch.cat([sigma_f, sigma], 2)
            sigma_raw = torch.cat([sigma_raw_f, sigma_raw], 2)
            di        = torch.cat([di_fine, di], 2)
            di_trs    = torch.cat([di_trs_fine, di_trs], 2)
            
            di, indices = torch.sort(di, dim=2)
            di_trs    = torch.gather(di_trs, 2, indices)
            sigma     = torch.gather(sigma, 2, indices)
            sigma_raw = torch.gather(sigma_raw, 2, indices)
            feat      = torch.gather(feat, 2, repeat(indices, 'b n s -> b n s d', d=feat.size(-1)))

            fg_weights, bg_lambda = self.C.calc_volume_weights(
                sigma, di if self.C.dists_normalized else di_trs,  # use real dists for computing weights, 
                ray_vector, last_dist=0 if not H.fg_inf_depth else 1e10)[:2]

        fg_feat = torch.sum(fg_weights.unsqueeze(-1) * feat, dim=-2)
        
        output.feat       += [fg_feat]
        output.full_out   += [feat]
        output.fg_weights  = fg_weights
        output.bg_lambda   = bg_lambda
        output.fg_depths   = (di, di_trs)
        return output

    def forward_rendering_background(self, H, output, bg_nerf, nerf_input_cams, latent_codes, styles_bg):
        pixels_world, camera_world, _ = nerf_input_cams
        z_shape_bg, z_app_bg = latent_codes[2:]
        height, width = dividable(H.n_points)
        bg_shape = [H.batch_size, height, width, H.n_bg_steps]            
        if H.fixed_input_cams is not None:
            pixels_world, camera_world, _ = H.fixed_input_cams

        # render background, use NeRF++ inverse sphere parameterization
        di = torch.linspace(-1., 0., steps=H.n_bg_steps).to(H.device)
        di = repeat(di, 's -> b n s', b=H.batch_size, n=H.n_points) * self.C.bg_start
        if (H.training and (not H.get('disable_noise', False))) or H.get('force_noise', False):
            di = self.C.add_noise_to_interval(di)
        p_bg, r_bg = self.C.get_evaluation_points_bg(pixels_world, camera_world, -di)

        feat, sigma_raw = bg_nerf(p_bg, r_bg, z_shape_bg, z_app_bg, ws=styles_bg, shape=bg_shape)
        feat      = rearrange(feat, 'b (n s) d -> b n s d', s=H.n_bg_steps)
        sigma_raw = rearrange(sigma_raw.squeeze(-1), 'b (n s) -> b n s', s=H.n_bg_steps)
        sigma     = self.get_density(sigma_raw, bg_nerf, training=H.training)
        bg_weights = self.C.calc_volume_weights(sigma, di, None)[0]
        bg_feat = torch.sum(bg_weights.unsqueeze(-1) * feat, dim=-2)
        
        if output.get('bg_lambda', None) is not None:
            bg_feat = output.bg_lambda.unsqueeze(-1) * bg_feat
        output.feat       += [bg_feat]
        output.full_out   += [feat]
        output.bg_weights  = bg_weights
        output.bg_depths   = di
        return output
        
    def forward_volume_rendering(
        self, 
        nerf_modules,      # (fg_nerf, bg_nerf)
        camera_matrices,   # camera (K, RT)
        vol_pixels,

        nerf_input_feats       = None,
        latent_codes           = None,
        styles                 = None,
        styles_bg              = None,
        not_render_background  = False, 
        only_render_background = False,

        render_option          = None,
        return_full            = False,
        
        alpha                  = 0,
        **unused):

        assert (latent_codes is not None) or (styles is not None)
        assert self.no_background or (nerf_input_feats is None), "input features do not support background field"
        
        # hyper-parameters for rendering
        H      = EasyDict(**unused)
        output = EasyDict()
        output.reg_loss = EasyDict()
        output.feat = []
        output.full_out = []

        if render_option is None:
            render_option = ""
        H.render_option = render_option
        H.alpha         = alpha

        # prepare for rendering (parameters)
        fg_nerf, bg_nerf  = nerf_modules
        
        H.training     = fg_nerf.training
        H.device       = camera_matrices[0].device
        H.batch_size   = camera_matrices[0].shape[0]
        H.img_channels = fg_nerf.img_channels
        H.n_steps      = self.n_ray_samples
        H.n_bg_steps   = self.n_bg_samples
        if alpha == -1:
            H.n_steps  = 20  # just for memory safe.
        if "steps" in render_option:
            H.n_steps  = [int(r.split(':')[1]) for r in H.render_option.split(',') if r[:5] == 'steps'][0]

        # prepare for pixels for generating images
        if isinstance(vol_pixels, tuple):
            vol_pixels, rand_pixels = vol_pixels
            pixels    = torch.cat([vol_pixels, rand_pixels], 1)
            H.rnd_res = int(math.sqrt(rand_pixels.size(1)))
        else:
            pixels, rand_pixels, H.rnd_res = vol_pixels, None, None
        H.tgt_res, H.n_points = int(math.sqrt(vol_pixels.size(1))), pixels.size(1)
        nerf_input_cams = self.C.get_origin_direction(pixels, camera_matrices)

        # set up an frozen camera for background if necessary
        if ('freeze_bg' in H.render_option) and (bg_nerf is not None):
            pitch, yaw = 0.2 + np.pi/2, 0
            range_u, range_v = self.C.range_u, self.C.range_v
            u = (yaw - range_u[0])   / (range_u[1] - range_u[0])
            v = (pitch - range_v[0]) / (range_v[1] - range_v[0])
            fixed_camera = self.C.get_camera(
                batch_size=H.batch_size, mode=[u, v, 0.5], device=H.device)
            H.fixed_input_cams = self.C.get_origin_direction(pixels, fixed_camera)
        else:
            H.fixed_input_cams = None
        
        H.fg_inf_depth = (self.no_background or not_render_background) and (not self.white_background)
        assert(not (not_render_background and only_render_background))
        
        # volume rendering options: bg_weights, bg_lambda = None, None
        if (nerf_input_feats is not None) and \
            len(nerf_input_feats) == 4 and \
            nerf_input_feats[2] == 'volume' and \
            H.fg_inf_depth:   
            # volume rendering with voxel-based density
            output = self.forward_rendering_with_grid(
                H, output, fg_nerf, nerf_input_cams, nerf_input_feats, latent_codes, styles)

        else: 
            # standard volume rendering 
            if not only_render_background:
                output = self.forward_rendering(
                    H, output, fg_nerf, nerf_input_cams, nerf_input_feats, latent_codes, styles)
                
            # background rendering (NeRF++)
            if (not not_render_background) and (not self.no_background):
                output = self.forward_rendering_background(
                    H, output, bg_nerf, nerf_input_cams, latent_codes, styles_bg)
                         
        if ('early' in render_option) and ('value' not in render_option):
            return self.gen_optional_output(
                H, fg_nerf, nerf_input_cams, nerf_input_feats, latent_codes, styles, output)

        # ------------------------------------------- PREPARE FULL OUTPUT (NO 2D aggregation) -------------------------------------------- #        
        vol_len   = vol_pixels.size(1)
        feat_map  = sum(output.feat)
        full_x    = rearrange(feat_map[:, :vol_len], 'b (h w) d -> b d h w', h=H.tgt_res)
        split_rgb = fg_nerf.add_rgb or fg_nerf.predict_rgb
        
        full_out = self.split_feat(full_x, H.img_channels, None, split_rgb=split_rgb) 
        if rand_pixels is not None:   # used in full supervision (debug later)
            if return_full:
                assert (fg_nerf.predict_rgb or fg_nerf.add_rgb)
                rand_outputs = [f[:,vol_pixels.size(1):] for f in output.full_out]
                full_weights = torch.cat([output.fg_weights, output.bg_weights * output.bg_lambda.unsqueeze(-1)], -1) \
                    if output.get('bg_weights', None) is not None else output.fg_weights
                full_weights = full_weights[:,vol_pixels.size(1):]
                full_weights = rearrange(full_weights, 'b (h w) s -> b s h w', h=H.rnd_res, w=H.rnd_res)

                lh, lw = dividable(full_weights.size(1))
                full_x = rearrange(torch.cat(rand_outputs, 2), 'b (h w) (l m) d -> b d (l h) (m w)', 
                                   h=H.rnd_res, w=H.rnd_res, l=lh, m=lw)
                full_x, full_img = self.split_feat(full_x, H.img_channels, split_rgb=split_rgb)
                output.rand_out = (full_x, full_img, full_weights)
            
            else:
                rand_x = rearrange(feat_map[:, vol_len:], 'b (h w) d -> b d h w', h=H.rnd_res)
                output.rand_out = self.split_feat(rand_x, H.img_channels, split_rgb=split_rgb)
        output.full_out = full_out            
        return output

    def post_process_outputs(self, outputs, freeze_nerf=False):
        if freeze_nerf:
            outputs = [x.detach() if isinstance(x, torch.Tensor) else x for x in outputs]
        x, img = outputs[0], outputs[1]
        probs  = outputs[2] if len(outputs) == 3 else None
        return x, img, probs

    def gen_optional_output(self, H, fg_nerf, nerf_input_cams, nerf_input_feats, latent_codes, styles, output):
        _, camera_world, ray_vector = nerf_input_cams
        z_shape_obj, z_app_obj = latent_codes[:2]
        fg_depth_map = torch.sum(output.fg_weights * output.fg_depths[1], dim=-1, keepdim=True)
        img = camera_world[:, :1] + fg_depth_map * ray_vector
        img = img.permute(0,2,1).reshape(-1, 3, H.tgt_res, H.tgt_res)
        
        if 'input_feats' in H.render_option:
            a, b = [r.split(':')[1:] for r in H.render_option.split(',') if r.startswith('input_feats')][0]
            a, b = int(a), int(b)
            if nerf_input_feats[0] == 'volume':
                img = nerf_input_feats[1][:,a:a+3,b,:,:]
            elif nerf_input_feats[0] == 'tri_plane':
                img = nerf_input_feats[1][:,b,a:a+3,:,:]
            elif nerf_input_feats[0] == 'hash_table':
                assert self.I.hash_mode == 'grid_hash'
                img = nerf_input_feats[1][:,self.I.offsets[b]:self.I.offsets[b+1], :]
                siz = int(np.ceil(img.size(1)**(1/3)))
                img = rearrange(img, 'b (d h w) c -> b (d c) h w', h=siz, w=siz, d=siz)
                img = img[:, a:a+3]
            else:
                raise NotImplementedError

        if 'normal' in H.render_option.split(','):
            shift_l, shift_r = img[:,:,2:,:], img[:,:,:-2,:]
            shift_u, shift_d = img[:,:,:,2:], img[:,:,:,:-2]
            diff_hor = normalize(shift_r - shift_l, axis=1)[0][:, :, :, 1:-1]
            diff_ver = normalize(shift_u - shift_d, axis=1)[0][:, :, 1:-1, :]
            normal = torch.cross(diff_hor, diff_ver, dim=1)
            img = normalize(normal, axis=1)[0]

        if 'gradient' in H.render_option.split(','):
            points, _ = self.C.get_evaluation_points(camera_world + ray_vector, camera_world, output.fg_depths[1])
            fg_shape  = [H.batch_size, H.tgt_res, H.tgt_res, output.fg_depths[1].size(-1)]
            with torch.enable_grad():
                points.requires_grad_(True)
                if (nerf_input_feats is not None) and len(nerf_input_feats) == 4 and nerf_input_feats[2] == 'volume':  # with voxel grid density
                    sigma_out = self.I.query_input_features(points, ('volume', nerf_input_feats[3]), fg_shape, self.get_bound())
                else:
                    inputs = self.I.query_input_features(points, nerf_input_feats, fg_shape, self.get_bound(), True) \
                        if nerf_input_feats is not None else points
                    _, sigma_out = fg_nerf(inputs, None, ws=styles, shape=fg_shape, z_shape=z_shape_obj, z_app=z_app_obj, requires_grad=True)
                gradients, = grad(
                    outputs=sigma_out, inputs=points, 
                    grad_outputs=torch.ones_like(sigma_out, requires_grad=False), 
                    retain_graph=True, create_graph=True, only_inputs=True)
            gradients = rearrange(gradients, 'b (n s) d -> b n s d', s=output.fg_depths[1].size(-1))
            avg_grads = (gradients * output.fg_weights.unsqueeze(-1)).sum(-2)
            avg_grads = F.normalize(avg_grads, p=2, dim=-1)
            normal    = rearrange(avg_grads, 'b (h w) s -> b s h w', h=H.tgt_res, w=H.tgt_res)
            img       = -normal

        return {'full_out': (None, img)}


@persistence.persistent_class
class Upsampler(object):

    no_2d_renderer   = False
    no_residual_img  = False
    block_reses      = None
    shared_rgb_style = False
    upsample_type    = 'default'
    img_channels     = 3
    in_res           = 32
    out_res          = 512
    channel_base     = 1
    channel_base_sz  = None
    channel_max      = 512
    channel_dict     = None
    out_channel_dict = None

    def __init__(self, upsampler_kwargs, **other_kwargs):
        # for compitatbility of old checkpoints
        for key in other_kwargs:
            if hasattr(self, key) and (key not in upsampler_kwargs):
                upsampler_kwargs[key] = other_kwargs[key]
        for key in upsampler_kwargs:
            if hasattr(self, key):
                setattr(self, key, upsampler_kwargs[key])

        self.out_res_log2 = int(np.log2(self.out_res))

        # set up upsamplers
        if self.block_reses is None:
            self.block_resolutions = [2 ** i for i in range(2, self.out_res_log2 + 1)]
            self.block_resolutions = [b for b in self.block_resolutions if b > self.in_res]
        else:
            self.block_resolutions = self.block_reses
        
        if self.no_2d_renderer:
            self.block_resolutions = []

    def build_network(self, w_dim, input_dim, **block_kwargs):
        upsamplers = []
        if len(self.block_resolutions) > 0:  # nerf resolution smaller than image
            channel_base       = int(self.channel_base * 32768) if self.channel_base_sz is None else self.channel_base_sz
            fp16_resolution    = self.block_resolutions[0] * 2 # do not use fp16 for the first block

            if self.channel_dict is None:
                channels_dict  = {res: min(channel_base // res, self.channel_max) for res in self.block_resolutions}
            else:
                channels_dict  = self.channel_dict 

            if self.out_channel_dict is not None:
                img_channels   = self.out_channel_dict
            else:
                img_channels   = {res: self.img_channels for res in self.block_resolutions}
            
            for ir, res in enumerate(self.block_resolutions):
                res_before   = self.block_resolutions[ir-1] if ir > 0 else self.in_res
                in_channels  = channels_dict[res_before] if ir > 0 else input_dim
                out_channels = channels_dict[res]                
                use_fp16     = (res >= fp16_resolution) # TRY False
                is_last      = (ir == (len(self.block_resolutions) - 1))
                no_upsample  = (res == res_before)
                block        = util.construct_class_by_name(
                    class_name=block_kwargs.get('block_name', "training.networks.SynthesisBlock"),
                    in_channels=in_channels, 
                    out_channels=out_channels, 
                    w_dim=w_dim, 
                    resolution=res,
                    img_channels=img_channels[res], 
                    is_last=is_last, 
                    use_fp16=use_fp16, 
                    disable_upsample=no_upsample,
                    block_id=ir,
                    **block_kwargs)

                upsamplers  += [{
                    'block': block,
                    'num_ws': block.num_conv if not is_last else block.num_conv + block.num_torgb,
                    'name': f'b{res}' if res_before != res else f'b{res}_l{ir}'
                }]
            self.num_ws = sum([u['num_ws'] for u in upsamplers])
        return upsamplers

    def forward_ws_split(self, ws, blocks):
        block_ws, w_idx = [], 0
        for ir, res in enumerate(self.block_resolutions):
            block = blocks[ir]
            if self.shared_rgb_style:
                w     = ws.narrow(1, w_idx, block.num_conv)
                w_img = ws.narrow(1, -block.num_torgb, block.num_torgb)  # TODO: tRGB to use the same style (?)
                block_ws.append(torch.cat([w, w_img], 1))
            else:
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
            w_idx += block.num_conv
        return block_ws

    def forward_network(self, blocks, block_ws, x, img, target_res, alpha, skip_up=False, **block_kwargs):
        imgs = []
        for index_l, (res, cur_ws) in enumerate(zip(self.block_resolutions, block_ws)):
            if res > target_res:
                break

            block  = blocks[index_l]
            block_noise = block_kwargs['voxel_noise'][index_l] if "voxel_noise" in block_kwargs else None
            x, img  = block(
                x, 
                img if not self.no_residual_img else None, 
                cur_ws,
                block_noise=block_noise,
                skip_up=skip_up,
                **block_kwargs)

            imgs += [img]
        return imgs


@persistence.persistent_class
class NeRFInput(Upsampler):
    """ Instead of positional encoding, it learns additional features for each points.
        However, it is important to normalize the input points 
    """
    output_mode  = 'none'    # tri_plane_reshape, tri_plane_concat, single_plane
    input_mode   = 'random'  # coordinates

    architecture = 'skip'

    # only useful for triplane/volume inputs
    in_res       = 4
    out_res      = 256
    out_dim      = 32
    split_size   = 64

    # only useful for hashtable inputs
    hash_n_min   = 16
    hash_n_max   = 512
    hash_size    = 16
    hash_level   = 16
    hash_dim_in  = 32
    hash_dim_mid = None
    hash_dim_out = 2
    hash_n_layer = 4 
    hash_mode    = 'fast_hash'  # grid_hash (like volumes)

    keep_posenc  = -1
    keep_nerf_latents = False

    def build_network(self, w_dim, **block_kwargs):
        # change global settings for input field.
        kwargs_copy = copy.deepcopy(block_kwargs)
        kwargs_copy['kernel_size'] = 3
        kwargs_copy['upsample_mode'] = 'default'
        kwargs_copy['use_noise'] = True
        kwargs_copy['architecture'] = self.architecture
        self._flag = 0
        
        assert self.input_mode == 'random', \
            "currently only support normal StyleGAN2. in the future we may work on other inputs."

        # plane-based inputs with modulated 2D convolutions
        if self.output_mode   == 'tri_plane_reshape':
            self.img_channels, in_channels, const = 3 * self.out_dim, 0, None        
        elif self.output_mode == 'tri_plane_concat':  # xy, xz and yz planes are not shared #
            self.img_channels, in_channels = self.out_dim, self.channel_max
            const = torch.nn.Parameter(torch.randn([3, in_channels, self.in_res, self.in_res]))
        elif self.output_mode == 'tri_plane_reshape_extend':
            self.img_channels, in_channels, const = 3 * self.out_dim + self.split_size, 0, None
            const = torch.nn.Parameter(torch.randn([in_channels, self.in_res, self.in_res * 3]))
        elif self.output_mode == 'multi_planes':
            self.img_channels, in_channels, const = self.out_dim * self.split_size, 0, None
            kwargs_copy['architecture'] = 'orig'
        
        # volume-based inputs with modulated 3D convolutions
        elif self.output_mode == '3d_volume':  # use 3D convolution to generate
            kwargs_copy['architecture'] = 'orig'
            kwargs_copy['mode'] = '3d'
            self.img_channels, in_channels, const = self.out_dim, 0, None        
        elif self.output_mode == 'ms_volume':  # multi-resolution voulume, between hashtable and volumes
            kwargs_copy['architecture'] = 'orig'
            kwargs_copy['mode'] = '3d'
            self.img_channels, in_channels, const = self.out_dim, 0, None

        # embedding-based inputs with modulated MLPs
        elif self.output_mode == 'hash_table':
            if self.hash_mode == 'grid_hash':
                assert self.hash_size % 3 == 0, "needs to be 3D"
            kwargs_copy['hash_size'], self._flag = 2 ** self.hash_size, 1
            assert self.hash_dim_out * self.hash_level == self.out_dim, "size must matched"
            return self.build_modulated_embedding(w_dim, **kwargs_copy)
        
        elif self.output_mode == 'ms_nerf_hash':
            self.hash_mode, self._flag = 'grid_hash', 2
            ms_nerf = NeRFBlock({
                'rgb_out_dim': self.hash_dim_out * self.hash_level,  # HACK
                'magnitude_ema_beta': block_kwargs['magnitude_ema_beta'],
                'no_sigma': True, 'predict_rgb': False, 'add_rgb': False,
                'n_freq_posenc': 5,
            })
            self.num_ws = ms_nerf.num_ws
            return [{'block': ms_nerf, 'num_ws': ms_nerf.num_ws, 'name': 'ms_nerf'}]
            
        else:
            raise NotImplementedError

        networks = super().build_network(w_dim, in_channels, **kwargs_copy)
        if const is not None:
            networks.append({'block': const, 'num_ws': 0, 'name': 'const'})
        return networks

    def forward_ws_split(self, ws, blocks):
        if self._flag == 1:
            return ws.split(1, dim=1)[:len(blocks)-1]
        elif self._flag == 0:
            return super().forward_ws_split(ws, blocks)
        else:
            return ws  # do not split

    def forward_network(self, blocks, block_ws, batch_size, **block_kwargs):
        x, img, out = None, None, None
        def _forward_conv_networks(x, img, blocks, block_ws):
            for index_l, (res, cur_ws) in enumerate(zip(self.block_resolutions, block_ws)):
                x, img  = blocks[index_l](x, img, cur_ws, **block_kwargs)
            return img

        def _forward_ffn_networks(x, blocks, block_ws):  
            #TODO: FFN is implemented as 1x1 conv for now #
            h, w = dividable(x.size(0))
            x = repeat(x, 'n d -> b n d', b=batch_size)
            x = rearrange(x, 'b (h w) d -> b d h w', h=h, w=w)
            for index_l, cur_ws in enumerate(block_ws):
                block, cur_ws = blocks[index_l], cur_ws[:, 0]
                x = block(x, cur_ws)
            return x

        # tri-plane outputs
        if self.output_mode == 'tri_plane_reshape':
            img = _forward_conv_networks(x, img, blocks, block_ws)
            out = ('tri_plane', rearrange(img, 'b (s c) h w -> b s c h w', s=3))
        elif self.output_mode == 'tri_plane_concat':
            x, blocks = blocks[-1], blocks[:-1]
            x = repeat(x, 's d h w -> (b s) d h w', b=batch_size)
            img = _forward_conv_networks(x, img, blocks, block_ws)
            out = ('tri_plane', rearrange(img, '(b s) d h w -> b s d h w', s=3))
        elif self.output_mode == 'tri_plane_reshape_extend':
            img = _forward_conv_networks(x, img, blocks, block_ws)
            den, img = img[:, -self.split_size:], img[:, :-self.split_size]
            out = ('tri_plane', rearrange(img, 'b (s c) h w -> b s c h w', s=3),
                   'volume', rearrange(den, 'b d h w -> b () d h w'))  # additional density volume

        # volume/3d voxel outputs
        elif self.output_mode == 'multi_planes':
            img = _forward_conv_networks(x, img, blocks, block_ws)
            out = ('volume', rearrange(img, 'b (s c) h w -> b s c h w', s=self.out_dim))
        elif self.output_mode == '3d_volume':
            img = _forward_conv_networks(x, img, blocks, block_ws)
            out = ('volume', img)

        # multi-resolution 3d volume outputs (similar to hash-table)
        elif self.output_mode == 'ms_volume':
            img = _forward_conv_networks(x, img, blocks, block_ws)
            out = ('ms_volume', rearrange(img, 'b (l m) d h w -> b l m d h w', l=self.hash_level))
            
        # hash-table outputs (need hash sample implemented #TODO#
        elif self.output_mode == 'hash_table':
            x, blocks = blocks[-1], blocks[:-1]
            if len(blocks) > 0:
                x = _forward_ffn_networks(x, blocks, block_ws)
                out = ('hash_table', rearrange(x, 'b d h w -> b (h w) d'))
            else:
                out = ('hash_table', repeat(x, 'n d -> b n d', b=batch_size))
        elif self.output_mode == 'ms_nerf_hash':
            # prepare inputs for nerf
            x = torch.linspace(-1, 1, steps=self.out_res, device=block_ws.device)
            x = torch.stack(torch.meshgrid(x,x,x), -1).reshape(-1, 3)
            x = repeat(x, 'n s -> b n s', b=block_ws.size(0))
            x = blocks[0](x, None, ws=block_ws, shape=[block_ws.size(0), 32, 32, 32])[0]
            x = rearrange(x, 'b (d h w) (l m) -> b l m d h w', l=self.hash_level, d=32, h=32, w=32)
            out = ('ms_volume', x)
            
        else:
            raise NotImplementedError

        return out

    def query_input_features(self, p_i, input_feats, p_shape, bound, grad_inputs=False):
        batch_size, height, width, n_steps = p_shape        
        p_i = p_i / bound
        
        if input_feats[0] == 'tri_plane':
            # TODO!! Our world space, x->depth, y->width, z->height
            lh, lw = dividable(n_steps)
            p_ds = rearrange(p_i, 'b (h w l m) d -> b (l h) (m w) d',
                b=batch_size, h=height, w=width, l=lh, m=lw).split(1, dim=-1)
            px, py, pz = p_ds[0], p_ds[1], p_ds[2]
            
            # HACK here
            if len(input_feats) == 4:  pz = -pz

            # project points onto three planes
            p_xy = torch.cat([px, py], -1)
            p_xz = torch.cat([px, pz], -1)
            p_yz = torch.cat([py, pz], -1)
            p_gs = torch.cat([p_xy, p_xz, p_yz], 0)
            f_in = torch.cat([input_feats[1][:, i] for i in range(3)], 0)
            # p_f  = F.grid_sample(f_in, p_gs, mode='bilinear', align_corners=False)
            p_f  = grid_sample(f_in, p_gs)  # gradient-fix bilinear interpolation
            p_f  = sum(p_f[i * batch_size: (i+1) * batch_size] for i in range(3))
            p_f  = rearrange(p_f, 'b d (l h) (m w) -> b (h w l m) d', l=lh, m=lw)
        
        elif input_feats[0] == 'volume':
            # TODO!! Our world space, x->depth, y->width, z->height
            # (width-c, height-c, depth-c), volume (B x N x D x H x W)

            p_ds  = rearrange(p_i, 'b (h w s) d -> b s h w d',
                b=batch_size, h=height, w=width, s=n_steps).split(1, dim=-1)
            px, py, pz = p_ds[0], p_ds[1], p_ds[2]
            p_yzx = torch.cat([py, -pz, px], -1)
            p_f   = F.grid_sample(input_feats[1], p_yzx, mode='bilinear', align_corners=False)
            p_f   = rearrange(p_f, 'b c s h w -> b (h w s) c')

        elif input_feats[0] == 'ms_volume':
            # TODO!! Multi-resolution volumes (experimental)
            # for smoothness, maybe we should expand the volume? (TODO)
            # print(p_i.shape)
            ms_v = input_feats[1].new_zeros(
                batch_size, self.hash_level, self.hash_dim_out, self.out_res+1, self.out_res+1, self.out_res+1)
            ms_v[..., 1:, 1:, 1:] = input_feats[1].flip([3,4,5])
            ms_v[..., :self.out_res, :self.out_res, :self.out_res] = input_feats[1]
            v_size = ms_v.size(-1)

            # multi-resolutions
            b = math.exp((math.log(self.hash_n_max) - math.log(self.hash_n_min))/(self.hash_level-1))
            hash_res_ls = [round(self.hash_n_min * b ** l) for l in range(self.hash_level)]

            # prepare interpolate grids
            p_ds = rearrange(p_i, 'b (h w s) d -> b s h w d',
                b=batch_size, h=height, w=width, s=n_steps).split(1, dim=-1)
            px, py, pz = p_ds[0], p_ds[1], p_ds[2]
            p_yzx = torch.cat([py, -pz, px], -1)
            p_yzx = ((p_yzx + 1) / 2).clamp(min=0, max=1)     # normalize to 0~1 (just for safe)
            p_yzx = torch.stack([p_yzx if n < v_size else torch.fmod(p_yzx * n, v_size) / v_size for n in hash_res_ls], 1)
            p_yzx = (p_yzx * 2 - 1).view(-1, n_steps, height, width, 3)
            
            ms_v  = ms_v.view(-1, self.hash_dim_out, v_size, v_size, v_size)  # back to -1~1
            p_f   = F.grid_sample(ms_v, p_yzx, mode='bilinear', align_corners=False)
            p_f   = rearrange(p_f, '(b l) c s h w -> b (h w s) (l c)', l=self.hash_level)
        
        elif input_feats[0] == 'hash_table':
            # TODO:!! Experimental code trying to learn hashtable used in (maybe buggy)
            # https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.pdf
            
            p_xyz = ((p_i + 1) / 2).clamp(min=0, max=1)  # normalize to 0~1
            p_f = hash_sample(
                p_xyz, input_feats[1], self.offsets.to(p_xyz.device), 
                self.beta, self.hash_n_min, grad_inputs, mode=self.hash_mode)

        else:
            raise NotImplementedError

        if self.keep_posenc > -1:
            if self.keep_posenc > 0:
                p_f = torch.cat([p_f, positional_encoding(p_i, self.keep_posenc, use_pos=True)], -1)
            else:
                p_f = torch.cat([p_f, p_i], -1)
            
        return p_f

    def build_hashtable_info(self, hash_size):
        self.beta = math.exp((math.log(self.hash_n_max) - math.log(self.hash_n_min)) / (self.hash_level-1))
        self.hash_res_ls = [round(self.hash_n_min * self.beta ** l) for l in range(self.hash_level)]
        offsets, offset = [], 0
        for i in range(self.hash_level):
            resolution = self.hash_res_ls[i]
            params_in_level = min(hash_size, (resolution + 1) ** 3)
            offsets.append(offset)
            offset += params_in_level
        offsets.append(offset)
        self.offsets = torch.from_numpy(np.array(offsets, dtype=np.int32))
        return offset

    def build_modulated_embedding(self, w_dim, hash_size, **block_kwargs):
        # allocate parameters
        offset = self.build_hashtable_info(hash_size)
        hash_const = torch.nn.Parameter(torch.zeros(
            [offset, self.hash_dim_in if self.hash_n_layer > -1 else self.hash_dim_out]))
        hash_const.data.uniform_(-1e-4, 1e-4)

        hash_networks = []
        if self.hash_n_layer > -1:
            input_dim     = self.hash_dim_in
            for l in range(self.hash_n_layer):
                output_dim = self.hash_dim_mid if self.hash_dim_mid is not None else self.hash_dim_in
                hash_networks.append({
                    'block': Style2Layer(input_dim, output_dim, w_dim),
                    'num_ws': 1, 'name': f'hmlp{l}'
                })
                input_dim  = output_dim 
            hash_networks.append({
                'block': ToRGBLayer(input_dim, self.hash_dim_out, w_dim, kernel_size=1),
                'num_ws': 1, 'name': 'hmlpout'})
        hash_networks.append({'block': hash_const, 'num_ws': 0, 'name': 'hash_const'})
        self.num_ws = sum([h['num_ws'] for h in hash_networks])
        return hash_networks
        

@persistence.persistent_class
class NeRFSynthesisNetwork(torch.nn.Module):
    def __init__(self,
        w_dim,                        # Intermediate latent (W) dimensionality.
        img_resolution,               # Output image resolution.
        img_channels,                 # Number of color channels.
        channel_base      = 1,
        channel_max       = 1024,

        # module settings
        camera_kwargs     = {},
        renderer_kwargs   = {},
        upsampler_kwargs  = {},
        input_kwargs      = {},
        foreground_kwargs = {},
        background_kwargs = {},

        # nerf space settings
        z_dim             = 256,
        z_dim_bg          = 128,
        rgb_out_dim       = 256,
        rgb_out_dim_bg    = None,
        resolution_vol    = 32,
        resolution_start  = None,
        progressive       = True,
        prog_nerf_only    = False,
        interp_steps      = None,   # (optional) "start_step:final_step"

        # others (regularization)
        regularization    = [],     # nv_beta, nv_vol 
        predict_camera    = False,
        n_reg_samples     = 0,
        reg_full          = False,
        
        cam_based_sampler = False,
        rectangular       = None,
        freeze_nerf       = False,
        **block_kwargs,            # Other arguments for SynthesisBlock.
    ):
        assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0
        super().__init__()

        # dimensions
        self.w_dim            = w_dim
        self.z_dim            = z_dim
        self.z_dim_bg         = z_dim_bg
        self.num_ws           = 0
        self.rgb_out_dim      = rgb_out_dim
        self.rgb_out_dim_bg   = rgb_out_dim_bg if rgb_out_dim_bg is not None else rgb_out_dim

        self.img_resolution   = img_resolution
        self.resolution_vol   = resolution_vol if resolution_vol < img_resolution else img_resolution
        self.resolution_start = resolution_start if resolution_start is not None else resolution_vol
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels     = img_channels
        
        # number of samples
        self.n_reg_samples = n_reg_samples
        self.reg_full = reg_full
        self.use_noise = block_kwargs.get('use_noise', False)
        
        # ---------------------------------- Initialize Modules ---------------------------------------- -#
        # camera module
        self.C = CameraRay(camera_kwargs, **block_kwargs)
        
        # input encoding module
        if (len(input_kwargs) > 0) and (input_kwargs['output_mode'] != 'none'):  # using synthezied inputs
            input_kwargs['channel_base'] = input_kwargs.get('channel_base', channel_base)
            input_kwargs['channel_max']  = input_kwargs.get('channel_max', channel_max)
            self.I = NeRFInput(input_kwargs, **block_kwargs)
        else:
            self.I = None

        # volume renderer module
        self.V = VolumeRenderer(renderer_kwargs, camera_ray=self.C, input_encoding=self.I, **block_kwargs)

        # upsampler module
        upsampler_kwargs.update(dict(
            img_channels=img_channels,
            in_res=resolution_vol, 
            out_res=img_resolution, 
            channel_max=channel_max, 
            channel_base=channel_base))
        self.U = Upsampler(upsampler_kwargs, **block_kwargs)

        # full model resolutions
        self.block_resolutions = copy.deepcopy(self.U.block_resolutions)
        if self.resolution_start < self.resolution_vol:
            r = self.resolution_vol
            while r > self.resolution_start:
                self.block_resolutions.insert(0, r)
                r = r // 2

        self.predict_camera = predict_camera
        if predict_camera:  # encoder side camera predictor (not very useful)
            self.camera_generator = CameraGenerator()

        # ray level choices
        self.regularization   = regularization
        self.margin           = block_kwargs.get('margin', 0)
        self.activation       = block_kwargs.get('activation', 'lrelu')
        self.rectangular_crop = rectangular  # [384, 512] ??
    
        # nerf (foregournd/background)
        foreground_kwargs.update(dict(
            z_dim=self.z_dim, 
            w_dim=w_dim, 
            rgb_out_dim=self.rgb_out_dim,
            activation=self.activation))

        # disable positional encoding if input encoding is given
        if self.I is not None:  
            foreground_kwargs.update(dict(
                disable_latents=(not self.I.keep_nerf_latents), 
                input_dim=self.I.out_dim + 3 * (2 * self.I.keep_posenc + 1) 
                    if self.I.keep_posenc > -1 else self.I.out_dim,
                positional_encoding='none'))
        
        self.fg_nerf = NeRFBlock(foreground_kwargs)
        self.num_ws += self.fg_nerf.num_ws

        if not self.V.no_background:
            background_kwargs.update(dict(
                z_dim=self.z_dim_bg, w_dim=w_dim, 
                rgb_out_dim=self.rgb_out_dim_bg,
                activation=self.activation))
            self.bg_nerf = NeRFBlock(background_kwargs)
            self.num_ws += self.bg_nerf.num_ws
        else:
            self.bg_nerf = None
        
        # ---------------------------------- Build Networks ---------------------------------------- -#
        # input encoding (optional)
        if self.I is not None:
            assert self.V.no_background, "does not support background field"
            nerf_inputs = self.I.build_network(w_dim, **block_kwargs)
            self.input_block_names = ['in_' + i['name'] for i in nerf_inputs]
            self.num_ws += sum([i['num_ws'] for i in nerf_inputs])
            for i in nerf_inputs:
                setattr(self, 'in_' + i['name'], i['block'])
                
        # upsampler
        upsamplers = self.U.build_network(w_dim, self.fg_nerf.rgb_out_dim, **block_kwargs)
        if len(upsamplers) > 0:
            self.block_names = [u['name'] for u in upsamplers]
            self.num_ws += sum([u['num_ws'] for u in upsamplers])
            for u in upsamplers:
                setattr(self, u['name'], u['block'])

        # data-sampler
        if cam_based_sampler:
            self.sampler = (CameraQueriedSampler, {'camera_module': self.C})
            
        # other hyperameters
        self.progressive_growing   = progressive
        self.progressive_nerf_only = prog_nerf_only
        assert not (self.progressive_growing and self.progressive_nerf_only)
        if prog_nerf_only:
            assert (self.n_reg_samples == 0) and (not reg_full), "does not support regularization"

        self.register_buffer("alpha", torch.scalar_tensor(-1))
        if predict_camera:
            self.num_ws += 1  # additional w for camera
        self.freeze_nerf = freeze_nerf
        self.steps = None
        self.interp_steps = [int(a) for a in interp_steps.split(':')] \
            if interp_steps is not None else None  #TODO two-stage training trick (from EG3d paper, not working so far)

    def set_alpha(self, alpha):
        if alpha is not None:
            self.alpha.fill_(alpha)

    def set_steps(self, steps):
        if hasattr(self, "steps"):
            if self.steps is not None:
                self.steps = self.steps * 0 + steps / 1000.0
            else:
                self.steps = steps / 1000.0

    def forward(self, ws, **block_kwargs):
        block_ws, imgs, rand_imgs = [], [], []
        batch_size = block_kwargs['batch_size'] = ws.size(0)
        n_levels, end_l, _, target_res = self.get_current_resolution()

        # cameras, background codes        
        if "camera_matrices" not in block_kwargs:
            if 'camera_mode' in block_kwargs:
                block_kwargs["camera_matrices"] = self.get_camera(batch_size, device=ws.device, mode=block_kwargs["camera_mode"])
            else:
                if self.predict_camera:
                    rand_mode = ws.new_zeros(ws.size(0), 2)
                    if self.C.gaussian_camera:
                        rand_mode = rand_mode.normal_()
                        pred_mode = self.camera_generator(rand_mode)
                    else:
                        rand_mode = rand_mode.uniform_()
                        pred_mode = self.camera_generator(rand_mode - 0.5)
                    mode = rand_mode if self.alpha <= 0 else rand_mode + pred_mode * 0.1
                    block_kwargs["camera_matrices"] = self.get_camera(batch_size, device=ws.device, mode=mode)
                else:   
                    block_kwargs["camera_matrices"] = self.get_camera(batch_size, device=ws.device)
            
            if ('camera_RT' in block_kwargs) or ('camera_UV' in block_kwargs):
                camera_matrices = list(block_kwargs["camera_matrices"])
                camera_mask = torch.rand(batch_size).type_as(camera_matrices[1]).lt(self.alpha)
                if 'camera_RT' in block_kwargs:
                    image_RT = block_kwargs['camera_RT'].reshape(-1, 4, 4)
                    camera_matrices[1][camera_mask] = image_RT[camera_mask]  # replacing with inferred cameras
                else:  # sample uv instead of sampling the extrinsic matrix
                    image_UV = block_kwargs['camera_UV']
                    image_RT = self.get_camera(batch_size, device=ws.device, mode=image_UV, force_uniform=True)[1]           
                    camera_matrices[1][camera_mask] = image_RT[camera_mask]  # replacing with inferred cameras
                    camera_matrices[2][camera_mask] = image_UV[camera_mask]  # replacing with inferred uvs
                block_kwargs["camera_matrices"] = tuple(camera_matrices)

        if "latent_codes" not in block_kwargs:
            block_kwargs["latent_codes"] = self.get_latent_codes(batch_size, device=ws.device)

        # deal with roll in cameras
        block_kwargs['theta'] = self.C.get_roll(ws, self.training, **block_kwargs)
        
        # generate features for input points (Optional, default not use)
        with torch.autograd.profiler.record_function('nerf_input_feats'):
            if self.I is not None:
                ws = ws.to(torch.float32)
                blocks   = [getattr(self, name) for name in self.input_block_names]
                block_ws = self.I.forward_ws_split(ws, blocks)
                nerf_input_feats = self.I.forward_network(blocks, block_ws, **block_kwargs)
                ws = ws[:, self.I.num_ws:]
            else:
                nerf_input_feats = None

        # prepare for NeRF part
        with torch.autograd.profiler.record_function('prepare_nerf_path'):
            if self.progressive_nerf_only and (self.alpha > -1):
                cur_resolution = int(self.resolution_start * (1 - self.alpha) + self.resolution_vol * self.alpha)
            elif (end_l == 0) or len(self.block_resolutions) == 0:
                cur_resolution = self.resolution_start
            else:
                cur_resolution = self.block_resolutions[end_l-1]

            vol_resolution  = self.resolution_vol if self.resolution_vol < cur_resolution else cur_resolution
            nerf_resolution = vol_resolution
            if (self.interp_steps is not None) and (self.steps is not None) and (self.alpha > 0):  # interpolation trick (maybe work??)
                if self.steps < self.interp_steps[0]:
                    nerf_resolution = vol_resolution // 2
                elif self.steps < self.interp_steps[1]:
                    nerf_resolution = (self.steps - self.interp_steps[0]) / (self.interp_steps[1] - self.interp_steps[0])
                    nerf_resolution = int(nerf_resolution * (vol_resolution / 2) + vol_resolution / 2)
            
            vol_pixels, tgt_pixels = self.C.prepare_pixels(self.img_resolution, cur_resolution, nerf_resolution, **block_kwargs)
            if (end_l > 0) and (self.n_reg_samples > 0) and self.training:
                rand_pixels, rand_indexs = self.C.prepare_pixels_regularization(tgt_pixels, self.n_reg_samples)
            else:
                rand_pixels, rand_indexs = None, None
                
            if self.fg_nerf.num_ws > 0:  # use style vector instead of latent codes?
                block_kwargs["styles"] = ws[:, :self.fg_nerf.num_ws]
                ws = ws[:, self.fg_nerf.num_ws:]
            if (self.bg_nerf is not None) and self.bg_nerf.num_ws > 0:
                block_kwargs["styles_bg"] = ws[:, :self.bg_nerf.num_ws]
                ws = ws[:, self.bg_nerf.num_ws:]
        
        # volume rendering
        with torch.autograd.profiler.record_function('nerf'):
            if (rand_pixels is not None) and self.training:
                vol_pixels = (vol_pixels, rand_pixels)
            outputs = self.V.forward_volume_rendering(
                nerf_modules=(self.fg_nerf, self.bg_nerf),
                vol_pixels=vol_pixels, 
                nerf_input_feats=nerf_input_feats,
                return_full=self.reg_full,
                alpha=self.alpha,
                **block_kwargs)
            
            reg_loss = outputs.get('reg_loss', {})
            x, img, _ = self.V.post_process_outputs(outputs['full_out'], self.freeze_nerf)
            if nerf_resolution < vol_resolution:
                x   = F.interpolate(x,   vol_resolution, mode='bilinear', align_corners=False)
                img = F.interpolate(img, vol_resolution, mode='bilinear', align_corners=False)
            
            # early output from the network (used for visualization)
            if 'meshes' in block_kwargs:
                from dnnlib.geometry import render_mesh
                block_kwargs['voxel_noise'] = render_mesh(block_kwargs['meshes'], block_kwargs["camera_matrices"])
            
            if (len(self.U.block_resolutions) == 0) or \
                (x is None) or \
                (block_kwargs.get("render_option", None) is not None and 
                    'early' in block_kwargs['render_option']):
                if 'value' in block_kwargs['render_option']:
                    img = x[:,:3]
                    img = img / img.norm(dim=1, keepdim=True)
                assert img is not None, "need to add RGB"
                return img

            if 'rand_out' in outputs:
                x_rand, img_rand, rand_probs = self.V.post_process_outputs(outputs['rand_out'], self.freeze_nerf)
                lh, lw  = dividable(rand_probs.size(1))
                rand_imgs += [img_rand]

            # append low-resolution image
            if img is not None:
                if self.progressive_nerf_only and (img.size(-1) < self.resolution_vol):
                    x   = upsample(x,   self.resolution_vol)
                    img = upsample(img, self.resolution_vol)
                block_kwargs['img_nerf'] = img

        # Use 2D upsampler
        if (cur_resolution > self.resolution_vol) or self.progressive_nerf_only:
            imgs += [img]

            # 2D feature map upsampling
            with torch.autograd.profiler.record_function('upsampling'):
                ws       = ws.to(torch.float32)
                blocks   = [getattr(self, name) for name in self.block_names]
                block_ws = self.U.forward_ws_split(ws, blocks)               
                imgs    += self.U.forward_network(blocks, block_ws, x, img, target_res, self.alpha, **block_kwargs)
                img      = imgs[-1]
                if len(rand_imgs) > 0:   # nerf path regularization
                    rand_imgs += self.U.forward_network(
                        blocks, block_ws, x_rand, img_rand, target_res, self.alpha, skip_up=True, **block_kwargs)
                    img_rand = rand_imgs[-1]
            
            with torch.autograd.profiler.record_function('rgb_interp'):
                if (self.alpha > -1) and (not self.progressive_nerf_only) and self.progressive_growing:
                    if (self.alpha < 1) and (self.alpha > 0):     
                        alpha, _ = math.modf(self.alpha * n_levels)
                        img_nerf = imgs[-2]
                        if img_nerf.size(-1) < img.size(-1):  # need upsample image
                            img_nerf = upsample(img_nerf, 2 * img_nerf.size(-1))
                        img = img_nerf * (1 - alpha) + img * alpha
                        if len(rand_imgs) > 0:
                            img_rand = rand_imgs[-2] * (1 - alpha) + img_rand * alpha

            with torch.autograd.profiler.record_function('nerf_path_reg_loss'):
                if len(rand_imgs) > 0: # and self.training:  # random pixel regularization??            
                    assert self.progressive_growing
                    if self.reg_full:     # aggregate RGB in the end.
                        lh, lw = img_rand.size(2) // self.n_reg_samples, img_rand.size(3) // self.n_reg_samples
                        img_rand = rearrange(img_rand, 'b d (l h) (m w) -> b d (l m) h w', l=lh, m=lw)
                        img_rand = (img_rand * rand_probs[:, None]).sum(2)
                        if self.V.white_background:
                            img_rand = img_rand + (1 - rand_probs.sum(1, keepdim=True))
                    rand_indexs = repeat(rand_indexs, 'b n -> b d n', d=img_rand.size(1))
                    img_ff = rearrange(rearrange(img, 'b d l h -> b d (l h)').gather(2, rand_indexs), 'b d (l h) -> b d l h', l=self.n_reg_samples)
                
                    def l2(img_ff, img_nf):
                        batch_size = img_nf.size(0)
                        return ((img_ff - img_nf) ** 2).sum(1).reshape(batch_size, -1).mean(-1, keepdim=True)
                    
                    reg_loss['reg_loss'] = l2(img_ff, img_rand) * 2.0

        if len(reg_loss) > 0:
            for key in reg_loss:
                block_kwargs[key] = reg_loss[key]
        
        if self.rectangular_crop is not None:   # in case rectangular 
            h, w = self.rectangular_crop
            c = int(img.size(-1) * (1 - h / w) / 2)
            mask = torch.ones_like(img)
            mask[:, :, c:-c, :] = 0
            img = img.masked_fill(mask > 0, -1)

        block_kwargs['img'] = img
        return block_kwargs

    def get_current_resolution(self):
        n_levels = len(self.block_resolutions)
        if not self.progressive_growing:
            end_l = n_levels
        elif (self.alpha > -1) and (not self.progressive_nerf_only):
            if self.alpha == 0:
                end_l = 0
            elif self.alpha == 1:
                end_l = n_levels
            elif self.alpha < 1:
                end_l = int(math.modf(self.alpha * n_levels)[1] + 1)
        else:
            end_l = n_levels
        target_res = self.resolution_start if end_l <= 0 else self.block_resolutions[end_l-1]
        before_res = self.resolution_start if end_l <= 1 else self.block_resolutions[end_l-2]
        return n_levels, end_l, before_res, target_res

    def get_latent_codes(self, batch_size=32, device="cpu", tmp=1.):
        z_dim, z_dim_bg = self.z_dim, self.z_dim_bg

        def sample_z(*size):
            torch.randn(*size).to(device)
            return torch.randn(*size).to(device) * tmp

        z_shape_obj = sample_z(batch_size, z_dim)
        z_app_obj = sample_z(batch_size, z_dim)
        z_shape_bg = sample_z(batch_size, z_dim_bg) if not self.V.no_background else None
        z_app_bg = sample_z(batch_size, z_dim_bg) if not self.V.no_background else None
        return z_shape_obj, z_app_obj, z_shape_bg, z_app_bg

    def get_camera(self, *args, **kwargs):
        return self.C.get_camera(*args, **kwargs)   # for compatibility


@persistence.persistent_class
class Discriminator(torch.nn.Module):
    def __init__(self,
        c_dim,                          # Conditioning label (C) dimensionality.
        img_resolution,                 # Input resolution.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        channel_base        = 1,        # Overall multiplier for the number of channels.
        channel_max         = 512,      # Maximum number of channels in any layer.
        num_fp16_res        = 0,        # Use FP16 for the N highest resolutions.
        conv_clamp          = None,     # Clamp the output of convolution layers to +-X, None = disable clamping.
        cmap_dim            = None,     # Dimensionality of mapped conditioning label, None = default.
        lowres_head         = None,     # add a low-resolution discriminator head
        dual_discriminator  = False,    # add low-resolution (NeRF) image 
        dual_input_ratio    = None,     # optional another low-res image input, which will be interpolated to the main input
        block_kwargs        = {},       # Arguments for DiscriminatorBlock.
        mapping_kwargs      = {},       # Arguments for MappingNetwork.
        epilogue_kwargs     = {},       # Arguments for DiscriminatorEpilogue.
        upsample_type       = 'default',
        
        progressive         = False,
        resize_real_early   = False,    # Peform resizing before the training loop
        enable_ema          = False,    # Additionally save an EMA checkpoint
        
        predict_camera      = False,    # Learn camera predictor as InfoGAN
        predict_9d_camera   = False,    # Use 9D camera distribution
        predict_3d_camera   = False,    # Use 3D camera (u, v, r), assuming camera is on the unit sphere
        no_camera_condition = False,    # Disable camera conditioning in the discriminator
        saperate_camera     = False,    # by default, only works in the lowest resolution.
        **unused
    ):
        super().__init__()
        # setup parameters
        self.img_resolution      = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels        = img_channels
        self.block_resolutions   = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]
        self.architecture        = architecture
        self.lowres_head         = lowres_head
        self.dual_input_ratio    = dual_input_ratio
        self.dual_discriminator  = dual_discriminator
        self.upsample_type       = upsample_type
        self.progressive         = progressive
        self.resize_real_early   = resize_real_early
        self.enable_ema          = enable_ema
        self.predict_camera      = predict_camera
        self.predict_9d_camera   = predict_9d_camera
        self.predict_3d_camera   = predict_3d_camera
        self.no_camera_condition = no_camera_condition
        self.separate_camera     = saperate_camera
        if self.progressive:
            assert self.architecture == 'skip', "not supporting other types for now."
        if self.dual_input_ratio is not None:  # similar to EG3d, concat low/high-res images
            self.img_channels    = self.img_channels * 2
        if self.predict_camera:
            assert not (self.predict_9d_camera and self.predict_3d_camera), "cannot achieve at the same time"
        channel_base = int(channel_base * 32768)
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)
        
        # camera prediction module
        self.c_dim = c_dim
        if predict_camera:
            if not self.no_camera_condition:  
                if self.predict_3d_camera:
                    self.c_dim = out_dim = 3     # (u, v) on the sphere
                else:
                    self.c_dim = 16              # extrinsic 4x4 (for now)
                    if self.predict_9d_camera:
                        out_dim = 9
                    else:
                        out_dim = 16            
            self.projector = EqualConv2d(channels_dict[4], out_dim, 4, padding=0, bias=False)
                  
        if cmap_dim is None:
            cmap_dim = channels_dict[4]
        if self.c_dim == 0:
            cmap_dim = 0
        if self.c_dim > 0:
            self.mapping = MappingNetwork(z_dim=0, c_dim=self.c_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None, **mapping_kwargs)
            
        # main discriminator blocks
        common_kwargs = dict(img_channels=self.img_channels, architecture=architecture, conv_clamp=conv_clamp)    
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
        
        # dual discriminator or separate camera predictor
        if self.separate_camera or self.dual_discriminator:
            cur_layer_idx = 0
            for res in [r for r in self.block_resolutions if r <= self.lowres_head]:
                in_channels = channels_dict[res] if res < img_resolution else 0
                tmp_channels = channels_dict[res]
                out_channels = channels_dict[res // 2]
                block = DiscriminatorBlock(in_channels, tmp_channels, out_channels, resolution=res,
                    first_layer_idx=cur_layer_idx, use_fp16=False, **block_kwargs, **common_kwargs)
                setattr(self, f'c{res}', block)
                cur_layer_idx += block.num_layers
            
        # final output module
        self.b4 = DiscriminatorEpilogue(channels_dict[4], cmap_dim=cmap_dim, resolution=4, **epilogue_kwargs, **common_kwargs)
        self.register_buffer("alpha", torch.scalar_tensor(-1))

    def set_alpha(self, alpha):
        if alpha is not None:
            self.alpha = self.alpha * 0 + alpha
    
    def set_resolution(self, res):
        self.curr_status = res

    def get_estimated_camera(self, img, **block_kwargs):
        if isinstance(img, dict):
            img = img['img']
        img4cam = img.clone()
        if self.progressive and (img.size(-1) != self.lowres_head):
            img4cam = downsample(img, self.lowres_head)
                    
        c, xc = None, None
        for res in [r for r in self.block_resolutions if r <= self.lowres_head or (not self.progressive)]:
            xc, img4cam = getattr(self, f'c{res}')(xc, img4cam, **block_kwargs)
        
        if self.separate_camera:
            c = self.projector(xc)[:,:,0,0]
            if self.predict_9d_camera:
                c = camera_9d_to_16d(c)    
        return c, xc, img4cam

    def get_camera_loss(self, RT=None, UV=None, c=None):
        if UV is not None:  # UV has higher priority?
            return F.mse_loss(UV, c)
            # lu = torch.stack([(UV[:,0] - c[:, 0]) ** 2, (UV[:,0] - c[:, 0] + 1) ** 2, (UV[:,0] - c[:, 0] - 1) ** 2], 0).min(0).values
            # return torch.mean(sum(lu + (UV[:,1] - c[:, 1]) ** 2 + (UV[:,2] - c[:, 2]) ** 2))
        elif RT is not None:
            return F.smooth_l1_loss(RT.reshape(RT.size(0), -1), c) * 10
        return None

    def get_block_resolutions(self, input_img):
        block_resolutions = self.block_resolutions
        lowres_head = self.lowres_head
        alpha = self.alpha
        img_res = input_img.size(-1)
        if self.progressive and (self.lowres_head is not None) and (self.alpha > -1):
            if (self.alpha < 1) and (self.alpha > 0): 
                try:
                    n_levels, _, before_res, target_res = self.curr_status
                    alpha, index = math.modf(self.alpha * n_levels)
                    index = int(index)
                except Exception as e:  # TODO: this is a hack, better to save status as buffers.
                    before_res = target_res = img_res
                if before_res == target_res:  # no upsampling was used in generator, do not increase the discriminator
                    alpha = 0    
                block_resolutions = [res for res in self.block_resolutions if res <= target_res]
                lowres_head = before_res
            elif self.alpha == 0:
                block_resolutions = [res for res in self.block_resolutions if res <= lowres_head]
        return block_resolutions, alpha, lowres_head

    def forward(self, inputs, c=None, aug_pipe=None, return_camera=False, **block_kwargs):
        if not isinstance(inputs, dict):
            inputs = {'img': inputs}
        img = inputs['img']
        block_resolutions, alpha, lowres_head = self.get_block_resolutions(img)
        if img.size(-1) > block_resolutions[0]:
            img = downsample(img, block_resolutions[0])

        # this is to handle real images to obtain nerf-size image.
        if (self.dual_discriminator or (self.dual_input_ratio is not None)) and ('img_nerf' not in inputs):
            inputs['img_nerf'] = img    
            if self.dual_discriminator and (inputs['img_nerf'].size(-1) > self.lowres_head):  # using Conv to read image.
                inputs['img_nerf'] = downsample(inputs['img_nerf'], self.lowres_head)
            elif self.dual_input_ratio is not None:   # similar to EG3d
                if inputs['img_nerf'].size(-1) > (img.size(-1) // self.dual_input_ratio):
                    inputs['img_nerf'] = downsample(inputs['img_nerf'], img.size(-1) // self.dual_input_ratio)
                img = torch.cat([img, upsample(inputs['img_nerf'], img.size(-1))], 1)

        camera_loss = None
        RT = inputs['camera_matrices'][1].detach() if 'camera_matrices' in inputs else None
        UV = inputs['camera_matrices'][2].detach() if 'camera_matrices' in inputs else None
        
        # perform separate camera predictor or dual discriminator
        if self.dual_discriminator or self.separate_camera:
            temp_img = img if not self.dual_discriminator else inputs['img_nerf']
            c_nerf, x_nerf, img_nerf = self.get_estimated_camera(temp_img, **block_kwargs)
            if c.size(-1) == 0 and self.separate_camera:
                c = c_nerf
                if self.predict_3d_camera:
                    camera_loss = self.get_camera_loss(RT, UV, c)

        # if applied data augmentation for discriminator
        if aug_pipe is not None:
            assert self.separate_camera or (not self.predict_camera), "ada may break the camera predictor."
            img = aug_pipe(img)

        # obtain the downsampled image for progressive growing
        if self.progressive and (self.lowres_head is not None) and (self.alpha > -1) and (self.alpha < 1) and (alpha > 0):
            img0 = downsample(img, img.size(-1) // 2)
           
        x = None if (not self.progressive) or (block_resolutions[0] == self.img_resolution) \
            else getattr(self, f'b{block_resolutions[0]}').fromrgb(img)
        for res in block_resolutions:
            block = getattr(self, f'b{res}')
            if (lowres_head == res) and (self.alpha > -1) and (self.alpha < 1) and (alpha > 0):
                if self.architecture == 'skip':
                    img = img * alpha + img0 * (1 - alpha)
                if self.progressive:
                    x = x * alpha + block.fromrgb(img0) * (1 - alpha)
            x, img = block(x, img, **block_kwargs)
        
        # predict camera based on discriminator features
        if (c.size(-1) == 0) and self.predict_camera and (not self.separate_camera):
            c = self.projector(x)[:,:,0,0]
            if self.predict_9d_camera:
                c = camera_9d_to_16d(c)
            if self.predict_3d_camera:
                camera_loss = self.get_camera_loss(RT, UV, c)
                
        # camera conditional discriminator
        cmap = None
        if self.c_dim > 0:
            cc = c.clone().detach()
            cmap = self.mapping(None, cc)
        logits  = self.b4(x, img, cmap)
        if self.dual_discriminator:
            logits = torch.cat([logits, self.b4(x_nerf, img_nerf, cmap)], 0)
                
        outputs = {'logits': logits}
        if self.predict_camera and (camera_loss is not None):
            outputs['camera_loss'] = camera_loss
        if return_camera:
            outputs['camera'] = c
        return outputs

      
@persistence.persistent_class
class Encoder(torch.nn.Module):
    def __init__(self,
        img_resolution,                 # Input resolution.
        img_channels,                   # Number of input color channels.
        bottleneck_factor   = 2,        # By default, the same as discriminator we use 4x4 features
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        channel_base        = 1,        # Overall multiplier for the number of channels.
        channel_max         = 512,      # Maximum number of channels in any layer.
        num_fp16_res        = 0,        # Use FP16 for the N highest resolutions.
        conv_clamp          = None,     # Clamp the output of convolution layers to +-X, None = disable clamping
        lowres_head         = None,     # add a low-resolution discriminator head
        block_kwargs        = {},       # Arguments for DiscriminatorBlock.
        model_kwargs        = {},
        upsample_type       = 'default',
        progressive         = False,
        **unused
    ):
        super().__init__()
        self.img_resolution      = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels        = img_channels
        self.block_resolutions   = [2 ** i for i in range(self.img_resolution_log2, bottleneck_factor, -1)]
        self.architecture        = architecture
        self.lowres_head         = lowres_head
        self.upsample_type       = upsample_type
        self.progressive         = progressive
        self.model_kwargs        = model_kwargs
        self.output_mode         = model_kwargs.get('output_mode', 'styles')
        if self.progressive:
            assert self.architecture == 'skip', "not supporting other types for now."
        self.predict_camera      = model_kwargs.get('predict_camera', False)

        channel_base = int(channel_base * 32768)
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)        
        common_kwargs = dict(img_channels=self.img_channels, architecture=architecture, conv_clamp=conv_clamp)    
        cur_layer_idx = 0
        for res in self.block_resolutions:
            in_channels  = channels_dict[res] if res < img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            use_fp16 = (res >= fp16_resolution)
            block = DiscriminatorBlock(in_channels, tmp_channels, out_channels, resolution=res,
                first_layer_idx=cur_layer_idx, use_fp16=use_fp16, **block_kwargs, **common_kwargs)
            setattr(self, f'b{res}', block)
            cur_layer_idx += block.num_layers

        # this is an encoder
        if self.output_mode in ['W', 'W+', 'None']:
            self.num_ws    = self.model_kwargs.get('num_ws', 0)
            self.n_latents = self.num_ws if self.output_mode == 'W+' else (0 if self.output_mode == 'None' else 1) 
            self.w_dim     = self.model_kwargs.get('w_dim', 512)
            self.add_dim   = self.model_kwargs.get('add_dim', 0) if not self.predict_camera else 9
            self.out_dim   = self.w_dim * self.n_latents + self.add_dim
            assert self.out_dim > 0, 'output dimenstion has to be larger than 0'
            assert self.block_resolutions[-1] // 2 == 4, "make sure the last resolution is 4x4"
            self.projector = EqualConv2d(channels_dict[4], self.out_dim, 4, padding=0, bias=False)
        else:
            raise NotImplementedError
        self.register_buffer("alpha", torch.scalar_tensor(-1))

    def set_alpha(self, alpha):
        if alpha is not None:
            self.alpha.fill_(alpha)
    
    def set_resolution(self, res):
        self.curr_status = res

    def get_block_resolutions(self, input_img):
        block_resolutions = self.block_resolutions
        lowres_head = self.lowres_head
        alpha = self.alpha
        img_res = input_img.size(-1)
        if self.progressive and (self.lowres_head is not None) and (self.alpha > -1):
            if (self.alpha < 1) and (self.alpha > 0): 
                try:
                    n_levels, _, before_res, target_res = self.curr_status
                    alpha, index = math.modf(self.alpha * n_levels)
                    index = int(index)
                except Exception as e:  # TODO: this is a hack, better to save status as buffers.
                    before_res = target_res = img_res 
                if before_res == target_res:
                    # no upsampling was used in generator, do not increase the discriminator
                    alpha = 0    
                block_resolutions = [res for res in self.block_resolutions if res <= target_res]
                lowres_head = before_res
            elif self.alpha == 0:
                block_resolutions = [res for res in self.block_resolutions if res <= lowres_head]
        return block_resolutions, alpha, lowres_head

    def forward(self, inputs, **block_kwargs):
        if isinstance(inputs, dict):
            img = inputs['img']
        else:
            img = inputs

        block_resolutions, alpha, lowres_head = self.get_block_resolutions(img)
        if img.size(-1) > block_resolutions[0]:
            img = downsample(img, block_resolutions[0])

        if self.progressive and (self.lowres_head is not None) and (self.alpha > -1) and (self.alpha < 1) and (alpha > 0):
            img0 = downsample(img, img.size(-1) // 2)
           
        x = None if (not self.progressive) or (block_resolutions[0] == self.img_resolution) \
            else getattr(self, f'b{block_resolutions[0]}').fromrgb(img)

        for res in block_resolutions:
            block = getattr(self, f'b{res}')
            if (lowres_head == res) and (self.alpha > -1) and (self.alpha < 1) and (alpha > 0):
                if self.architecture == 'skip':
                    img = img * alpha + img0 * (1 - alpha)
                if self.progressive:
                    x = x * alpha + block.fromrgb(img0) * (1 - alpha)      # combine from img0           
            x, img = block(x, img, **block_kwargs)
        
        outputs = {}
        if self.output_mode in ['W', 'W+', 'None']:
            out = self.projector(x)[:,:,0,0]
            if self.predict_camera:
                out, out_cam_9d = out[:, 9:], out[:, :9]
                outputs['camera'] = camera_9d_to_16d(out_cam_9d)
            
            if self.output_mode == 'W+':
                out = rearrange(out, 'b (n s) -> b n s', n=self.num_ws, s=self.w_dim)
            elif self.output_mode == 'W':
                out = repeat(out, 'b s -> b n s', n=self.num_ws)
            else:
                out = None
            outputs['ws'] = out

        return outputs
    
# ------------------------------------------------------------------------------------------- #

class CameraQueriedSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, camera_module, nearest_neighbors=400, rank=0, num_replicas=1, device='cpu', seed=0):
        assert len(dataset) > 0

        super().__init__(dataset)
        self.dataset = dataset
        self.dataset_cameras = None
        self.seed = seed
        self.rank = rank
        self.device = device
        self.num_replicas = num_replicas
        self.C = camera_module
        self.K = nearest_neighbors
        self.B = 1000
        
    def update_dataset_cameras(self, estimator):
        import tqdm
        from torch_utils.distributed_utils import gather_list_and_concat
        output = torch.ones(len(self.dataset), 16).to(self.device)
        with torch.no_grad():
            predicted_cameras, image_indices, bsz = [], [], 64
            item_subset = [(i * self.num_replicas + self.rank) % len(self.dataset) for i in range((len(self.dataset) - 1) // self.num_replicas + 1)]
            for _, (images, _, indices) in tqdm.tqdm(enumerate(torch.utils.data.DataLoader(
                    dataset=copy.deepcopy(self.dataset), sampler=item_subset, batch_size=bsz)), 
                total=len(item_subset)//bsz+1, colour='red', desc=f'Estimating camera poses for the training set at'):
                predicted_cameras += [estimator(images.to(self.device).to(torch.float32) / 127.5 - 1)]
                image_indices += [indices.to(self.device).long()]
            predicted_cameras = torch.cat(predicted_cameras, 0)
            image_indices = torch.cat(image_indices, 0)
            if self.num_replicas > 1:
                predicted_cameras = gather_list_and_concat(predicted_cameras)
                image_indices = gather_list_and_concat(image_indices)
        output[image_indices] = predicted_cameras
        self.dataset_cameras = output        
                
    def get_knn_cameras(self):
        return torch.norm(
            self.dataset_cameras.unsqueeze(1) - 
            self.C.get_camera(self.B, self.device)[0].reshape(1,self.B,16), dim=2, p=None
        ).topk(self.K, largest=False, dim=0)[1]   # K x B

    def __iter__(self):
        order = np.arange(len(self.dataset))
        rnd = np.random.RandomState(self.seed+self.rank)
        while True:
            if self.dataset_cameras is None:
                rand_idx = rnd.randint(order.size)
                yield rand_idx
            else:
                knn_idxs = self.get_knn_cameras()
                for i in range(self.B):
                    rand_idx = rnd.randint(self.K)
                    yield knn_idxs[rand_idx, i].item()
