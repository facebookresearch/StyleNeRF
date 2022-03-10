# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from email import generator

from cv2 import DescriptorMatcher
import training
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, **kwargs): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(
        self, device, G_mapping, G_synthesis, D, 
        G_encoder=None, augment_pipe=None, D_ema=None,
        style_mixing_prob=0.9, r1_gamma=10, 
        pl_batch_shrink=2, pl_decay=0.01, pl_weight=2, other_weights=None,
        curriculum=None, alpha_start=0.0, cycle_consistency=False, label_smooth=0,
        generator_mode='random_z_random_c'):

        super().__init__()
        self.device            = device
        self.G_mapping         = G_mapping
        self.G_synthesis       = G_synthesis
        self.G_encoder         = G_encoder
        self.D                 = D
        self.D_ema             = D_ema
        self.augment_pipe      = augment_pipe
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma          = r1_gamma
        self.pl_batch_shrink   = pl_batch_shrink
        self.pl_decay          = pl_decay
        self.pl_weight         = pl_weight
        self.other_weights     = other_weights
        self.pl_mean           = torch.zeros([], device=device)
        self.curriculum        = curriculum
        self.alpha_start       = alpha_start
        self.alpha             = None
        self.cycle_consistency = cycle_consistency
        self.label_smooth      = label_smooth
        self.generator_mode    = generator_mode

        if self.G_encoder is not None:
            import lpips
            self.lpips_loss      = lpips.LPIPS(net='vgg').to(device=device)

    def set_alpha(self, steps):
        alpha = None
        if self.curriculum is not None:
            if self.curriculum == 'upsample':
                alpha = 0.0
            else:
                assert len(self.curriculum) == 2, "currently support one stage for now"
                start, end = self.curriculum
                alpha = min(1., max(0., (steps / 1e3 - start) / (end - start)))
                if self.alpha_start > 0:
                    alpha = self.alpha_start + (1 - self.alpha_start) * alpha
        self.alpha = alpha
        self.steps = steps
        self.curr_status = None

        def _apply(m):
            if hasattr(m, "set_alpha") and m != self:
                m.set_alpha(alpha)
            if hasattr(m, "set_steps") and m != self:
                m.set_steps(steps)
            if hasattr(m, "set_resolution") and m != self:
                m.set_resolution(self.curr_status)
        
        self.G_synthesis.apply(_apply)
        self.curr_status = self.resolution
        self.D.apply(_apply)
        if self.G_encoder is not None:
            self.G_encoder.apply(_apply)

    def run_G(self, z, c, sync, img=None, mode=None, get_loss=True):
        synthesis_kwargs = {'camera_mode': 'random'}
        generator_mode   = self.generator_mode if mode is None else mode

        if (generator_mode == 'image_z_random_c') or (generator_mode == 'image_z_image_c'):
            assert (self.G_encoder is not None) and (img is not None)
            with misc.ddp_sync(self.G_encoder, sync):
                ws  = self.G_encoder(img)['ws']
            if generator_mode == 'image_z_image_c':
                with misc.ddp_sync(self.D, False):
                    synthesis_kwargs['camera_RT'] = misc.get_func(self.D, 'get_estimated_camera')[0](img)
            with misc.ddp_sync(self.G_synthesis, sync):
                out = self.G_synthesis(ws, **synthesis_kwargs)            
            if get_loss:  # consistency loss given the image predicted camera (train the image encoder jointly)
                out['consist_l1_loss']    = F.smooth_l1_loss(out['img'], img['img']) * 2.0   # TODO: DEBUG
                out['consist_lpips_loss'] = self.lpips_loss(out['img'],  img['img']) * 10.0  # TODO: DEBUG
            
        elif (generator_mode == 'random_z_random_c') or (generator_mode == 'random_z_image_c'):
            with misc.ddp_sync(self.G_mapping, sync):
                ws  = self.G_mapping(z, c)
                if self.style_mixing_prob > 0:
                    with torch.autograd.profiler.record_function('style_mixing'):
                        cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                        cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                        ws[:, cutoff:] = self.G_mapping(torch.randn_like(z), c, skip_w_avg_update=True)[:, cutoff:]
            if generator_mode == 'random_z_image_c':
                assert img is not None
                with torch.no_grad():
                    D = self.D_ema if self.D_ema is not None else self.D
                    with misc.ddp_sync(D, sync):
                        estimated_c = misc.get_func(D, 'get_estimated_camera')(img)[0].detach()
                        if estimated_c.size(-1) == 16:
                            synthesis_kwargs['camera_RT'] = estimated_c
                        if estimated_c.size(-1) == 3:
                            synthesis_kwargs['camera_UV'] = estimated_c
            with misc.ddp_sync(self.G_synthesis, sync):
                out = self.G_synthesis(ws, **synthesis_kwargs)
        else:
            raise NotImplementedError(f'wrong generator_mode {generator_mode}')
        return out, ws

    def run_D(self, img, c, sync):
        with misc.ddp_sync(self.D, sync):
            logits = self.D(img, c, aug_pipe=self.augment_pipe)
        return logits

    def get_loss(self, outputs, module='D'):
        reg_loss, logs, del_keys = 0, [], []
        if isinstance(outputs, dict):
            for key in outputs:
                if key[-5:] == '_loss':
                    logs += [(f'Loss/{module}/{key}', outputs[key])]
                    del_keys += [key]
                    if (self.other_weights is not None) and (key in self.other_weights):
                        reg_loss = reg_loss + outputs[key].mean() * self.other_weights[key]
                    else:
                        reg_loss = reg_loss + outputs[key].mean()
            for key in del_keys:
                del outputs[key]
            for key, loss in logs:
                training_stats.report(key, loss)
        return reg_loss

    @property
    def resolution(self):
        return misc.get_func(self.G_synthesis, 'get_current_resolution')()[-1]

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, fake_img, sync, gain, scaler=None):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Gpl   = (phase in ['Greg', 'Gboth']) 
        do_Dr1   = (phase in ['Dreg', 'Dboth'])
        losses   = {}

        # Gmain: Maximize logits for generated images.
        loss_Gmain, reg_loss = 0, 0
        if isinstance(fake_img, dict): fake_img = fake_img['img']
        if do_Gmain:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, gen_ws = self.run_G(gen_z, gen_c, sync=(sync and not do_Gpl), img=fake_img)   # May get synced by Gpl.
                reg_loss  += self.get_loss(gen_img, 'G')
                gen_logits = self.run_D(gen_img, gen_c, sync=False)
                reg_loss  += self.get_loss(gen_logits, 'G')
                if isinstance(gen_logits, dict):
                    gen_logits = gen_logits['logits']
                    
                loss_Gmain = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))
                if self.label_smooth > 0:
                    loss_Gmain = loss_Gmain * (1 - self.label_smooth) +  torch.nn.functional.softplus(gen_logits) * self.label_smooth
                
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                training_stats.report('Loss/G/loss', loss_Gmain)

            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain  = loss_Gmain + reg_loss
                losses['Gmain'] = loss_Gmain.mean().mul(gain)
                loss = scaler.scale(losses['Gmain']) if scaler is not None else losses['Gmain']
                loss.backward()

        # Gpl: Apply path length regularization.
        if do_Gpl and (self.pl_weight != 0):
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = max(1, gen_z.shape[0] // self.pl_batch_shrink)
                gen_img, gen_ws = self.run_G(
                    gen_z[:batch_size], gen_c[:batch_size], sync=sync, 
                    img=fake_img[:batch_size] if fake_img is not None else None)
                if isinstance(gen_img, dict):  gen_img = gen_img['img']
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
                # with torch.autograd.profiler.record_function('pl_grads'):
                    pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True, allow_unused=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)

            with torch.autograd.profiler.record_function('Gpl_backward'):
                losses['Gpl'] = (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain)
                loss = scaler.scale(losses['Gpl']) if scaler is not None else losses['Gpl']
                loss.backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen, reg_loss = 0, 0
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img    = self.run_G(gen_z, gen_c, sync=False, img=fake_img)[0]                
                reg_loss  += self.get_loss(gen_img, 'D')
                gen_logits = self.run_D(gen_img, gen_c, sync=False) # Gets synced by loss_Dreal.
                reg_loss  += self.get_loss(gen_logits, 'D')
                if isinstance(gen_logits, dict):
                    gen_logits = gen_logits['logits']
                   
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake',  gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))

            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen  = loss_Dgen + reg_loss
                losses['Dgen'] = loss_Dgen.mean().mul(gain)
                loss = scaler.scale(losses['Dgen']) if scaler is not None else losses['Dgen']
                loss.backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or (do_Dr1 and (self.r1_gamma != 0)):
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                if isinstance(real_img, dict):
                    real_img['img'] = real_img['img'].requires_grad_(do_Dr1)
                else:
                    real_img = real_img.requires_grad_(do_Dr1)
                real_logits = self.run_D(real_img, real_c, sync=sync)
                if isinstance(real_logits, dict):
                    real_logits = real_logits['logits']

                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real',  real_logits.sign())

                loss_Dreal = 0
                if do_Dmain:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    if self.label_smooth > 0:
                        loss_Dreal = loss_Dreal * (1 - self.label_smooth) +  torch.nn.functional.softplus(real_logits) * self.label_smooth
                    
                    training_stats.report('Loss/D/loss', loss_Dgen.mean() + loss_Dreal.mean())

                loss_Dr1 = 0
                if do_Dr1:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        real_img_tmp = real_img['img'] if isinstance(real_img, dict) else real_img
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                losses['Dr1'] = (real_logits * 0 + loss_Dreal + loss_Dr1).mean().mul(gain)
                loss = scaler.scale(losses['Dr1']) if scaler is not None else losses['Dr1']
                loss.backward()

        return losses

#----------------------------------------------------------------------------
