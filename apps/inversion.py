# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


"""
https://github.com/pacifinapacific/StyleGAN_LatentEditor/blob/master/encode_image.py

"""

import numpy as np 
import matplotlib.pyplot as plt 
import os
import glob
import imageio
import torch
import torch.nn as nn

from torchvision import models 
import torch.nn.functional as F

import torch.optim as optim
import click
import dnnlib
import legacy
import copy
import pickle
import PIL.Image

from collections import OrderedDict
from torchvision.utils import save_image
from training.networks import Generator, ResNetEncoder
from renderer import Renderer
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--encoder', 'encoder_pkl', help='pre-trained encoder for initialization', default=None)
@click.option('--encoder_z', 'ez',        help='use encoder to predict z', type=bool, default=False)
@click.option('--target', 'target_fname', help='Target image file to project to', required=True, metavar='FILE')
@click.option('--num-steps',              help='Number of optimization steps', type=int, default=1000, show_default=True)
@click.option('--seed',                   help='Random seed', type=int, default=303, show_default=True)
@click.option('--save-video',             help='Save an mp4 video of optimization progress', type=bool, default=True, show_default=True)
@click.option('--outdir',                 help='Where to save the output images', required=True, metavar='DIR')
@click.option('--l2_lambda', default=1, type=float)
@click.option('--pl_lambda', default=1, type=float)
def main(
    network_pkl: str,
    encoder_pkl: str,
    ez: bool,
    target_fname: str,
    outdir: str,
    save_video: bool,
    seed: int,
    num_steps: int,
    l2_lambda: float,
    pl_lambda: float,
):

    np.random.seed(seed)
    torch.manual_seed(seed)
    conv2d_gradfix.enabled = True  # Improves training speed.

    # Load networks.
    if os.path.isdir(network_pkl):
        network_pkl = sorted(glob.glob(network_pkl + '/*.pkl'))[-1]
    print('Loading networks from "%s"...' % network_pkl)
    with dnnlib.util.open_url(network_pkl) as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device) # type: ignore
    G = copy.deepcopy(G).eval().requires_grad_(False).to(device) # type: ignore
    
    E = None
    if encoder_pkl is not None:
        if os.path.isdir(encoder_pkl):
            encoder_pkl = sorted(glob.glob(encoder_pkl + '/*.pkl'))[-1]
        print('Loading pretrained encoder from "%s"...' % encoder_pkl)
        with dnnlib.util.open_url(encoder_pkl) as fp:
            E = legacy.load_network_pkl(fp)['E'].requires_grad_(False).to(device) # type: ignore

    try:
        with torch.no_grad():
            G2 = Generator(*G.init_args, **G.init_kwargs).to(device)
            misc.copy_params_and_buffers(G, G2, require_all=False)
    except RuntimeError:
        G2 = G
    
    G2 = Renderer(G2, None, program=None)

    # Load target image.
    if 'gen' != target_fname[:3]:
        target_pil = PIL.Image.open(target_fname).convert('RGB')
        w, h = target_pil.size
        s = min(w, h)
        target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        target_pil = target_pil.resize((G.img_resolution, G.img_resolution), PIL.Image.LANCZOS)
        target_uint8 = np.array(target_pil, dtype=np.uint8)
        target_image = torch.tensor(target_uint8.transpose([2, 0, 1]), device=device)
    else:
        z = np.random.RandomState(int(target_fname[3:])).randn(1, G.z_dim)
        t = np.random.rand() if E is not None else 0
        camera_matrices = G2.get_camera_traj(t, 1, device=device)
        target_image = G2(torch.from_numpy(z).to(device), None, camera_matrices=camera_matrices)[0]
        target_image = ((target_image * 0.5 + 0.5) * 255).clamp(0,255).to(torch.uint8)

    if E is None:  # starting from initial
        z_samples = np.random.RandomState(123).randn(10000, G.z_dim)
        w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
        w_samples = w_samples.mean(0, keepdim=True)
        ws = w_samples.clone()
        ws.requires_grad = True
        cm = None
    else:
        if not ez:
            ws, cm = E(target_image[None,:].to(torch.float32) / 127.5 - 1)
        else:
            # from fairseq import pdb;pdb.set_trace()
            zs, cm = E(target_image[None,:].to(torch.float32) / 127.5 - 1)
            ws = G.mapping(zs, None)

        ws = ws.clone()
        ws.requires_grad = True

    MSE_Loss        = nn.MSELoss(reduction="mean")
    # MSE_Loss        = nn.SmoothL1Loss(reduction='mean')
    perceptual_net  = VGG16_for_Perceptual(n_layers=[2,4,14,21]).to(device)
    target_image    = target_image.clone().unsqueeze(0).to(torch.float32) / 255.
    target_image_p  = F.interpolate(target_image, size=(256, 256), mode='area')
    target_features = perceptual_net(target_image_p)

    opt_weights = [{'params': ws}]
    kwargs = G2.get_additional_params(ws)
    if cm is not None:
        kwargs['camera_matrices'] = G.synthesis.get_camera(1, device, mode=cm)

    if len(kwargs) > 0:
        # latent codes for the background
        if len(kwargs['latent_codes'][2].size()) > 0:
            kwargs['latent_codes'][2].requires_grad = True
            opt_weights += [{'params': kwargs['latent_codes'][2]}]
        if len(kwargs['latent_codes'][3].size()) > 0:
            kwargs['latent_codes'][3].requires_grad = True
            opt_weights += [{'params': kwargs['latent_codes'][3]}]

    optimizer = optim.Adam(opt_weights, lr=0.01, betas=(0.9,0.999), eps=1e-8)
    
    print("Start...")
    loss_list = []
    
    os.makedirs(outdir, exist_ok=True)
    if save_video:
        import time
        timestamp = time.strftime('%Y%m%d.%H%M%S',time.localtime(time.time()))
        video = imageio.get_writer(f'{outdir}/proj_{timestamp}.mp4', mode='I', fps=24, codec='libx264', bitrate='16M')
        print (f'Saving optimization progress video "{outdir}/proj.mp4"')

    for i in range(num_steps):
        optimizer.zero_grad()
        # kwargs['camera_matrices'] = G.synthesis.get_camera(1, device, cs)
        synth_image = G2(styles=ws, **kwargs)
        synth_image = (synth_image + 1.0) / 2.0
        
        mse_loss, perceptual_loss = caluclate_loss(
            synth_image, target_image, target_features, perceptual_net, MSE_Loss)
        mse_loss = mse_loss * l2_lambda
        perceptual_loss = perceptual_loss * pl_lambda
        loss= mse_loss + perceptual_loss
        loss.backward()
        optimizer.step()

        loss_np = loss.detach().cpu().numpy()
        loss_p  = perceptual_loss.detach().cpu().numpy()
        loss_m  = mse_loss.detach().cpu().numpy()
        loss_list.append(loss_np)

        if i % 10 == 0:
            print("iter {}: loss -- {:.5f} \t mse_loss --{:.5f} \t percep_loss --{:.5f}".format(i,loss_np,loss_m,loss_p))
            if save_video:
                image = torch.cat([target_image, synth_image], -1)
                image = image.permute(0, 2, 3, 1) * 255.
                image = image.clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                video.append_data(image)
            
        if i % 100 == 0:
            save_image(torch.cat([target_image, synth_image], -1).clamp(0,1), f"{outdir}/{i}.png")
            np.save("loss_list.npy",loss_list)
            np.save(f"{outdir}/latent_W_{i}.npy", ws.detach().cpu().numpy())
    
    np.save(f"{outdir}/latent_last.npy", ws.detach().cpu().numpy())
    # render the learned model
    if len(kwargs) > 0:  # stylenerf
        assert save_video
        G2.program = 'rotation_camera3'
        all_images = G2(styles=ws)
        def proc_img(img): 
            return (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu()

        target_image = proc_img(target_image * 2 - 1).numpy()[0]
        all_images = torch.stack([proc_img(i) for i in all_images], dim=-1).numpy()[0]
        for i in range(all_images.shape[-1]):
            video.append_data(np.concatenate([target_image, all_images[..., i]], 1))
        
        outdir = f'{outdir}/proj_{timestamp}'
        os.makedirs(outdir, exist_ok=True)
        for step in range(all_images.shape[-1]):
            img = all_images[..., i]
            PIL.Image.fromarray(img, 'RGB').save(f'{outdir}/{step:04d}.png')



    if save_video:
        video.close()
    
   
        
def caluclate_loss(synth_image, target_image, target_features, perceptual_net, MSE_Loss):
     #calculate MSE Loss
     mse_loss = MSE_Loss(synth_image, target_image) # (lamda_mse/N)*||G(w)-I||^2

     #calculate Perceptual Loss
     real_0, real_1, real_2, real_3 = target_features
     synth_image_p = F.interpolate(synth_image, size=(256, 256), mode='area')
     synth_0, synth_1, synth_2, synth_3 = perceptual_net(synth_image_p)
     perceptual_loss = 0
     perceptual_loss += MSE_Loss(synth_0, real_0)
     perceptual_loss += MSE_Loss(synth_1, real_1)
     perceptual_loss += MSE_Loss(synth_2, real_2)
     perceptual_loss += MSE_Loss(synth_3, real_3)
     return mse_loss, perceptual_loss

class VGG16_for_Perceptual(torch.nn.Module):
    def __init__(self,requires_grad=False,n_layers=[2,4,14,21]):
        super(VGG16_for_Perceptual,self).__init__()
        vgg_pretrained_features=models.vgg16(pretrained=True).features 

        self.slice0=torch.nn.Sequential()
        self.slice1=torch.nn.Sequential()
        self.slice2=torch.nn.Sequential()
        self.slice3=torch.nn.Sequential()

        for x in range(n_layers[0]):#relu1_1
            self.slice0.add_module(str(x),vgg_pretrained_features[x])
        for x in range(n_layers[0],n_layers[1]): #relu1_2
            self.slice1.add_module(str(x),vgg_pretrained_features[x])
        for x in range(n_layers[1],n_layers[2]): #relu3_2
            self.slice2.add_module(str(x),vgg_pretrained_features[x])

        for x in range(n_layers[2],n_layers[3]):#relu4_2
            self.slice3.add_module(str(x),vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad=False
        
    def forward(self,x):
        h0=self.slice0(x)
        h1=self.slice1(h0)
        h2=self.slice2(h1)
        h3=self.slice3(h2)

        return h0,h1,h2,h3


if __name__ == "__main__":
    main()


