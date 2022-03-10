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
import clip
import math

from torchvision import models 
import torch.nn.functional as F

import torch.optim as optim
import click
import dnnlib
import legacy
import copy
import PIL.Image

from collections import OrderedDict
from tqdm import tqdm
from torchvision.utils import save_image
from training.networks import Generator
from renderer import Renderer
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

@click.command()
@click.option('--description', 'text',    help='the text that guides the generation', default="a person with purple hair")
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--num-steps',              help='Number of optimization steps', type=int, default=1000, show_default=True)
@click.option('--seed',                   help='Random seed', type=int, default=303, show_default=True)
@click.option('--save-video',             help='Save an mp4 video of optimization progress', type=bool, default=True, show_default=True)
@click.option('--outdir',                 help='Where to save the output images', required=True, metavar='DIR')
@click.option('--lr_rampup', type=float, default=0.05)
@click.option('--lr_init',   type=float, default=0.1)
@click.option('--l2_lambda', type=float, default=0.008)
@click.option('--id_lambda', type=float, default=0.000)
@click.option('--trunc', type=float, default=0.7)
@click.option('--mode', type=click.Choice(['free', 'edit']), default='edit')
def main(
    text: str,
    network_pkl: str,
    outdir: str,
    save_video: bool,
    seed: int,
    num_steps: int,
    lr_init: float,
    lr_rampup: float,
    l2_lambda: float,
    id_lambda: float,
    trunc: 0.7,
    mode: str,
):

    np.random.seed(seed)
    torch.manual_seed(seed)
    conv2d_gradfix.enabled = True  # Improves training speed.
    os.makedirs(outdir, exist_ok=True)
    if save_video:
        video = imageio.get_writer(f'{outdir}/edit.mp4', mode='I', fps=24, codec='libx264', bitrate='16M')
        print (f'Saving optimization progress video "{outdir}/edit.mp4"')

    # Load networks.
    if os.path.isdir(network_pkl):
        network_pkl = sorted(glob.glob(network_pkl + '/*.pkl'))[-1]
    print('Loading networks from "%s"...' % network_pkl)
    with dnnlib.util.open_url(network_pkl) as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device) # type: ignore
    G = copy.deepcopy(G).eval().requires_grad_(False).to(device) # type: ignore

    with torch.no_grad():
        G2 = Generator(*G.init_args, **G.init_kwargs).to(device)
        misc.copy_params_and_buffers(G, G2, require_all=False)
    G2 = Renderer(G2, None, program=None)

    # start from an average 
    z = np.random.RandomState(seed).randn(1, G.z_dim)
    camera_matrices = G2.get_camera_traj(0, 1, device=device)
    ws_init = G.mapping(torch.from_numpy(z).to(device), None, truncation_psi=trunc)    
    initial_image = G2(styles=ws_init, camera_matrices=camera_matrices)

    ws = ws_init.clone()
    ws.requires_grad = True
    clip_loss   = CLIPLoss(stylegan_size=G.img_resolution)
    if id_lambda > 0:
        id_loss = IDLoss()
    optimizer   = optim.Adam([ws], lr=lr_init, betas=(0.9,0.999), eps=1e-8)
    pbar        = tqdm(range(num_steps))
    text_input  = torch.cat([clip.tokenize(text)]).to(device)

    for i in pbar:
        # t = i / float(num_steps)
        # lr = get_lr(t, lr_init, rampup=lr_rampup)
        # optimizer.param_groups[0]["lr"] = lr
        optimizer.zero_grad()

        img_gen = G2(styles=ws, camera_matrices=camera_matrices)
        c_loss = clip_loss(img_gen, text_input)
    
        if id_lambda > 0:
            i_loss = id_loss(img_gen, initial_image)[0]
        else:
            i_loss = 0

        if mode == "edit":
            l2_loss = ((ws - ws_init) ** 2).sum()
            loss = c_loss + l2_lambda * l2_loss + id_lambda * i_loss
        else:
            l2_loss = 0
            loss = c_loss

        loss.backward()
        optimizer.step()
        pbar.set_description((f"loss: {loss.item():.4f}; c:{c_loss.item():.4f}; l2:{l2_loss:.4f}; id:{i_loss:.4f}"))
        if i % 10 == 0:
            if save_video:
                image = torch.cat([initial_image, img_gen], -1) * 0.5 + 0.5
                image = image.permute(0, 2, 3, 1) * 255.
                image = image.clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                video.append_data(image)
            
        if i % 100 == 0:
            save_image(torch.cat([initial_image, img_gen], -1).clamp(-1,1), f"{outdir}/{i}.png", normalize=True, range=(-1, 1))
            # np.save("latent_W/{}.npy".format(name),dlatent.detach().cpu().numpy())
        
    # # render the learned model
    # if len(kwargs) > 0:  # stylenerf
    #     assert save_video
    #     G2.program = 'rotation_camera3'
    #     all_images = G2(styles=ws)
    #     def proc_img(img): 
    #         return (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu()

    #     initial_image = proc_img(initial_image * 2 - 1).numpy()[0]
    #     all_images = torch.stack([proc_img(i) for i in all_images], dim=-1).numpy()[0]
    #     for i in range(all_images.shape[-1]):
    #         video.append_data(np.concatenate([initial_image, all_images[..., i]], 1))
        
    if save_video:
        video.close()
        

def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)
    return initial_lr * lr_ramp


class CLIPLoss(torch.nn.Module):
    def __init__(self, stylegan_size):
        super(CLIPLoss, self).__init__()
        self.model, self.preprocess = clip.load("ViT-B/32", device="cuda")
        self.upsample = torch.nn.Upsample(scale_factor=7)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=stylegan_size // 32)

    def forward(self, image, text):
        
        def preprocess_tensor(x):
            import torchvision.transforms.functional as F
            x = F.resize(x, size=224, interpolation=PIL.Image.BICUBIC)
            x = F.normalize(x, (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            return x

        image = preprocess_tensor(image)
        # image = self.avg_pool(self.upsample(image))
        # image = self.preprocess(image)
        similarity = 1 - self.model(image, text)[0] / 100
        return similarity


class IDLoss(nn.Module):
    def __init__(self):
        super(IDLoss, self).__init__()
        from training.facial_recognition.model_irse import Backbone

        print('Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load('/private/home/jgu/.torch/models/model_ir_se50.pth'))
        self.pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()
        self.facenet.cuda()

    def extract_feats(self, x):
        if x.shape[2] != 256:
            x = self.pool(x)
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, y_hat, y):
        n_samples = y.shape[0]
        y_feats = self.extract_feats(y)  # Otherwise use the feature from there
        y_hat_feats = self.extract_feats(y_hat)
        y_feats = y_feats.detach()
        loss = 0
        sim_improvement = 0
        count = 0
        for i in range(n_samples):
            diff_target = y_hat_feats[i].dot(y_feats[i])
            loss += 1 - diff_target
            count += 1

        return loss / count, sim_improvement / count

if __name__ == "__main__":
    main()


