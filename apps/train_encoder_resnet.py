# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from random import random
from dnnlib import camera
import os
import numpy as np
import torch
import copy
import torch.distributed as dist
import torchvision
import click
import dnnlib
import legacy
import pickle

from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
from torch_utils.ops import conv2d_gradfix
from torch_utils import misc
from torchvision import transforms, utils
from tqdm import tqdm

from training.networks import EqualConv2d, ResNetEncoder

try:
    from tensorboardX import SummaryWriter
except ImportError:
    SummaryWriter = None


def data_sampler(dataset, shuffle):
    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)
    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()
    return loss


class VGGLoss(nn.Module):
    def __init__(self, device, n_layers=5):
        super().__init__()

        feature_layers = (2, 7, 12, 21, 30)
        self.weights = (1.0, 1.0, 1.0, 1.0, 1.0)

        vgg = torchvision.models.vgg19(pretrained=True).features

        self.layers = nn.ModuleList()
        prev_layer = 0
        for next_layer in feature_layers[:n_layers]:
            layers = nn.Sequential()
            for layer in range(prev_layer, next_layer):
                layers.add_module(str(layer), vgg[layer])
            self.layers.append(layers.to(device))
            prev_layer = next_layer

        for param in self.parameters():
            param.requires_grad = False

        self.criterion = nn.L1Loss().to(device)

    def forward(self, source, target):
        loss = 0
        source, target = (source + 1) / 2, (target + 1) / 2
        for layer, weight in zip(self.layers, self.weights):
            source = layer(source)
            with torch.no_grad():
                target = layer(target)
            loss += weight * self.criterion(source, target)

        return loss


@click.command()
@click.option("--data", type=str, default=None)
@click.option("--g_ckpt", type=str, default=None)
@click.option("--e_ckpt", type=str, default=None)
@click.option("--max_steps", type=int, default=1000000)
@click.option("--batch", type=int, default=8)
@click.option("--lr", type=float, default=0.0001)
@click.option("--local_rank", type=int, default=0)
@click.option("--vgg", type=float, default=1.0)
@click.option("--l2", type=float, default=1.0)
@click.option("--adv", type=float, default=0.05)
@click.option("--tensorboard", type=bool, default=True)
@click.option("--outdir", type=str, required=True)

def main(data, outdir, g_ckpt, e_ckpt,
         max_steps, batch, lr, local_rank, vgg,
         l2, adv, tensorboard):
    random_seed = 22
    np.random.seed(random_seed)
    use_image_loss = False
    conv2d_gradfix.enabled = True  # Improves training speed.
    device = torch.device('cuda', local_rank)

    # load the pre-trained model
    if os.path.isdir(g_ckpt):
        import glob
        g_ckpt = sorted(glob.glob(g_ckpt + '/*.pkl'))[-1]
    print('Loading networks from "%s"...' % g_ckpt)
    with dnnlib.util.open_url(g_ckpt) as fp:
        network = legacy.load_network_pkl(fp)
        G = network['G_ema'].requires_grad_(False).to(device)
        D = network['D'].requires_grad_(False).to(device)
    G = copy.deepcopy(G).eval().requires_grad_(False).to(device)
    D = copy.deepcopy(D).eval().requires_grad_(False).to(device)
    E = ResNetEncoder(G.img_resolution, G.mapping.num_ws, G.mapping.w_dim, add_dim=2).to(device)
    E_optim = optim.Adam(E.parameters(), lr=lr, betas=(0.9, 0.99))
    requires_grad(E, True)

    start_iter = 0
    pbar = range(max_steps)
    pbar = tqdm(pbar, initial=start_iter, dynamic_ncols=True, smoothing=0.01)

    loss_dict = {}
    # vgg_loss   = VGGLoss(device=device)
    truncation = 0.7
    ws_avg = G.mapping.w_avg[None, None, :]

    if SummaryWriter and tensorboard:
        logger = SummaryWriter(logdir=f'{outdir}/tensorboard')

    for idx in pbar:
        i = idx + start_iter
        if i > max_steps:
            print("Done!")
            break

        E_optim.zero_grad()  # zero-out gradients

        z_samples = np.random.randn(batch, ws_avg.size(-1))
        z_samples = torch.from_numpy(z_samples).to(torch.float32).to(device)
        w_samples = G.mapping(z_samples, None)
        c_samples = torch.from_numpy(np.random.randn(batch, 2)).to(torch.float32).to(device)
        if truncation < 1.0:
            w_samples = ws_avg + (w_samples - ws_avg) * truncation
        camera_matrices = G.synthesis.get_camera(batch, device, mode=c_samples)
        gen_img = G.get_final_output(styles=w_samples, camera_matrices=camera_matrices)
        rec_zs, rec_cm = E(gen_img)
        rec_ws = G.mapping(rec_zs, None)

        loss_dict['loss_ws'] = F.smooth_l1_loss(rec_ws, w_samples).mean()
        loss_dict['loss_cm'] = F.smooth_l1_loss(rec_cm, c_samples).mean()

        if use_image_loss and (i > 500):
            rec_camera_matrices = G.synthesis.get_camera(batch, device, mode=rec_cm)
            rec_img = G.get_final_output(styles=rec_ws, camera_matrices=rec_camera_matrices)
            recon_l2_loss = F.mse_loss(rec_img, gen_img)
            loss_dict["l2"] = recon_l2_loss * l2
        else:
            loss_dict["l2"] = torch.tensor(0.0).to(device)
        
        # real_img, _ = next(training_set_iterator)
        # real_img = real_img.to(device).to(torch.float32) / 127.5 - 1
        # pws, pcm = E(real_img)

        # pws = pws + ws_avg   # make sure it starts from the average (?)
        # if truncation < 1.0:
        #     pws = ws_avg + (pws - ws_avg) * truncation
        # camera_matrices = G.synthesis.get_camera(batch, device, mode=pcm)

        # recon_img  = G.get_final_output(styles=pws, camera_matrices=camera_matrices)
        # recon_pred = D(recon_img, None)

        # recon_vgg_loss = vgg_loss(recon_img, real_img)
        # loss_dict["vgg"] = recon_vgg_loss * vgg

        # recon_l2_loss = F.mse_loss(recon_img, real_img)
        # loss_dict["l2"] = recon_l2_loss * l2

        # adv_loss = g_nonsaturating_loss(recon_pred) * adv
        # loss_dict["adv"] = adv_loss

        # E_loss = recon_vgg_loss + recon_l2_loss + adv_loss
        # loss_dict["e_loss"] = E_loss

        E_loss = sum([loss_dict[l] for l in loss_dict])
        E_loss.backward()
        E_optim.step()

        desp = '\t'.join([f'{name}: {loss_dict[name].item():.4f}' for name in loss_dict])
        pbar.set_description((desp))

        if SummaryWriter and tensorboard:
            logger.add_scalar('E_loss/total', E_loss, i)
            logger.add_scalar('E_loss/loss_ws', loss_dict['loss_ws'], i)
            logger.add_scalar('E_loss/loss_cm', loss_dict['loss_cm'], i)
            logger.add_scalar('E_loss/l2', loss_dict["l2"], i)

        if i > 0 and i % 1000 == 0:
            os.makedirs(f'{outdir}/sample', exist_ok=True)
            with torch.no_grad():
                if not use_image_loss:
                    rec_camera_matrices = G.synthesis.get_camera(batch, device, mode=rec_cm)
                    rec_img = G.get_final_output(styles=rec_ws, camera_matrices=rec_camera_matrices)

                sample = torch.cat([gen_img.detach(), rec_img.detach()])
                utils.save_image(
                    sample,
                    f"{outdir}/sample/{str(i).zfill(6)}.png",
                    nrow=int(batch),
                    normalize=True,
                    range=(-1, 1),
                )

        if i % 10000 == 0:
            os.makedirs(f'{outdir}/checkpoints', exist_ok=True)
            snapshot_pkl = os.path.join(f'{outdir}/checkpoints/', f'network-snapshot-{i // 1000:06d}.pkl')
            # snapshot_data = dict(training_set_kwargs=dict(training_set_kwargs))
            snapshot_data = dict()
            snapshot_data['E'] = E
            with open(snapshot_pkl, 'wb') as f:
                pickle.dump(snapshot_data, f)


if __name__ == "__main__":
    main()