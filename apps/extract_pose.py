# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


from genericpath import exists
import numpy as np 
import os
import glob
import torch
import json
import click
import dnnlib
import legacy
import copy
import PIL.Image
import glob
import tqdm
import time
import imageio
import cv2
import torch.nn.functional as F
from torchvision.utils import save_image
from training.networks import Generator
from renderer import Renderer
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix

device  = 'cuda:0' if torch.cuda.is_available() else 'cpu'
DECODERPATH = "/checkpoint/jgu/space/gan/ffhq/debug3/00486-nores_critical_bgstyle-ffhq_512-mirror-paper512-stylenerf_pgc-noaug"
ENCODERPATH = DECODERPATH + '/encoder3.1/checkpoints'

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', default=DECODERPATH)
@click.option('--encoder', 'encoder_pkl', help='pre-trained encoder for initialization', default=ENCODERPATH)
@click.option('--target',  'target_path', help='Target image file to project to', required=True, metavar='DIR')
@click.option('--seed',                   help='Random seed', type=int, default=303, show_default=True)
@click.option('--outdir',                 help='Where to save the output images', required=True, metavar='DIR')
@click.option('--no_image', default=False, type=bool)
def main(
    network_pkl: str,
    encoder_pkl: str,
    target_path: str,
    outdir: str,
    seed: int,
    no_image
):

    np.random.seed(seed)
    torch.manual_seed(seed)
    conv2d_gradfix.enabled = True  # Improves training speed.

    # Load networks.
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

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

    # Output files
    inferred_poses = {}
    target_files   = sorted(glob.glob(target_path + '/*.png'))
    timestamp      = time.strftime('%Y%m%d.%H%M%S',time.localtime(time.time()))

    if not no_image:
        video      = imageio.get_writer(f'{outdir}/proj_{timestamp}.mp4', mode='I', fps=4, codec='libx264', bitrate='16M')

    for step, target_fname in enumerate(tqdm.tqdm(target_files)):
        target_id0 = target_fname.split('/')[-1]
        target_pil = PIL.Image.open(target_fname).convert('RGB')
        w, h = target_pil.size
        s = min(w, h)
        target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        target_pil = target_pil.resize((G.img_resolution, G.img_resolution), PIL.Image.LANCZOS)
        target_uint8 = np.array(target_pil, dtype=np.uint8)
        target_image = torch.tensor(target_uint8.transpose([2, 0, 1]), device=device)

        ws, cm = E(target_image[None,:].to(torch.float32) / 127.5 - 1)
        target_image = target_image.clone().unsqueeze(0).to(torch.float32) / 255.
        opt_weights  = [{'params': ws}]
        kwargs = G2.get_additional_params(ws)
        kwargs['camera_matrices']  = G.synthesis.get_camera(1, device, mode=cm)
        inferred_poses[target_id0] = kwargs['camera_matrices'][1].cpu().numpy().reshape(-1).tolist()
        # if step > 200: break
        if not no_image:
            if len(kwargs) > 0:
                # latent codes for the background
                if len(kwargs['latent_codes'][2].size()) > 0:
                    kwargs['latent_codes'][2].requires_grad = True
                    opt_weights += [{'params': kwargs['latent_codes'][2]}]
                if len(kwargs['latent_codes'][3].size()) > 0:
                    kwargs['latent_codes'][3].requires_grad = True
                    opt_weights += [{'params': kwargs['latent_codes'][3]}]

            synth_image = G2(styles=ws, **kwargs)
            synth_image = (synth_image + 1.0) / 2.0
            image = torch.cat([target_image, synth_image], -1).clamp(0,1)[0]
            image = (image.permute(1,2,0).detach().cpu().numpy() * 255).astype('uint8')
            image = cv2.resize(image, (256, 128), interpolation=cv2.INTER_AREA)
            video.append_data(image)
            # save_image(torch.cat([target_image, synth_image], -1).clamp(0,1), f"{outdir}/{target_id0}.png")

    json.dump(inferred_poses, open(f'{outdir}/extracted_poses.json', 'w'))
    print('done')
    if not no_image: video.close()

if __name__ == "__main__":
    main()


