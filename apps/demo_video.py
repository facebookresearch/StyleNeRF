# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import gradio as gr
import numpy as np
import dnnlib
import time
import legacy
import torch
import glob
import os
import cv2
import tempfile
import imageio

from torch_utils import misc
from renderer import Renderer
from training.networks import Generator

device = torch.device('cuda')
render_option = 'freeze_bg,steps36'


def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_camera_traj(model, pitch, yaw, fov=12, batch_size=1, model_name='FFHQ512'):
    gen = model.synthesis
    range_u, range_v = gen.C.range_u, gen.C.range_v
    if not (('car' in model_name) or ('Car' in model_name)):  # TODO: hack, better option?
        yaw, pitch = 0.5 * yaw, 0.3  * pitch
        pitch = pitch + np.pi/2
        u = (yaw - range_u[0]) / (range_u[1] - range_u[0])
        v = (pitch - range_v[0]) / (range_v[1] - range_v[0])
    else:
        u = (yaw + 1) / 2
        v = (pitch + 1) / 2
    cam = gen.get_camera(batch_size=batch_size, mode=[u, v, 0.5], device=device, fov=fov)
    return cam


def check_name(model_name='FFHQ512'):
    """Gets model by name."""
    if model_name == 'FFHQ512':
        network_pkl = "./pretrained/ffhq_512.pkl"
    elif model_name == 'FFHQ512v2':
        network_pkl = "./pretrained/ffhq_512.v2.pkl"
    elif model_name == 'AFHQ512':
        network_pkl = "./pretrained/afhq_512.pkl"
    elif model_name == 'MetFaces512':
        network_pkl = "./pretrained/metfaces_512.pkl"
    elif model_name == 'CompCars256':
        network_pkl = "./pretrained/cars_256.pkl"
    elif model_name == 'FFHQ1024':
        network_pkl = "./pretrained/ffhq_1024.pkl"
    else:
        if os.path.isdir(model_name):
            network_pkl = sorted(glob.glob(model_name + '/*.pkl'))[-1]
        else:
            network_pkl = model_name
    return network_pkl


def get_model(network_pkl):
    print('Loading networks from "%s"...' % network_pkl)
    with dnnlib.util.open_url(network_pkl) as f:
        network = legacy.load_network_pkl(f)
        G = network['G_ema'].to(device)  # type: ignore

    with torch.no_grad():
        G2 = Generator(*G.init_args, **G.init_kwargs).to(device)
        misc.copy_params_and_buffers(G, G2, require_all=False)

    print('compile and go through the initial image')
    G2 = G2.eval()
    
    init_z = torch.from_numpy(np.random.RandomState(0).rand(1, G2.z_dim)).to(device)
    init_cam = get_camera_traj(G2, 0, 0, model_name=network_pkl)
    dummy = G2(z=init_z, c=None, camera_matrices=init_cam, render_option=render_option, theta=0)
    res = dummy['img'].shape[-1]
    imgs = [None, None]
    return G2, res, imgs


global_states = list(get_model(check_name()))
wss  = [None, None]

def proc_seed(history, seed):
    if isinstance(seed, str):
        seed = 0
    else:
        seed = int(seed)

def stack_imgs(imgs):
    img = torch.stack(imgs, dim=2)
    return img.reshape(img.size(0) * img.size(1), img.size(2) * img.size(3), 3)

def proc_img(img): 
    return (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu()

def f_synthesis(model_name, program, model_find, trunc, seed1, seed2, mix1, mix2, roll, fov):
    history = gr.get_state() or {}
    seeds = []
    
    if model_find != "":
        model_name = model_find

    model_name = check_name(model_name)
    if model_name != history.get("model_name", None):
        model, res, imgs = get_model(model_name)
        global_states[0] = model
        global_states[1] = res
        global_states[2] = imgs

    model, res, imgs = global_states
    if program  == 'image':
        program = 'rotation_camera3'
    elif program == 'image+normal':
        program = 'rotation_both'
    renderer = Renderer(model, None, program=program)

    for idx, seed in enumerate([seed1, seed2]):
        if isinstance(seed, str):
            seed = 0
        else:
            seed = int(seed)
        
        if (seed != history.get(f'seed{idx}', -1)) or \
            (model_name != history.get("model_name", None)) or \
            (trunc != history.get("trunc", 0.7)) or \
            (wss[idx] is None):
            print(f'use seed {seed}')
            set_random_seed(seed)
            with torch.no_grad():
                z   = torch.from_numpy(np.random.RandomState(int(seed)).randn(1, model.z_dim).astype('float32')).to(device)
                ws  = model.mapping(z=z, c=None, truncation_psi=trunc)
                imgs[idx] = [proc_img(i) for i in renderer(styles=ws, render_option=render_option)]
                ws  = ws.detach().cpu().numpy()
            wss[idx]  = ws
        else:
            seed = history[f'seed{idx}']
        
        seeds += [seed]
        history[f'seed{idx}'] = seed

    history['trunc'] = trunc
    history['model_name'] = model_name
    gr.set_state(history)
    set_random_seed(sum(seeds))

    # style mixing (?)
    ws1, ws2 = [torch.from_numpy(ws).to(device) for ws in wss]
    ws = ws1.clone()
    ws[:, :8] = ws1[:, :8] * mix1 + ws2[:, :8] * (1 - mix1)
    ws[:, 8:] = ws1[:, 8:] * mix2 + ws2[:, 8:] * (1 - mix2)

    
    dirpath = tempfile.mkdtemp()
    start_t = time.time()
    with torch.no_grad():
        outputs  = [proc_img(i) for i in renderer(
            styles=ws.detach(), 
            theta=roll * np.pi,
            render_option=render_option)]
        all_imgs = [imgs[0], outputs, imgs[1]]
        all_imgs = [stack_imgs([a[k] for a in all_imgs]).numpy() for k in range(len(all_imgs[0]))]
        imageio.mimwrite(f'{dirpath}/output.mp4', all_imgs, fps=30, quality=8)
    end_t = time.time()
    print(f'rendering time = {end_t-start_t:.4f}s')
    return f'{dirpath}/output.mp4'

model_name = gr.inputs.Dropdown(['FFHQ512', 'FFHQ512v2', 'AFHQ512', 'MetFaces512', 'CompCars256', 'FFHQ1024'])
model_find = gr.inputs.Textbox(label="checkpoint path", default="")
program = gr.inputs.Dropdown(['image', 'image+normal'], default='image')
trunc  = gr.inputs.Slider(default=0.7, maximum=1.0, minimum=0.0, label='truncation trick')
seed1  = gr.inputs.Number(default=1, label="seed1")
seed2  = gr.inputs.Number(default=9, label="seed2")
mix1   = gr.inputs.Slider(minimum=0, maximum=1, default=0, label="linear mixing ratio (geometry)")
mix2   = gr.inputs.Slider(minimum=0, maximum=1, default=0, label="linear mixing ratio (apparence)")
roll   = gr.inputs.Slider(minimum=-1, maximum=1, default=0, label="roll (optional, not suggested)")
fov    = gr.inputs.Slider(minimum=9, maximum=15, default=12, label="fov")
css = ".output_video {height: 40rem !important; width: 100% !important;}"
gr.Interface(fn=f_synthesis,
             inputs=[model_name, program, model_find, trunc, seed1, seed2, mix1, mix2, roll, fov],
             outputs="video",
             layout='unaligned',
             server_port=20011,
             css=css).launch()
