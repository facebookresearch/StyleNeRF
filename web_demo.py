# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import gradio as gr
import numpy as np
import dnnlib
import time
import legacy
import torch
import glob
import os, sys
import cv2
from torch_utils import misc
from renderer import Renderer
from training.networks import Generator

device = torch.device('cuda')
port   = int(sys.argv[1]) if len(sys.argv) > 1 else 21111

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
        network_pkl = "./pretrained/ffhq_512_eg3d.pkl"
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


def get_model(network_pkl, render_option=None):
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
    imgs = np.zeros((res, res//2, 3))
    return G2, res, imgs


global_states = list(get_model(check_name()))
wss  = [None, None]

def proc_seed(history, seed):
    if isinstance(seed, str):
        seed = 0
    else:
        seed = int(seed)


def f_synthesis(model_name, model_find, render_option, trunc, seed1, seed2, mix1, mix2, yaw, pitch, roll, fov):
    history = gr.get_state() or {}
    seeds = []
    
    if model_find != "":
        model_name = model_find

    model_name = check_name(model_name)
    if model_name != history.get("model_name", None):
        model, res, imgs = get_model(model_name, render_option)
        global_states[0] = model
        global_states[1] = res
        global_states[2] = imgs

    model, res, imgs = global_states
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
            z   = torch.from_numpy(np.random.RandomState(int(seed)).randn(1, model.z_dim).astype('float32')).to(device)
            ws  = model.mapping(z=z, c=None, truncation_psi=trunc)
            img = model.get_final_output(styles=ws, camera_matrices=get_camera_traj(model, 0, 0), render_option=render_option)
            ws  = ws.detach().cpu().numpy()
            img = img[0].permute(1,2,0).detach().cpu().numpy()

            
            imgs[idx * res // 2: (1 + idx) * res // 2] = cv2.resize(
                np.asarray(img).clip(-1, 1) * 0.5 + 0.5,
                (res//2, res//2), cv2.INTER_AREA)
            wss[idx] = ws
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

    start_t = time.time()
    with torch.no_grad():
        cam = get_camera_traj(model, pitch, yaw, fov, model_name=model_name)
        image = model.get_final_output(
            styles=ws, camera_matrices=cam, 
            theta=roll * np.pi,
            render_option=render_option)
    end_t = time.time()

    image = image[0].permute(1,2,0).detach().cpu().numpy().clip(-1, 1) * 0.5 + 0.5

    if imgs.shape[0] == image.shape[0]:
        image = np.concatenate([imgs, image], 1)
    else:
        a = image.shape[0]
        b = int(imgs.shape[1] / imgs.shape[0] * a)
        print(f'resize {a} {b} {image.shape} {imgs.shape}')
        image = np.concatenate([cv2.resize(imgs, (b, a), cv2.INTER_AREA), image], 1)
  
    print(f'rendering time = {end_t-start_t:.4f}s')
    return (image * 255).astype('uint8')

model_name = gr.inputs.Dropdown(['FFHQ512', 'FFHQ512v2', 'AFHQ512', 'MetFaces512', 'CompCars256', 'FFHQ1024'])
model_find = gr.inputs.Textbox(label="checkpoint path", default="")
render_option = gr.inputs.Textbox(label="rendering options", default='freeze_bg,steps:40')
trunc  = gr.inputs.Slider(default=0.7, maximum=1.0, minimum=0.0, label='truncation trick')
seed1  = gr.inputs.Number(default=1, label="seed1")
seed2  = gr.inputs.Number(default=9, label="seed2")
mix1   = gr.inputs.Slider(minimum=0, maximum=1, default=0, label="linear mixing ratio (geometry)")
mix2   = gr.inputs.Slider(minimum=0, maximum=1, default=0, label="linear mixing ratio (apparence)")
yaw    = gr.inputs.Slider(minimum=-1, maximum=1, default=0, label="yaw")
pitch  = gr.inputs.Slider(minimum=-1, maximum=1, default=0, label="pitch")
roll   = gr.inputs.Slider(minimum=-1, maximum=1, default=0, label="roll (optional, not suggested)")
fov    = gr.inputs.Slider(minimum=9, maximum=15, default=12, label="fov")
css = ".output_image {height: 40rem !important; width: 100% !important;}"
gr.Interface(fn=f_synthesis,
             inputs=[model_name, model_find, render_option, trunc, seed1, seed2, mix1, mix2, yaw, pitch, roll, fov],
             outputs="image",
             layout='unaligned',
             css=css,
             live=True,
             server_port=port).launch()
