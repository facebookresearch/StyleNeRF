# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os, sys
# os.system('pip install -r requirements.txt')

import gradio as gr
from matplotlib.pyplot import hist
import numpy as np
import dnnlib
import time
import legacy
import torch
import glob
import cv2
import imageio
import tempfile

from torch_utils import misc
from renderer import Renderer
from training.networks import Generator
from huggingface_hub import hf_hub_download


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
port   = int(sys.argv[1]) if len(sys.argv) > 1 else 21111

model_lists = {
    'ffhq-512x512-basic':   dict(repo_id='facebook/stylenerf-ffhq-config-basic', filename='ffhq_512.pkl'),
    'ffhq-256x256-basic':   dict(repo_id='facebook/stylenerf-ffhq-config-basic', filename='ffhq_256.pkl'), 
    'ffhq-1024x1024-basic': dict(repo_id='facebook/stylenerf-ffhq-config-basic', filename='ffhq_1024.pkl'), 
}
model_names = [name for name in model_lists]


def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def stack_imgs(imgs):
    img = torch.stack(imgs, dim=2)
    return img.reshape(img.size(0) * img.size(1), img.size(2) * img.size(3), 3)


def proc_img(img): 
    return (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu()


def get_camera_traj(model, pitch, yaw, fov=12, batch_size=1, model_name=None):
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


def check_name(model_name):
    """Gets model by name."""
    if model_name in model_lists:
        network_pkl = hf_hub_download(**model_lists[model_name])
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

    print('compile and generate the initial video')
    G2 = G2.eval()
    init_z = torch.from_numpy(np.random.RandomState(0).rand(1, G2.z_dim)).to(device)
    init_cam = get_camera_traj(G2, 0, 0, model_name=network_pkl)
    dummy = G2(z=init_z, c=None, camera_matrices=init_cam, render_option=render_option, theta=0)
    res = dummy['img'].shape[-1]
    imgs = np.zeros((res, res//2, 3))

    # render the video
    R = Renderer(G2, program='rotation_camera3')
    seeds, all_imgs = np.random.randint(0, 10000, 3), []
    with torch.no_grad():
        for idx, seed in enumerate(seeds):
            R.set_random_seed(seed)
            z = torch.from_numpy(np.random.RandomState(seed).randn(1, G2.z_dim)).to(device)
            vid_imgs = R(z=z, truncation_psi=0.7, render_option=render_option, n_steps=36, batch_size=1)
            vid_imgs = [proc_img(i) for i in vid_imgs]
            all_imgs += [vid_imgs]

    # write to video
    all_imgs = [stack_imgs([a[k] for a in all_imgs]).numpy() for k in range(len(all_imgs[0]))]
    temp = "gradio_cached_examples/temp_video.mp4"
    imageio.mimwrite(temp, all_imgs, fps=30, quality=8)            
    return G2, res, imgs, temp


global_states = list(get_model(check_name(model_names[0])))
wss  = [None, None]

def proc_seed(history, seed):
    if isinstance(seed, str):
        seed = 0
    else:
        seed = int(seed)


def f_synthesis(model_name, model_find, render_option, early, trunc, seed1, seed2, mix1, mix2, yaw, pitch, roll, fov, history):
    history = history or {}
    seeds = []
    trunc = trunc / 100
    mix1 = mix1 / 100
    mix2 = mix2 / 100
    
    if 'log' not in history:
        history['log'] = []

    if model_find != "":
        model_name = model_find

    model_name = check_name(model_name)
    if model_name != history.get("model_name", None):
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        history['log'].append((f'load model ({timestamp})', model_name))
        _states = list(get_model(model_name, render_option))
        for j in range(len(_states)):
            global_states[j] = _states[j]
    model, res, imgs, video_name = global_states
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
            img = model.get_final_output(styles=ws, camera_matrices=get_camera_traj(model, 0, 0, model_name=model_name), render_option=render_option)
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

    set_random_seed(sum(seeds))

    # style mixing (?)
    ws1, ws2 = [torch.from_numpy(ws).to(device) for ws in wss]
    ws = ws1.clone()
    ws[:, :8] = ws1[:, :8] * mix1 + ws2[:, :8] * (1 - mix1)
    ws[:, 8:] = ws1[:, 8:] * mix2 + ws2[:, 8:] * (1 - mix2)

    # set visualization for other types of inputs.
    if early == 'Normal Map':
        render_option += ',normal,early'
    elif early == 'Gradient Map':
        render_option += ',gradient,early'

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
    image = (image * 255).astype('uint8')

    return image, video_name, history['log'], history

model_name = gr.inputs.Dropdown(model_names)
model_find = gr.inputs.Textbox(label="Checkpoint path (folder or .pkl file)", default="")
render_option = gr.inputs.Textbox(label="Additional rendering options", default='freeze_bg,steps:50')
trunc  = gr.inputs.Slider(default=70, maximum=100, minimum=0, label='Truncation trick (%)')
seed1  = gr.inputs.Number(default=1, label="Random seed1")
seed2  = gr.inputs.Number(default=9, label="Random seed2")
mix1   = gr.inputs.Slider(minimum=0, maximum=100, default=50, label="Linear mixing ratio (geometry) %")
mix2   = gr.inputs.Slider(minimum=0, maximum=100, default=50, label="Linear mixing ratio (apparence) %")
early  = gr.inputs.Radio(['None', 'Normal Map', 'Gradient Map'], default='None', label='Intermedia output')
yaw    = gr.inputs.Slider(minimum=-1, maximum=1, default=0, label="Yaw")
pitch  = gr.inputs.Slider(minimum=-1, maximum=1, default=0, label="Pitch")
roll   = gr.inputs.Slider(minimum=-1, maximum=1, default=0, label="Roll (optional, not suggested for basic config)")
fov    = gr.inputs.Slider(minimum=10, maximum=14, default=12, label="Fov")
css = ".output-image, .input-image, .image-preview {height: 600px !important} "

gr.Interface(fn=f_synthesis,
             inputs=[model_name, model_find, render_option, early, trunc, seed1, seed2, mix1, mix2, yaw, pitch, roll, fov, "state"],
             title="Interactive Web Demo for StyleNeRF (ICLR 2022)",
             description="StyleNeRF: A Style-based 3D-Aware Generator for High-resolution Image Synthesis. Currently the demo runs on CPU only.",
             outputs=["image", "video", "chatbot", "state"],
             layout='unaligned',
             css=css, theme='huggingface',
             live=True).launch(enable_queue=True)
