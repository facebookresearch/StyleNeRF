# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate style mixing image matrix using pretrained network pickle."""

import os
import glob
import re
from typing import List

import click
from numpy.lib.function_base import interp
import dnnlib
import numpy as np
import PIL.Image
import torch

import legacy

#----------------------------------------------------------------------------

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''
    print(s)
    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]


def lerp(t, v0, v1):
    '''
    Linear interpolation
    Args:
        t (float/np.ndarray): Value between 0.0 and 1.0
        v0 (np.ndarray): Starting vector
        v1 (np.ndarray): Final vector
    Returns:
        v2 (np.ndarray): Interpolation vector between v0 and v1
    '''
    v2 = (1.0 - t) * v0 + t * v1
    return v2


# Taken and adapted from wikipedia's slerp article
# https://en.wikipedia.org/wiki/Slerp
def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    '''
    Spherical linear interpolation
    Args:
        t (float/np.ndarray): Float value between 0.0 and 1.0
        v0 (np.ndarray): Starting vector
        v1 (np.ndarray): Final vector
        DOT_THRESHOLD (float): Threshold for considering the two vectors as
                               colineal. Not recommended to alter this.
    Returns:
        v2 (np.ndarray): Interpolation vector between v0 and v1
    '''
    # Copy the vectors to reuse them later
    v0_copy = np.copy(v0)
    v1_copy = np.copy(v1)
    # Normalize the vectors to get the directions and angles
    v0 = v0 / np.linalg.norm(v0)
    v1 = v1 / np.linalg.norm(v1)
    # Dot product with the normalized vectors (can't use np.dot in W)
    dot = np.sum(v0 * v1)
    # If absolute value of dot product is almost 1, vectors are ~colineal, so use lerp
    if np.abs(dot) > DOT_THRESHOLD:
        return lerp(t, v0_copy, v1_copy)
    # Calculate initial angle between v0 and v1
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    # Angle at timestep t
    theta_t = theta_0 * t
    sin_theta_t = np.sin(theta_t)
    # Finish the slerp algorithm
    s0 = np.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    v2 = s0 * v0_copy + s1 * v1_copy
    return v2


# Helper function for interpolation
def interpolate(v0, v1, n_steps, interp_type='spherical', smooth=False):
    '''
    Input:
        v0, v1 (np.ndarray): latent vectors in the spaces Z or W
        n_steps (int): number of steps to take between both latent vectors
        interp_type (str): Type of interpolation between latent vectors (linear or spherical)
        smooth (bool): whether or not to smoothly transition between dlatents
    Output:
        vectors (np.ndarray): interpolation of latent vectors, without including v1
    '''
    # Get the timesteps
    t_array = np.linspace(0, 1, num=n_steps, endpoint=False).reshape(-1, 1)
    if smooth:
        # Smooth interpolation, constructed following
        # https://math.stackexchange.com/a/1142755
        t_array = t_array**2 * (3 - 2 * t_array)
    
    # TODO: no need of a for loop; this can be optimized using the fact that they're numpy arrays!
    vectors = list()
    for t in t_array:
        if interp_type == 'linear':
            v = lerp(t, v0, v1)
        elif interp_type == 'spherical':
            v = slerp(t, v0, v1)
        vectors.append(v)
    
    return np.asarray(vectors)

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', 'seeds', type=num_range, help='Random seeds to use for interpolation', required=True)
@click.option('--steps', 'n_steps', type=int, default=120)
@click.option('--interp_type', 'interp_type', help='linear or spherical', default='spherical', show_default=True)
@click.option('--interp_space', 'interp_space', default='z', show_default=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--outdir', type=str, required=True)
@click.option('--render-program', default=None, show_default=True)
@click.option('--render-option', default=None, type=str, help="e.g. up256, camera, depth")
def generate_interpolation(
    network_pkl: str,
    seeds: List[int],
    interp_type: str,
    interp_space: str,
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    n_steps: int,
    render_program=None,
    render_option=None,
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    python style_mixing.py --outdir=out --rows=85,100,75,458,1500 --cols=55,821,1789,293 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl
    """
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    if os.path.isdir(network_pkl):
        network_pkl = sorted(glob.glob(network_pkl + '/*.pkl'))[-1]
    print('Loading networks from "%s"...' % network_pkl)

    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    os.makedirs(outdir, exist_ok=True)

    # avoid persistent classes... 
    from training.networks import Generator
    from renderer import Renderer
    from torch_utils import misc
    with torch.no_grad():
        G2 = Generator(*G.init_args, **G.init_kwargs).to(device)
        misc.copy_params_and_buffers(G, G2, require_all=False)
    G2 = Renderer(G2, None, program=None)
    w_avg = G2.generator.mapping.w_avg

    print('Generating W vectors...')
    all_z  = np.stack([np.random.RandomState(seed).randn(G.z_dim) for seed in seeds])
    all_w  = G2.generator.mapping(torch.from_numpy(all_z).to(device), None)

    # copy the same w
    # k, m = 20, 20
    # for i in range(len(all_w)):
    #     all_w[i, :m] = all_w[0, :m]
    #     # all_w[i, k:] = all_w[0, k:]
    # from fairseq import pdb;pdb.set_trace()

    kwargs = G2.get_additional_params(all_w)

    if interp_space == 'z':
        print('Interpolation in Z space')
        interp_z = [interpolate(all_z[t], all_z[t+1], n_steps=n_steps, interp_type=interp_type) for t in range(len(seeds) - 1)]
        interp_z = np.concatenate(interp_z, 0)
    elif interp_space == 'w':
        print('Interpolation in W space')
        all_w = all_w.cpu().numpy()
        interp_w = [interpolate(all_w[t], all_w[t+1], n_steps=n_steps, interp_type=interp_type) for t in range(len(seeds) - 1)]
        interp_w = np.concatenate(interp_w, 0)
    else:
        raise NotImplementedError

    interp_codes = None
    if kwargs.get('latent_codes', None) is not None:
        codes = kwargs['latent_codes']
        interp_codes = []
        for c in codes:
            if len(c.size()) != 0:
                c = c.cpu().numpy()
                c = [interpolate(c[t], c[t+1], n_steps=n_steps, interp_type=interp_type) for t in range(len(seeds) - 1)]
                interp_codes += [torch.from_numpy(np.concatenate(c, 0)).float().to(device)]
            else:
                interp_codes += [c]
    
    batch_size = 20    
    interp_images = []
    if render_program == 'rotation':
        tspace = np.linspace(0, 1, 120)
    else:
        tspace = np.zeros(10)

    for s in range(0, (len(seeds)-1) * n_steps, batch_size):
        if interp_space == 'z':
            all_z = interp_z[s: s + batch_size]
            all_w = G2.generator.mapping(torch.from_numpy(all_z).to(device), None)
        elif interp_space == 'w':
            all_w = interp_w[s: s + batch_size]
            all_w = torch.from_numpy(all_w).to(device)
            
        all_w = w_avg + (all_w - w_avg) * truncation_psi
        if interp_codes is not None:
            kwargs = {}
            # kwargs['latent_codes'] = tuple([c[s: s + batch_size] if (len(c.size())>0) else c for c in interp_codes])
            kwargs['latent_codes'] = tuple([c[:1].repeat(all_w.size(0), 1) if (len(c.size())>0) else c for c in interp_codes])
            cams = [G2.get_camera_traj(tspace[st % tspace.shape[0]], device=all_w.device) for st in range(s, s+all_w.size(0))]

            kwargs['camera_matrices'] = tuple([torch.cat([cams[j][i] for j in range(len(cams))], 0)
            if isinstance(cams[0][i], torch.Tensor) else cams[0][i]
            for i in range(len(cams[0]))])
       
        print(f'Generating images...{s}')
        all_images = G2(styles=all_w, noise_mode=noise_mode, render_option=render_option, **kwargs)

        def proc_img(img): 
            return (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu()

        if isinstance(all_images, List):
            all_images = torch.stack([proc_img(i) for i in all_images], dim=-1).numpy()
        else:
            all_images = proc_img(all_images).numpy()
        
        interp_images += [img for img in all_images]

    print('Saving image/video grid...')
    import imageio, time
    timestamp = time.strftime('%Y%m%d.%H%M%S',time.localtime(time.time()))
    network_pkl = network_pkl.split('/')[-1].split('.')[0]
    imageio.mimwrite(f'{outdir}/interp_{network_pkl}_{timestamp}.mp4', interp_images, fps=30, quality=8)
    
    outdir = f'{outdir}/{network_pkl}_{timestamp}_{seeds}'
    os.makedirs(outdir, exist_ok=True)
    for step, img in enumerate(interp_images):
        PIL.Image.fromarray(img, 'RGB').save(f'{outdir}/{step:04d}.png')
    # from fairseq import pdb;pdb.set_trace()

#----------------------------------------------------------------------------

if __name__ == "__main__":
    with torch.no_grad():
        generate_interpolation() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
