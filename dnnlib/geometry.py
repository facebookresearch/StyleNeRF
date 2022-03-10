# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import torch
import torch.nn.functional as F
import math
import random
import numpy as np


def positional_encoding(p, size, pe='normal', use_pos=False):
    if pe == 'gauss':
        p_transformed = np.pi * p @ size
        p_transformed = torch.cat(
            [torch.sin(p_transformed), torch.cos(p_transformed)], dim=-1)
    else:
        p_transformed = torch.cat([torch.cat(
            [torch.sin((2 ** i) * np.pi * p),
            torch.cos((2 ** i) * np.pi * p)],
            dim=-1) for i in range(size)], dim=-1)
    if use_pos:
        p_transformed = torch.cat([p_transformed, p], -1)
    return p_transformed


def upsample(img_nerf, size, filter=None):
    up = size // img_nerf.size(-1)
    if up <= 1:
        return img_nerf

    if filter is not None:
        from torch_utils.ops import upfirdn2d
        for _ in range(int(math.log2(up))):
            img_nerf = upfirdn2d.downsample2d(img_nerf, filter, up=2)
    else:
        img_nerf = F.interpolate(img_nerf, (size, size), mode='bilinear', align_corners=False)
    return img_nerf


def downsample(img0, size, filter=None):
    down = img0.size(-1) // size
    if down <= 1:
        return img0

    if filter is not None:    
        from torch_utils.ops import upfirdn2d
        for _ in range(int(math.log2(down))):
            img0 = upfirdn2d.downsample2d(img0, filter, down=2)
    else:
        img0 = F.interpolate(img0, (size, size), mode='bilinear', align_corners=False)
    return img0
        

def normalize_vecs(vectors: torch.Tensor) -> torch.Tensor:
    """
    Normalize vector lengths.
    """
    return vectors / (torch.norm(vectors, dim=-1, keepdim=True))


def repeat_vecs(vecs, n, dim=0):
    return torch.stack(n*[vecs], dim=dim)


def get_grids(H, W, device, align=True):
    ch = 1 if align else 1 - (1 / H)
    cw = 1 if align else 1 - (1 / W)
    x, y = torch.meshgrid(torch.linspace(-cw, cw, W, device=device),
                          torch.linspace(ch, -ch, H, device=device))
    return torch.stack([x, y], -1)


def local_ensemble(pi, po, resolution):
    ii = range(resolution)
    ia = torch.tensor([max((i - 1)//2, 0) for i in ii]).long()
    ib = torch.tensor([min((i + 1)//2, resolution//2-1) for i in ii]).long()
    
    ul = torch.meshgrid(ia, ia)
    ur = torch.meshgrid(ia, ib)
    ll = torch.meshgrid(ib, ia)
    lr = torch.meshgrid(ib, ib)
    
    d_ul, p_ul = po - pi[ul], torch.stack(ul, -1)
    d_ur, p_ur = po - pi[ur], torch.stack(ur, -1)
    d_ll, p_ll = po - pi[ll], torch.stack(ll, -1)
    d_lr, p_lr = po - pi[lr], torch.stack(lr, -1)
    
    c_ul = d_ul.prod(dim=-1).abs()
    c_ur = d_ur.prod(dim=-1).abs()
    c_ll = d_ll.prod(dim=-1).abs()
    c_lr = d_lr.prod(dim=-1).abs()

    D = torch.stack([d_ul, d_ur, d_ll, d_lr], 0)
    P = torch.stack([p_ul, p_ur, p_ll, p_lr], 0)
    C = torch.stack([c_ul, c_ur, c_ll, c_lr], 0)
    C = C / C.sum(dim=0, keepdim=True)
    return D, P, C


def get_initial_rays_trig(num_steps, fov, resolution, ray_start, ray_end, device='cpu'):
    """Returns sample points, z_vals, ray directions in camera space."""

    W, H = resolution
    # Create full screen NDC (-1 to +1) coords [x, y, 0, 1].
    # Y is flipped to follow image memory layouts.
    x, y = torch.meshgrid(torch.linspace(-1, 1, W, device=device),
                          torch.linspace(1, -1, H, device=device))
    x = x.T.flatten()
    y = y.T.flatten()
    z = -torch.ones_like(x, device=device) / math.tan((2 * math.pi * fov / 360)/2)

    rays_d_cam = normalize_vecs(torch.stack([x, y, z], -1))

    z_vals = torch.linspace(ray_start, ray_end, num_steps, device=device).reshape(1, num_steps, 1).repeat(W*H, 1, 1)
    points = rays_d_cam.unsqueeze(1).repeat(1, num_steps, 1) * z_vals
    return points, z_vals, rays_d_cam


def sample_camera_positions(
    device, n=1, r=1, horizontal_stddev=1, vertical_stddev=1, 
    horizontal_mean=math.pi*0.5, vertical_mean=math.pi*0.5, mode='normal'):
    """
    Samples n random locations along a sphere of radius r. 
    Uses a gaussian distribution for pitch and yaw
    """
    if mode == 'uniform':
        theta = (torch.rand((n, 1),device=device) - 0.5) * 2 * horizontal_stddev + horizontal_mean
        phi = (torch.rand((n, 1),device=device) - 0.5) * 2 * vertical_stddev + vertical_mean

    elif mode == 'normal' or mode == 'gaussian':
        theta = torch.randn((n, 1), device=device) * horizontal_stddev + horizontal_mean
        phi = torch.randn((n, 1), device=device) * vertical_stddev + vertical_mean

    elif mode == 'hybrid':
        if random.random() < 0.5:
            theta = (torch.rand((n, 1),device=device) - 0.5) * 2 * horizontal_stddev * 2 + horizontal_mean
            phi = (torch.rand((n, 1),device=device) - 0.5) * 2 * vertical_stddev * 2 + vertical_mean
        else:
            theta = torch.randn((n, 1), device=device) * horizontal_stddev + horizontal_mean
            phi = torch.randn((n, 1), device=device) * vertical_stddev + vertical_mean
    else:
        phi = torch.ones((n, 1), device=device, dtype=torch.float) * vertical_mean
        theta = torch.ones((n, 1), device=device, dtype=torch.float) * horizontal_mean

    phi = torch.clamp(phi, 1e-5, math.pi - 1e-5)

    output_points = torch.zeros((n, 3), device=device)# torch.cuda.FloatTensor(n, 3).fill_(0)#torch.zeros((n, 3))

    output_points[:, 0:1] = r*torch.sin(phi) * torch.cos(theta)
    output_points[:, 2:3] = r*torch.sin(phi) * torch.sin(theta)
    output_points[:, 1:2] = r*torch.cos(phi)

    return output_points, phi, theta


def perturb_points(points, z_vals, ray_directions, device):
    distance_between_points = z_vals[:,:,1:2,:] - z_vals[:,:,0:1,:]
    offset = (torch.rand(z_vals.shape, device=device)-0.5) * distance_between_points
    z_vals = z_vals + offset
    points = points + offset * ray_directions.unsqueeze(2)
    return points, z_vals


def create_cam2world_matrix(forward_vector, origin, device=None):
    """Takes in the direction the camera is pointing and the camera origin and returns a world2cam matrix."""

    forward_vector = normalize_vecs(forward_vector)
    up_vector = torch.tensor([0, 1, 0], dtype=torch.float, device=device).expand_as(forward_vector)
    left_vector = normalize_vecs(torch.cross(up_vector, forward_vector, dim=-1))
    up_vector = normalize_vecs(torch.cross(forward_vector, left_vector, dim=-1))

    rotation_matrix = torch.eye(4, device=device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    rotation_matrix[:, :3, :3] = torch.stack((-left_vector, up_vector, -forward_vector), axis=-1)

    translation_matrix = torch.eye(4, device=device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    translation_matrix[:, :3, 3] = origin

    cam2world = translation_matrix @ rotation_matrix

    return cam2world


def transform_sampled_points(
    points, z_vals, ray_directions, device, 
    h_stddev=1, v_stddev=1, h_mean=math.pi * 0.5, 
    v_mean=math.pi * 0.5, mode='normal'):
    """
    points: batch_size x total_pixels x num_steps x 3
    z_vals: batch_size x total_pixels x num_steps
    """
    n, num_rays, num_steps, channels = points.shape
    points, z_vals = perturb_points(points, z_vals, ray_directions, device)
    camera_origin, pitch, yaw = sample_camera_positions(
        n=points.shape[0], r=1, 
        horizontal_stddev=h_stddev, vertical_stddev=v_stddev, 
        horizontal_mean=h_mean, vertical_mean=v_mean, 
        device=device, mode=mode)
    forward_vector = normalize_vecs(-camera_origin)
    cam2world_matrix = create_cam2world_matrix(forward_vector, camera_origin, device=device)

    points_homogeneous = torch.ones((points.shape[0], points.shape[1], points.shape[2], points.shape[3] + 1), device=device)
    points_homogeneous[:, :, :, :3] = points

    # should be n x 4 x 4 , n x r^2 x num_steps x 4
    transformed_points = torch.bmm(cam2world_matrix, points_homogeneous.reshape(n, -1, 4).permute(0,2,1)).permute(0, 2, 1).reshape(n, num_rays, num_steps, 4)
    transformed_ray_directions = torch.bmm(cam2world_matrix[..., :3, :3], ray_directions.reshape(n, -1, 3).permute(0,2,1)).permute(0, 2, 1).reshape(n, num_rays, 3)

    homogeneous_origins = torch.zeros((n, 4, num_rays), device=device)
    homogeneous_origins[:, 3, :] = 1

    transformed_ray_origins = torch.bmm(cam2world_matrix, homogeneous_origins).permute(0, 2, 1).reshape(n, num_rays, 4)[..., :3]
    return transformed_points[..., :3], z_vals, transformed_ray_directions, transformed_ray_origins, pitch, yaw


def integration(
    rgb_sigma, z_vals, device, noise_std=0.5,
    last_back=False, white_back=False, clamp_mode=None, fill_mode=None):

    rgbs = rgb_sigma[..., :3]
    sigmas = rgb_sigma[..., 3:]

    deltas = z_vals[..., 1:, :] - z_vals[..., :-1, :]
    delta_inf = 1e10 * torch.ones_like(deltas[..., :1, :])
    deltas = torch.cat([deltas, delta_inf], -2)

    if noise_std > 0:
        noise = torch.randn(sigmas.shape, device=device) * noise_std
    else:
        noise = 0

    if clamp_mode == 'softplus':
        alphas = 1 - torch.exp(-deltas * (F.softplus(sigmas + noise)))
    elif clamp_mode == 'relu':
        alphas = 1 - torch.exp(-deltas * (F.relu(sigmas + noise)))
    else:
        raise "Need to choose clamp mode"

    alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1, :]), 1-alphas + 1e-10], -2)
    weights = alphas * torch.cumprod(alphas_shifted, -2)[..., :-1, :]
    weights_sum = weights.sum(-2)

    if last_back:
        weights[..., -1, :] += (1 - weights_sum)

    rgb_final = torch.sum(weights * rgbs, -2)
    depth_final = torch.sum(weights * z_vals, -2)

    if white_back:
        rgb_final = rgb_final + 1-weights_sum

    if fill_mode == 'debug':
        rgb_final[weights_sum.squeeze(-1) < 0.9] = torch.tensor([1., 0, 0], device=rgb_final.device)
    elif fill_mode == 'weight':
        rgb_final = weights_sum.expand_as(rgb_final)

    return rgb_final, depth_final, weights


def get_sigma_field_np(nerf, styles, resolution=512, block_resolution=64):
    # return numpy array of forwarded sigma value
    bound = (nerf.depth_range[1] - nerf.depth_range[0]) * 0.5
    X = torch.linspace(-bound, bound, resolution).split(block_resolution)

    sigma_np = np.zeros([resolution, resolution, resolution], dtype=np.float32)

    for xi, xs in enumerate(X):
        for yi, ys in enumerate(X):
            for zi, zs in enumerate(X):
                xx, yy, zz = torch.meshgrid(xs, ys, zs)
                pts = torch.stack([xx, yy, zz], dim=-1).unsqueeze(0).to(styles.device)  # B, H, H, H, C
                block_shape = [1, len(xs), len(ys), len(zs)]
                feat_out, sigma_out = nerf.fg_nerf.forward_style2(pts, None, block_shape, ws=styles)
                sigma_np[xi * block_resolution: xi * block_resolution + len(xs), \
                yi * block_resolution: yi * block_resolution + len(ys), \
                zi * block_resolution: zi * block_resolution + len(zs)] = sigma_out.reshape(block_shape[1:]).detach().cpu().numpy()

    return sigma_np, bound


def extract_geometry(nerf, styles, resolution, threshold):
    import mcubes

    print('threshold: {}'.format(threshold))
    u, bound = get_sigma_field_np(nerf, styles, resolution)
    vertices, triangles = mcubes.marching_cubes(u, threshold)
    b_min_np = np.array([-bound, -bound, -bound])
    b_max_np = np.array([ bound,  bound,  bound])

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices.astype('float32'), triangles


def render_mesh(meshes, camera_matrices, render_noise=True):
    from pytorch3d.renderer import (
        FoVPerspectiveCameras, look_at_view_transform,
        RasterizationSettings, BlendParams,
        MeshRenderer, MeshRasterizer, HardPhongShader, TexturesVertex
    )
    from pytorch3d.ops import interpolate_face_attributes
    from pytorch3d.structures.meshes import Meshes

    intrinsics, poses, _, _ = camera_matrices
    device = poses.device
    c2w = torch.matmul(poses, torch.diag(torch.tensor([-1.0, 1.0, -1.0, 1.0], device=device))[None, :, :])  # Different camera model...
    w2c = torch.inverse(c2w)
    R = c2w[:, :3, :3]
    T = w2c[:, :3, 3]   # So weird..... Why one is c2w and another is w2c?
    focal = intrinsics[0, 0, 0]
    fov = torch.arctan(focal) * 2.0 / np.pi * 180
    

    colors = []
    offset = 1
    for res, (mesh, face_vert_noise) in meshes.items():
        raster_settings = RasterizationSettings(
                image_size=res,
                blur_radius=0.0,
                faces_per_pixel=1,
        )
        mesh = Meshes(
            verts=[torch.from_numpy(mesh.vertices).float().to(device)],
            faces=[torch.from_numpy(mesh.faces).long().to(device)])

        _colors = []
        for i in range(len(poses)):
            cameras = FoVPerspectiveCameras(device=device, R=R[i: i+1], T=T[i: i+1], fov=fov)
            rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
            pix_to_face, zbuf, bary_coord, dists = rasterizer(mesh)
            color = interpolate_face_attributes(pix_to_face, bary_coord, face_vert_noise).squeeze()

            # hack
            color[offset:, offset:] = color[:-offset, :-offset]
            _colors += [color]
        color = torch.stack(_colors, 0).permute(0,3,1,2)
        colors += [color]
        offset *= 2
    return colors


def rotate_vects(v, theta):
    theta = theta / math.pi * 2
    theta = theta + (theta < 0).type_as(theta) * 4
    v  = v.reshape(v.size(0), v.size(1) // 4, 4, v.size(2), v.size(3))
    vs = []
    order  = [0,2,3,1]  # Not working
    iorder = [0,3,1,2]  # Not working
    for b in range(len(v)):
        if (theta[b] - 0) < 1e-6:
            u, l = 0, 0
        elif (theta[b] - 1) < 1e-6:
            u, l = 0, 1
        elif (theta[b] - 2) < 1e-6:
            u, l = 0, 2
        elif (theta[b] - 3) < 1e-6:
            u, l = 0, 3
        else:
            u, l = math.modf(theta[b])
        l, r = int(l), int(l + 1) % 4
        vv = v[b, :, order]  # 0 -> 1 -> 3 -> 2
        vl   = torch.cat([vv[:, l:], vv[:, :l]], 1)
        if u > 0:
            vr = torch.cat([vv[:, r:], vv[:, :r]], 1)
            vv = vl * (1-u) + vr * u
        else:
            vv = vl
        vs.append(vv[:, iorder])
    v = torch.stack(vs, 0)
    v = v.reshape(v.size(0), -1, v.size(-2), v.size(-1))
    return v


def generate_option_outputs(render_option):
    # output debugging outputs (not used in normal rendering process)
    if ('depth' in render_option.split(',')):    
        img = camera_world[:, :1] + fg_depth_map * ray_vector
        img = reformat(img, tgt_res)

        if 'gradient' in render_option.split(','):
            points = (camera_world[:,:,None]+di[:,:,:,None]*ray_vector[:,:,None]).reshape(
                batch_size, tgt_res, tgt_res, di.size(-1), 3)
            with torch.enable_grad():
                gradients = self.fg_nerf.forward_style2(
                    points, None, [batch_size, tgt_res, di.size(-1), tgt_res], get_normal=True,
                    ws=styles, z_shape=z_shape_obj, z_app=z_app_obj).reshape(
                        batch_size, di.size(-1), 3, tgt_res * tgt_res).permute(0,3,1,2)
                avg_grads = (gradients * fg_weights.unsqueeze(-1)).sum(-2)
            normal = reformat(normalize(avg_grads, axis=2)[0], tgt_res)
            img = normal

        if 'value' in render_option.split(','):
            fg_feat = fg_feat[:,:,3:].norm(dim=-1,keepdim=True)
            img = reformat(fg_feat.repeat(1,1,3), tgt_res) / fg_feat.max() * 2 - 1
            
        if 'opacity' in render_option.split(','):
            opacity = bg_lambda.unsqueeze(-1).repeat(1,1,3) * 2 - 1
            img = reformat(opacity, tgt_res)

        if 'normal' in render_option.split(','):
            shift_l, shift_r = img[:,:,2:,:], img[:,:,:-2,:]
            shift_u, shift_d = img[:,:,:,2:], img[:,:,:,:-2]
            diff_hor = normalize(shift_r - shift_l, axis=1)[0][:, :, :, 1:-1]
            diff_ver = normalize(shift_u - shift_d, axis=1)[0][:, :, 1:-1, :]
            normal = torch.cross(diff_hor, diff_ver, dim=1)
            img = normalize(normal, axis=1)[0]
        
        return {'full_out': (None, img), 'reg_loss': {}}
