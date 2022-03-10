# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import numpy as np
from numpy.lib.function_base import angle
import torch
import torch.nn.functional as F
import math

from scipy.spatial.transform import Rotation as Rot
HUGE_NUMBER = 1e10
TINY_NUMBER = 1e-6      # float32 only has 7 decimal digits precision


def get_camera_mat(fov=49.13, invert=True):
    # fov = 2 * arctan(sensor / (2 * focal))
    # focal = (sensor / 2)  * 1 / (tan(0.5 * fov))
    # in our case, sensor = 2 as pixels are in [-1, 1]
    focal = 1. / np.tan(0.5 * fov * np.pi/180.)
    focal = focal.astype(np.float32)
    mat = torch.tensor([
        [focal, 0., 0., 0.],
        [0., focal, 0., 0.],
        [0., 0., 1, 0.],
        [0., 0., 0., 1.]
    ]).reshape(1, 4, 4)
    if invert:
        mat = torch.inverse(mat)
    return mat


def get_random_pose(range_u, range_v, range_radius, batch_size=32,
                    invert=False, gaussian=False, angular=False):
    loc, (u, v) = sample_on_sphere(range_u, range_v, size=(batch_size), gaussian=gaussian, angular=angular)
    radius = range_radius[0] + torch.rand(batch_size) * (range_radius[1] - range_radius[0])
    loc = loc * radius.unsqueeze(-1)
    R = look_at(loc)
    RT = torch.eye(4).reshape(1, 4, 4).repeat(batch_size, 1, 1)
    RT[:, :3, :3] = R
    RT[:, :3, -1] = loc

    if invert:
        RT = torch.inverse(RT)
    
    def N(a, range_a):
        if range_a[0] == range_a[1]:
            return a * 0
        return (a - range_a[0]) / (range_a[1] - range_a[0])
    
    val_u, val_v, val_r = N(u, range_u), N(v, range_v), N(radius, range_radius)
    return RT, (val_u, val_v, val_r)


def get_camera_pose(range_u, range_v, range_r, val_u=0.5, val_v=0.5, val_r=0.5,
                    batch_size=32, invert=False,  gaussian=False, angular=False):
    r0, rr = range_r[0], range_r[1] - range_r[0]
    r = r0 + val_r * rr
    if not gaussian:
        u0, ur = range_u[0], range_u[1] - range_u[0]
        v0, vr = range_v[0], range_v[1] - range_v[0]   
        u = u0 + val_u * ur
        v = v0 + val_v * vr
    else:
        mean_u, mean_v = sum(range_u) / 2, sum(range_v) / 2
        vu, vv = mean_u - range_u[0], mean_v - range_v[0]
        u = mean_u + vu * val_u
        v = mean_v + vv * val_v
        
    loc, _ = sample_on_sphere((u, u), (v, v), size=(batch_size), angular=angular)
    radius = torch.ones(batch_size) * r
    loc = loc * radius.unsqueeze(-1)
    R = look_at(loc)
    RT = torch.eye(4).reshape(1, 4, 4).repeat(batch_size, 1, 1)
    RT[:, :3, :3] = R
    RT[:, :3, -1] = loc

    if invert:
        RT = torch.inverse(RT)
    return RT


def get_camera_pose_v2(range_u, range_v, range_r, mode, invert=False, gaussian=False, angular=False):
    r0, rr = range_r[0], range_r[1] - range_r[0]
    val_u, val_v = mode[:,0], mode[:,1]
    val_r = torch.ones_like(val_u) * 0.5
    if not gaussian:
        u0, ur = range_u[0], range_u[1] - range_u[0]
        v0, vr = range_v[0], range_v[1] - range_v[0]
        u = u0 + val_u * ur
        v = v0 + val_v * vr
    else:
        mean_u, mean_v = sum(range_u) / 2, sum(range_v) / 2
        vu, vv = mean_u - range_u[0], mean_v - range_v[0]
        u = mean_u + vu * val_u
        v = mean_v + vv * val_v
    
    loc = to_sphere(u, v, angular)
    radius = r0 + val_r * rr
    loc = loc * radius.unsqueeze(-1)
    R = look_at(loc)
    RT = torch.eye(4).to(R.device).reshape(1, 4, 4).repeat(R.size(0), 1, 1)
    RT[:, :3, :3] = R
    RT[:, :3, -1] = loc

    if invert:
        RT = torch.inverse(RT)
    return RT, (val_u, val_v, val_r)


def to_sphere(u, v, angular=False):
    T = torch if isinstance(u, torch.Tensor) else np
    if not angular:
        theta = 2 * math.pi * u
        phi = T.arccos(1 - 2 * v)
    else:
        theta, phi = u, v
    
    cx = T.sin(phi) * T.cos(theta)
    cy = T.sin(phi) * T.sin(theta)
    cz = T.cos(phi)
    return T.stack([cx, cy, cz], -1)


def sample_on_sphere(range_u=(0, 1), range_v=(0, 1), size=(1,),
                     to_pytorch=True, gaussian=False, angular=False):
    if not gaussian:
        u = np.random.uniform(*range_u, size=size)
        v = np.random.uniform(*range_v, size=size)
    else:
        mean_u, mean_v = sum(range_u) / 2, sum(range_v) / 2
        var_u, var_v = mean_u - range_u[0], mean_v - range_v[0]
        u = np.random.normal(size=size) * var_u + mean_u
        v = np.random.normal(size=size) * var_v + mean_v

    sample = to_sphere(u, v, angular)
    if to_pytorch:
        sample = torch.tensor(sample).float()
        u, v = torch.tensor(u).float(), torch.tensor(v).float()

    return sample, (u, v)


def look_at(eye, at=np.array([0, 0, 0]), up=np.array([0, 0, 1]), eps=1e-5,
            to_pytorch=True):
    if not isinstance(eye, torch.Tensor):
        # this is the original code from GRAF
        at = at.astype(float).reshape(1, 3)
        up = up.astype(float).reshape(1, 3)
        eye = eye.reshape(-1, 3)
        up = up.repeat(eye.shape[0] // up.shape[0], axis=0)
        eps = np.array([eps]).reshape(1, 1).repeat(up.shape[0], axis=0)
        z_axis = eye - at
        z_axis /= np.max(np.stack([np.linalg.norm(z_axis,
                                                axis=1, keepdims=True), eps]))
        x_axis = np.cross(up, z_axis)
        x_axis /= np.max(np.stack([np.linalg.norm(x_axis,
                                                axis=1, keepdims=True), eps]))
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.max(np.stack([np.linalg.norm(y_axis,
                                                axis=1, keepdims=True), eps]))
        r_mat = np.concatenate(
            (x_axis.reshape(-1, 3, 1), y_axis.reshape(-1, 3, 1), z_axis.reshape(
                -1, 3, 1)), axis=2)
        if to_pytorch:
            r_mat = torch.tensor(r_mat).float()
    else:
        
        def normalize(x, axis=-1, order=2):
            l2 = x.norm(p=order, dim=axis, keepdim=True).clamp(min=1e-8)
            return x / l2
        
        at, up = torch.from_numpy(at).float().to(eye.device), torch.from_numpy(up).float().to(eye.device)
        z_axis = normalize(eye - at[None, :])
        x_axis = normalize(torch.cross(up[None,:].expand_as(z_axis), z_axis, dim=-1))
        y_axis = normalize(torch.cross(z_axis, x_axis, dim=-1))
        r_mat = torch.stack([x_axis, y_axis, z_axis], dim=-1)

    return r_mat


def get_rotation_matrix(axis='z', value=0., batch_size=32):
    r = Rot.from_euler(axis, value * 2 * np.pi).as_dcm()
    r = torch.from_numpy(r).reshape(1, 3, 3).repeat(batch_size, 1, 1)
    return r


def get_corner_rays(corner_pixels, camera_matrices, res):
    assert (res + 1) * (res + 1) == corner_pixels.size(1)
    batch_size = camera_matrices[0].size(0)
    rays, origins, _ = get_camera_rays(camera_matrices, corner_pixels)
    corner_rays = torch.cat([rays, torch.cross(origins, rays, dim=-1)], -1)
    corner_rays = corner_rays.reshape(batch_size, res+1, res+1, 6).permute(0,3,1,2)
    corner_rays = torch.cat([corner_rays[..., :-1, :-1], corner_rays[..., 1:, :-1], corner_rays[..., 1:, 1:], corner_rays[..., :-1, 1:]], 1)
    return corner_rays
    

def arange_pixels(
        resolution=(128, 128), 
        batch_size=1, 
        subsample_to=None, 
        invert_y_axis=False, 
        margin=0,
        corner_aligned=True,
        jitter=None
    ):
    ''' Arranges pixels for given resolution in range image_range.

    The function returns the unscaled pixel locations as integers and the
    scaled float values.

    Args:
        resolution (tuple): image resolution
        batch_size (int): batch size
        subsample_to (int): if integer and > 0, the points are randomly
            subsampled to this value
    '''
    h, w = resolution
    n_points = resolution[0] * resolution[1]
    uh = 1 if corner_aligned else 1 - (1 / h)
    uw = 1 if corner_aligned else 1 - (1 / w)
    if margin > 0:
        uh = uh + (2 / h) * margin
        uw = uw + (2 / w) * margin 
        w, h = w + margin * 2, h + margin * 2

    x, y = torch.linspace(-uw, uw, w), torch.linspace(-uh, uh, h)
    if jitter is not None:
        dx = (torch.ones_like(x).uniform_() - 0.5) * 2 / w * jitter
        dy = (torch.ones_like(y).uniform_() - 0.5) * 2 / h * jitter
        x, y = x + dx, y + dy
    x, y = torch.meshgrid(x, y)
    pixel_scaled = torch.stack([x, y], -1).permute(1,0,2).reshape(1, -1, 2).repeat(batch_size, 1, 1)
    
    # Subsample points if subsample_to is not None and > 0
    if (subsample_to is not None and subsample_to > 0 and subsample_to < n_points):
        idx = np.random.choice(pixel_scaled.shape[1], size=(subsample_to,),
                               replace=False)
        pixel_scaled = pixel_scaled[:, idx]

    if invert_y_axis:
        pixel_scaled[..., -1] *= -1.

    return pixel_scaled


def to_pytorch(tensor, return_type=False):
    ''' Converts input tensor to pytorch.

    Args:
        tensor (tensor): Numpy or Pytorch tensor
        return_type (bool): whether to return input type
    '''
    is_numpy = False
    if type(tensor) == np.ndarray:
        tensor = torch.from_numpy(tensor)
        is_numpy = True
    tensor = tensor.clone()
    if return_type:
        return tensor, is_numpy
    return tensor


def transform_to_world(pixels, depth, camera_mat, world_mat, scale_mat=None,
                       invert=True, use_absolute_depth=True):
    ''' Transforms pixel positions p with given depth value d to world coordinates.

    Args:
        pixels (tensor): pixel tensor of size B x N x 2
        depth (tensor): depth tensor of size B x N x 1
        camera_mat (tensor): camera matrix
        world_mat (tensor): world matrix
        scale_mat (tensor): scale matrix
        invert (bool): whether to invert matrices (default: true)
    '''
    assert(pixels.shape[-1] == 2)
    if scale_mat is None:
        scale_mat = torch.eye(4).unsqueeze(0).repeat(
            camera_mat.shape[0], 1, 1).to(camera_mat.device)

    # Convert to pytorch
    pixels, is_numpy = to_pytorch(pixels, True)
    depth = to_pytorch(depth)
    camera_mat = to_pytorch(camera_mat)
    world_mat = to_pytorch(world_mat)
    scale_mat = to_pytorch(scale_mat)

    # Invert camera matrices
    if invert:
        camera_mat = torch.inverse(camera_mat)
        world_mat = torch.inverse(world_mat)
        scale_mat = torch.inverse(scale_mat)

    # Transform pixels to homogen coordinates
    pixels = pixels.permute(0, 2, 1)
    pixels = torch.cat([pixels, torch.ones_like(pixels)], dim=1)

    # Project pixels into camera space
    if use_absolute_depth:
        pixels[:, :2] = pixels[:, :2] * depth.permute(0, 2, 1).abs()
        pixels[:, 2:3] = pixels[:, 2:3] * depth.permute(0, 2, 1)
    else:
        pixels[:, :3] = pixels[:, :3] * depth.permute(0, 2, 1)
    
    # Transform pixels to world space
    p_world = scale_mat @ world_mat @ camera_mat @ pixels

    # Transform p_world back to 3D coordinates
    p_world = p_world[:, :3].permute(0, 2, 1)

    if is_numpy:
        p_world = p_world.numpy()
    return p_world


def transform_to_camera_space(p_world, world_mat, camera_mat=None, scale_mat=None):
    ''' Transforms world points to camera space.
        Args:
        p_world (tensor): world points tensor of size B x N x 3
        camera_mat (tensor): camera matrix
        world_mat (tensor): world matrix
        scale_mat (tensor): scale matrix
    '''
    batch_size, n_p, _ = p_world.shape
    device = p_world.device

    # Transform world points to homogen coordinates
    p_world = torch.cat([p_world, torch.ones(
        batch_size, n_p, 1).to(device)], dim=-1).permute(0, 2, 1)

    # Apply matrices to transform p_world to camera space
    if scale_mat is None:
        if camera_mat is None:
            p_cam = world_mat @ p_world
        else:
            p_cam = camera_mat @ world_mat @ p_world
    else:
        p_cam = camera_mat @ world_mat @ scale_mat @ p_world

    # Transform points back to 3D coordinates
    p_cam = p_cam[:, :3].permute(0, 2, 1)
    return p_cam


def origin_to_world(n_points, camera_mat, world_mat, scale_mat=None,
                    invert=False):
    ''' Transforms origin (camera location) to world coordinates.

    Args:
        n_points (int): how often the transformed origin is repeated in the
            form (batch_size, n_points, 3)
        camera_mat (tensor): camera matrix
        world_mat (tensor): world matrix
        scale_mat (tensor): scale matrix
        invert (bool): whether to invert the matrices (default: true)
    '''
    batch_size = camera_mat.shape[0]
    device = camera_mat.device
    # Create origin in homogen coordinates
    p = torch.zeros(batch_size, 4, n_points).to(device)
    p[:, -1] = 1.

    if scale_mat is None:
        scale_mat = torch.eye(4).unsqueeze(
            0).repeat(batch_size, 1, 1).to(device)

    # Invert matrices
    if invert:
        camera_mat = torch.inverse(camera_mat)
        world_mat = torch.inverse(world_mat)
        scale_mat = torch.inverse(scale_mat)

    # Apply transformation
    p_world = scale_mat @ world_mat @ camera_mat @ p

    # Transform points back to 3D coordinates
    p_world = p_world[:, :3].permute(0, 2, 1)
    return p_world


def image_points_to_world(image_points, camera_mat, world_mat, scale_mat=None,
                          invert=False, negative_depth=True):
    ''' Transforms points on image plane to world coordinates.

    In contrast to transform_to_world, no depth value is needed as points on
    the image plane have a fixed depth of 1.

    Args:
        image_points (tensor): image points tensor of size B x N x 2
        camera_mat (tensor): camera matrix
        world_mat (tensor): world matrix
        scale_mat (tensor): scale matrix
        invert (bool): whether to invert matrices
    '''
    batch_size, n_pts, dim = image_points.shape
    assert(dim == 2)
    device = image_points.device
    d_image = torch.ones(batch_size, n_pts, 1).to(device)
    if negative_depth:
        d_image *= -1.
    return transform_to_world(image_points, d_image, camera_mat, world_mat,
                              scale_mat, invert=invert)


def image_points_to_camera(image_points, camera_mat, 
                           invert=False, negative_depth=True, use_absolute_depth=True):
    batch_size, n_pts, dim = image_points.shape
    assert(dim == 2)
    device = image_points.device
    d_image = torch.ones(batch_size, n_pts, 1).to(device)
    if negative_depth:
        d_image *= -1.

    # Convert to pytorch
    pixels, is_numpy = to_pytorch(image_points, True)
    depth = to_pytorch(d_image)
    camera_mat = to_pytorch(camera_mat)

    # Invert camera matrices
    if invert:
        camera_mat = torch.inverse(camera_mat)
    
    # Transform pixels to homogen coordinates
    pixels = pixels.permute(0, 2, 1)
    pixels = torch.cat([pixels, torch.ones_like(pixels)], dim=1)

    # Project pixels into camera space
    if use_absolute_depth:
        pixels[:, :2] = pixels[:, :2] * depth.permute(0, 2, 1).abs()
        pixels[:, 2:3] = pixels[:, 2:3] * depth.permute(0, 2, 1)
    else:
        pixels[:, :3] = pixels[:, :3] * depth.permute(0, 2, 1)

    # Transform pixels to world space
    p_camera = camera_mat @ pixels

    # Transform p_world back to 3D coordinates
    p_camera = p_camera[:, :3].permute(0, 2, 1)

    if is_numpy:
        p_camera = p_camera.numpy()
    return p_camera


def camera_points_to_image(camera_points, camera_mat, 
                           invert=False, negative_depth=True, use_absolute_depth=True):
    batch_size, n_pts, dim = camera_points.shape
    assert(dim == 3)
    device = camera_points.device

    # Convert to pytorch
    p_camera, is_numpy = to_pytorch(camera_points, True)
    camera_mat = to_pytorch(camera_mat)

    # Invert camera matrices
    if invert:
        camera_mat = torch.inverse(camera_mat)

    # Transform world camera space to pixels
    p_camera = p_camera.permute(0, 2, 1)  # B x 3 x N
    pixels = camera_mat[:, :3, :3] @ p_camera

    assert use_absolute_depth and negative_depth
    pixels, p_depths = pixels[:, :2], pixels[:, 2:3]
    p_depths = -p_depths  # negative depth
    pixels = pixels / p_depths

    pixels = pixels.permute(0, 2, 1)
    if is_numpy:
        pixels = pixels.numpy()
    return pixels


def angular_interpolation(res, camera_mat):
    batch_size = camera_mat.shape[0]
    device = camera_mat.device
    input_rays  = image_points_to_camera(arange_pixels((res, res), batch_size, 
        invert_y_axis=True).to(device), camera_mat)
    output_rays = image_points_to_camera(arange_pixels((res * 2, res * 2), batch_size,
        invert_y_axis=True).to(device), camera_mat)
    input_rays  = input_rays / input_rays.norm(dim=-1, keepdim=True)
    output_rays = output_rays / output_rays.norm(dim=-1, keepdim=True)

    def dir2sph(v):
        u = (v[..., :2] ** 2).sum(-1).sqrt()
        theta = torch.atan2(u, v[..., 2]) / math.pi
        phi = torch.atan2(v[..., 1], v[..., 0]) / math.pi
        return torch.stack([theta, phi], 1)

    input_rays  = dir2sph(input_rays).reshape(batch_size, 2, res, res)
    output_rays = dir2sph(output_rays).reshape(batch_size, 2, res * 2, res * 2)
    return input_rays


def interpolate_sphere(z1, z2, t):
    p = (z1 * z2).sum(dim=-1, keepdim=True)
    p = p / z1.pow(2).sum(dim=-1, keepdim=True).sqrt()
    p = p / z2.pow(2).sum(dim=-1, keepdim=True).sqrt()
    omega = torch.acos(p)
    s1 = torch.sin((1-t)*omega)/torch.sin(omega)
    s2 = torch.sin(t*omega)/torch.sin(omega)
    z = s1 * z1 + s2 * z2
    return z


def get_camera_rays(camera_matrices, pixels=None, res=None, margin=0):
    device     = camera_matrices[0].device
    batch_size = camera_matrices[0].shape[0]
    if pixels is None:
        assert res is not None
        pixels = arange_pixels((res, res), batch_size, invert_y_axis=True, margin=margin).to(device)
    n_points = pixels.size(1)
    pixels_world = image_points_to_world(
            pixels, camera_mat=camera_matrices[0],
            world_mat=camera_matrices[1])
    camera_world = origin_to_world(
            n_points, camera_mat=camera_matrices[0],
            world_mat=camera_matrices[1])
    ray_vector = pixels_world - camera_world
    ray_vector = ray_vector / ray_vector.norm(dim=-1, keepdim=True)
    return ray_vector, camera_world, pixels_world


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def camera_9d_to_16d(d9):
    d6, translation = d9[..., :6], d9[..., 6:]
    rotation = rotation_6d_to_matrix(d6)
    RT = torch.eye(4).to(device=d9.device, dtype=d9.dtype).reshape(
        1, 4, 4).repeat(d6.size(0), 1, 1)
    RT[:, :3, :3] = rotation
    RT[:, :3, -1] = translation
    return RT.reshape(-1, 16)

def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    return matrix[..., :2, :].clone().reshape(*matrix.size()[:-2], 6)


def depth2pts_outside(ray_o, ray_d, depth):
    '''
    ray_o, ray_d: [..., 3]
    depth: [...]; inverse of distance to sphere origin
    '''
    # note: d1 becomes negative if this mid point is behind camera
    d1 = -torch.sum(ray_d * ray_o, dim=-1) / torch.sum(ray_d * ray_d, dim=-1)
    p_mid = ray_o + d1.unsqueeze(-1) * ray_d
    p_mid_norm = torch.norm(p_mid, dim=-1)
    ray_d_cos = 1. / torch.norm(ray_d, dim=-1)
    d2 = torch.sqrt(1. - p_mid_norm * p_mid_norm) * ray_d_cos
    p_sphere = ray_o + (d1 + d2).unsqueeze(-1) * ray_d

    rot_axis = torch.cross(ray_o, p_sphere, dim=-1)
    rot_axis = rot_axis / torch.norm(rot_axis, dim=-1, keepdim=True)
    phi = torch.asin(p_mid_norm)
    theta = torch.asin(p_mid_norm * depth)  # depth is inside [0, 1]
    rot_angle = (phi - theta).unsqueeze(-1)     # [..., 1]

    # now rotate p_sphere
    # Rodrigues formula: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    p_sphere_new = p_sphere * torch.cos(rot_angle) + \
                torch.cross(rot_axis, p_sphere, dim=-1) * torch.sin(rot_angle) + \
                rot_axis * torch.sum(rot_axis*p_sphere, dim=-1, keepdim=True) * (1.-torch.cos(rot_angle))
    p_sphere_new = p_sphere_new / torch.norm(p_sphere_new, dim=-1, keepdim=True)
    pts = torch.cat((p_sphere_new, depth.unsqueeze(-1)), dim=-1)

    # now calculate conventional depth
    depth_real = 1. / (depth + TINY_NUMBER) * torch.cos(theta) * ray_d_cos + d1
    return pts, depth_real


def intersect_sphere(ray_o, ray_d, radius=1):
    '''
    ray_o, ray_d: [..., 3]
    compute the depth of the intersection point between this ray and unit sphere
    '''
    # note: d1 becomes negative if this mid point is behind camera
    d1 = -torch.sum(ray_d * ray_o, dim=-1) / torch.sum(ray_d * ray_d, dim=-1)
    p = ray_o + d1.unsqueeze(-1) * ray_d
    # consider the case where the ray does not intersect the sphere
    ray_d_cos = 1. / torch.norm(ray_d, dim=-1)
    d2 = radius ** 2 - torch.sum(p * p, dim=-1)
    mask = (d2 > 0)
    d2 = torch.sqrt(d2.clamp(min=1e-6)) * ray_d_cos
    d1, d2 = d1.unsqueeze(-1), d2.unsqueeze(-1)
    depth_range = [d1 - d2, d1 + d2]
    return depth_range, mask


def normalize(x, axis=-1, order=2):
    if isinstance(x, torch.Tensor):
        l2 = x.norm(p=order, dim=axis, keepdim=True)
        return x / (l2 + 1e-8), l2

    else:
        l2 = np.linalg.norm(x, order, axis)
        l2 = np.expand_dims(l2, axis)
        l2[l2==0] = 1
        return x / l2, l2


def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.
    Inputs:
        bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
        weights: (N_rays, N_samples_)
        N_importance: the number of samples to draw from the distribution
        det: deterministic or not
        eps: a small number to prevent division by zero
    Outputs:
        samples: the sampled samples
    Source: https://github.com/kwea123/nerf_pl/blob/master/models/rendering.py
    """
    N_rays, N_samples_ = weights.shape
    weights = weights + eps # prevent division by zero (don't do inplace op!)
    pdf = weights / torch.sum(weights, -1, keepdim=True) # (N_rays, N_samples_)
    cdf = torch.cumsum(pdf, -1) # (N_rays, N_samples), cumulative distribution function
    cdf = torch.cat([torch.zeros_like(cdf[: ,:1]), cdf], -1)  # (N_rays, N_samples_+1)
                                                               # padded to 0~1 inclusive

    if det:
        u = torch.linspace(0, 1, N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    else:
        u = torch.rand(N_rays, N_importance, device=bins.device)
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u)
    below = torch.clamp_min(inds-1, 0)
    above = torch.clamp_max(inds, N_samples_)

    inds_sampled = torch.stack([below, above], -1).view(N_rays, 2*N_importance)
    cdf_g = torch.gather(cdf, 1, inds_sampled)
    cdf_g = cdf_g.view(N_rays, N_importance, 2)
    bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

    denom = cdf_g[...,1]-cdf_g[...,0]
    denom[denom<eps] = 1 # denom equals 0 means a bin has weight 0, in which case it will not be sampled
                         # anyway, therefore any value for it is fine (set to 1 here)

    samples = bins_g[...,0] + (u-cdf_g[...,0])/denom * (bins_g[...,1]-bins_g[...,0])
    return samples


def normalization_inverse_sqrt_dist_centered(x_in_world, view_cell_center, max_depth):
    localized = x_in_world - view_cell_center
    local = torch.sqrt(torch.linalg.norm(localized, dim=-1))
    res = localized / (math.sqrt(max_depth) * local[..., None])
    return res


######################################################################################
