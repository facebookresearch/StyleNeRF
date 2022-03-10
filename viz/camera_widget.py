# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


import imgui
import dnnlib
from gui_utils import imgui_utils, imgui_window


class CameraWidget:
    def __init__(self, viz):
        self.viz = viz
        self.camera_kwargs = dnnlib.EasyDict(yaw=0, pitch=0, fov=12, anim=False, speed=0.25)
        self.camera_mode = False
        self.output_nerf = False
    
    def set_camera(self, dv, du):
        viz = self.viz
        du, dv = -du / viz.font_size * 5e-2, -dv / viz.font_size * 5e-2
        if ((self.camera_kwargs.yaw + du) <= 1 and (self.camera_kwargs.yaw + du) >= -1 and
            (self.camera_kwargs.pitch + dv) <= 1 and (self.camera_kwargs.pitch + dv) >=-1):
            self.camera_kwargs.yaw   += du
            self.camera_kwargs.pitch += dv

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        if show:
            imgui.text('Camera')
            imgui.same_line(viz.label_w)

            _clicked, self.camera_mode = imgui.checkbox('Control viewpoint##enable', self.camera_mode)
            imgui.same_line()
            _clicked, self.output_nerf = imgui.checkbox('NeRF output##enable', self.output_nerf)

        viz.args.camera = (self.camera_kwargs.yaw, self.camera_kwargs.pitch, self.camera_kwargs.fov)
        viz.args.output_lowres = self.output_nerf