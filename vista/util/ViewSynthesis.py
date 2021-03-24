import copy
import os
import sys
import time
import timeit
from enum import Enum

import cv2
import numpy as np
import pyrender
import trimesh

from . import Camera

MAX_DIST = 10000.
USE_LIGHTING = True
LIGHTING_DR = False


class DepthModes(Enum):
    FIXED_PLANE = 1
    INPUT_DISP = 2
    MONODEPTH = 3


class ViewSynthesis:
    """Object to synthesize new image frames given a new desired viewpoint and
    orientation.

    Args:
        camera (obj): The camera object to use.
        baseline (float): Metric distance for converting between disparity
            and depth
        mode (DepthModes): The mode of depth to use for projecting from 2d -> 3d
     """

    def __init__(
            self,
            camera,  # Camera object of the images
            baseline=0.42567,  # [m]
            mode=DepthModes.FIXED_PLANE):


        self.camera = camera
        self.dims = (self.camera.get_height(), self.camera.get_width())
        self.baseline = baseline
        self.mode = mode

        # Projection and re-projection parameters
        self.K = camera.get_K()
        self.K[0, 2] = camera.get_width() / 2.
        self.K[1, 2] = camera.get_height() / 2.
        self.K_inv = np.linalg.inv(self.K)  # camera.get_K_inv()

        # Mesh coordinates, faces, and rays
        self.homogeneous_coords, self.mesh_faces = \
            self._get_homogeneous_image_coords(camera, get_mesh=True)
        self.world_rays = np.matmul(self.K_inv, self.homogeneous_coords)

        # Objects for rendering the scene
        self.scene = pyrender.Scene(
            ambient_light=[1., 1., 1.], bg_color=[0, 0, 0])
        self.render_camera = pyrender.IntrinsicsCamera(
            fx=camera._fx,
            fy=camera._fy,
            cx=camera._cx,
            cy=camera._cy,
            znear=0.01,
            zfar=100000)
        self.renderer = pyrender.OffscreenRenderer(camera.get_width(),
                                                   camera.get_height())

        # Define a base mesh for the surroundings based on the camera
        self.mesh = pyrender.Mesh([
            pyrender.Primitive(
                positions=self.world_rays.T,
                indices=self.mesh_faces.T,
                color_0=np.ones((self.world_rays.shape[1], 4)),
                mode=pyrender.constants.GLTF.TRIANGLES,
            )
        ])

        # Add the mesh and camera to the scene
        # (these will be updated during simulation)
        self.scene.add(self.mesh, name="env")
        self.scene.add(self.render_camera, name="camera")

        # Use ground plane assumption to get a rough estimate of depth
        # based on a fixed plane and infinite depth above horizon.
        normal = np.reshape(self.camera.get_ground_plane()[0:3], [1, 3])
        d = self.camera.get_ground_plane()[3]
        k = np.divide(d, np.matmul(normal, self.world_rays))
        k[k < 0] = MAX_DIST
        self.depth = k

    def disp_to_depth(self, disparity):
        depth_img = np.exp(
            0.5 * np.clip(self.baseline * self.K[0, 0] /
                          (disparity * self.dims[1]), 0, MAX_DIST))
        return depth_img

    def synthesize(self,
                   theta,
                   translation_x,
                   translation_y,
                   image,
                   depth=None,
                   other_agents=[]):
        if depth is None:
            depth = self.depth

        # Update mesh vertex positions
        depth = depth.reshape([1, -1])
        world_coords = np.multiply(-depth, self.world_rays)
        self.mesh.primitives[0].positions = world_coords.T

        # Update mesh face colors
        colors = image[:, ::-1] / 255.
        self.mesh.primitives[0].color_0[:, :3] = colors.reshape(-1, 3)

        # Compute new camera pose based on the requested viewpoint args
        camera_pose = np.eye(4)
        cam_theta, cam_x, cam_y = self._to_ogl_coordinate(theta, translation_x, translation_y)
        camera_pose[:3, :3] = self._create_rotation_matrix(cam_theta)
        camera_pose[:3, 3] = [cam_x, 0, cam_y]

        # Clear scene and fill with new contents
        self.scene.clear()
        self.scene.add(copy.deepcopy(self.mesh), name="env")
        self.scene.add(self.render_camera, pose=camera_pose)

        if USE_LIGHTING:
            # Render background
            self.scene.ambient_light = [1., 1., 1.] # doesn't matter for FLAT rendering
            color_bg, depth_bg = self.renderer.render(
                self.scene, flags=pyrender.constants.RenderFlags.FLAT)

            # remove background
            env_node = [n for n in list(self.scene.nodes) if n.name == 'env'][0]
            self.scene.remove_node(env_node)

            # Add other agents to the scene
            if LIGHTING_DR:
                self.scene.ambient_light = [np.random.uniform(0.05, 0.3)] * 3
            else:
                self.scene.ambient_light = [.1, .1, .1]
            for other_agent in other_agents:
                self.scene.add_node(other_agent)

            # Add light
            if LIGHTING_DR:
                light_intensity = np.random.uniform(5, 15) # domain randomization
            else:
                light_intensity = 10
            light = pyrender.DirectionalLight([255, 255, 255], light_intensity)
            self.scene.add(light)

            # Render car
            color_agent, depth_agent = self.renderer.render(self.scene)

            # Overlay
            mask = np.any(color_agent != 0, axis=2, keepdims=True).astype(np.uint8)

            recoloring_factor = np.random.uniform(0.2, 0.7) # domain randomization
            color_agent_mean = (color_agent * mask).sum(0).sum(0) / mask.sum()
            color_bg_mean = color_bg.mean(0).mean(0)
            recolor_agent = color_agent + (color_bg_mean - color_agent_mean) * recoloring_factor
            recolor_agent = np.clip(recolor_agent, 0, 255)

            color = (1 - mask) * color_bg + mask * recolor_agent
        else:
            # Add other agents to the scene
            for other_agent in other_agents:
                self.scene.add_node(other_agent)

            # Render
            color, depth = self.renderer.render(
                self.scene, flags=pyrender.constants.RenderFlags.FLAT)

        return color, depth

    def _get_homogeneous_image_coords(self, camera, get_mesh=False):
        cam_w = camera.get_width()
        cam_h = camera.get_height()

        xx, yy = np.meshgrid(np.arange(cam_w), np.arange(cam_h))
        coords = np.stack(
            (xx.reshape(-1), yy.reshape(-1), np.ones_like(xx).reshape(-1)),
            axis=0)

        if not get_mesh:
            return coords

        else:
            upper = np.array([[0, 0], [0, 1], [1, 1]])
            lower = np.array([[0, 0], [1, 1], [1, 0]])
            mesh_tri = []
            tic = time.time()
            # FIXME TODO: vectorize this double for-loop
            for i in range(0, cam_h - 1):
                for j in range(0, cam_w - 1):
                    c = np.array([i, j])
                    mesh_tri.append(
                        np.ravel_multi_index((c + upper).T, (cam_h, cam_w)))
                    mesh_tri.append(
                        np.ravel_multi_index((c + lower).T, (cam_h, cam_w)))
            mesh_tri = np.stack(mesh_tri, axis=1)
            print(time.time() - tic)
            return coords, mesh_tri

    def _create_rotation_matrix(self, theta):
        s, c = np.sin(theta), np.cos(theta)
        R = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
        return R

    def _to_ogl_coordinate(self, theta, x, y):
        # OpenGL z-axis inverted, theta (x-z-plane) and translation_y (z)
        return -theta, x, -y


if __name__ == "main":

    camera = Camera("camera_front")
    camera.resize(250, 400)

    vs = ViewSynthesis(camera)

    img = np.random.rand(250, 400, 3)
    disp = np.ones((250, 400)) * 0.1
    depth = vs.disp_to_depth(disp)
    toc = []
    for i in np.linspace(-1, 1, 1000):
        tic = time.time()
        color, depth = vs.view_synthesizer(0, i, 0, img, depth=vs.depth)
        toc.append(time.time() - tic)
        print(toc[-1])
        # cv2.imshow('hi',cv2.normalize(np.log10(depth+1e-2), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))
        cv2.imshow('hi', color)
        cv2.waitKey(1)

    print("AVG: ", np.mean(toc))
