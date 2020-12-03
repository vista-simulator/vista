import copy
import os
import sys
import time
import timeit
from enum import Enum

import cv2
import numpy as np
import pyrender
import scipy.interpolate
import tensorflow as tf
import trimesh

from tensorflow.python.client import device_lib

from . import Camera

MAX_DIST = 10000.
HAS_GPU = len([x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU']) > 0

class DepthModes(Enum):
    FIXED_PLANE = 1
    INPUT_DISP = 2
    MONODEPTH = 3

class ViewSynthesis:
    """Object to create rotated frames.
    This object is initialized in Trace.py, then called in step(). The step
    function calls .process(R,trans), which handles the frame manipulation based
    on the input rotation and translation matrices. show_plots() is called within
    process().

    Args:
        camera (obj): The camera object to use.

    Attributes:
        cpu_coords (arr): Matrix of x, y, z coordinates held in CPU
        cpu_coords_transposed (arr): Transposed CPU coords to save time in process()
        sess (TF): The current TensorFlow session
        camera (obj): The camera object to use
        _K (arr): The intrinsic camera matrix
        _K_inv (arr): The inverse of the intrinsic camera matrix
        inds (arr): Indices to extract from the data to speed up computation time
        new_image_coords (arr): Contains the real-world location of pixel data
        depth (arr): Array holding real-world depth of each pixel; initialized as all 0s
     """
    def __init__(self,
                camera, # Camera object of the images
                sess=None, # tf.Session object if already defined
                baseline = 0.42567, # [m]
                lookahead_distance=20.0, # lookahead distance for recovery
                mode=DepthModes.FIXED_PLANE,
        ):
        self.camera = camera
        self.dims = (self.camera.get_height(), self.camera.get_width())
        self.baseline = baseline
        self.lookahead_distance = lookahead_distance
        self.mode = mode
        if sess == None:
            self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True))
        else:
            self.sess = sess

        self.homogeneous_coords, self.mesh_faces = \
            self._get_homogeneous_image_coords(camera, get_mesh=True)
        self.K = camera.get_K()
        self.K[0,2] = camera.get_width()/2.
        self.K[1,2] = camera.get_height()/2.
        self.K_inv = np.linalg.inv(self.K) # camera.get_K_inv()
        self.world_rays = np.matmul(self.K_inv, self.homogeneous_coords)

        self.scene = pyrender.Scene(ambient_light=[1.,1.,1.], bg_color=[0,0,0])
        self.render_camera = pyrender.IntrinsicsCamera(
            fx=camera._fx, fy=camera._fy, cx=camera._cx, cy=camera._cy, znear=0.0001, zfar=1000000)
        self.renderer = pyrender.OffscreenRenderer(camera.get_width(), camera.get_height())

        self.mesh = pyrender.Mesh([pyrender.Primitive(
            positions=self.world_rays.T,
            indices=self.mesh_faces.T,
            color_0=np.ones((self.world_rays.shape[1], 4))
        )])

        self.scene.add(self.mesh, name="env")
        self.scene.add(self.render_camera, name="camera")

        print("done creating mesh")
        # import pdb; pdb.set_trace()

        normal = np.reshape(self.camera.get_ground_plane()[0:3], [1,3])
        d = self.camera.get_ground_plane()[3]
        k = np.divide(d, np.matmul(normal, self.world_rays))
        k[k<0] = MAX_DIST
        self.depth = k

    def disp_to_depth(self, disparity):
        depth_img = np.exp(0.5*np.clip(self.baseline * self.K[0,0] / (disparity*self.dims[1]), 0, MAX_DIST))
        return depth_img

    def get_as_py_func(self, theta, translation_x, translation_y, image, depth=None):
        if depth is None:
            depth = self.depth

        import time

        depth = depth.reshape([1,-1])
        world_coords = np.multiply(-depth, self.world_rays)

        self.mesh.primitives[0].positions = world_coords.T
        self.mesh.primitives[0].color_0[:, :3] = image.reshape(-1,3) / 255.

        # color_0 = np.ones((self.world_rays.shape[1], 4))
        # color_0[:,:3] = image.reshape(-1,3) / 255.
        # new_mesh = pyrender.Mesh([pyrender.Primitive(
        #     positions=world_coords.T,
        #     indices=self.mesh_faces.T,
        #     color_0=color_0)])

        camera_pose = np.eye(4)
        camera_pose[:3, :3] = self._create_rotation_matrix(theta)
        camera_pose[:3, 3] = [translation_x, 0, translation_y]

        # [self.scene.remove_node(_n) for _n in self.scene.get_nodes(name="env")]
        self.scene.clear()

        self.scene.add(copy.deepcopy(self.mesh), name="env")

        # [self.scene.set_pose(_n, camera_pose) for _n in self.scene.get_nodes(name="camera")]
        self.scene.add(self.render_camera, pose=camera_pose)
        # print("tic4", time.time()-tic4)

        color, depth = self.renderer.render(self.scene, flags=pyrender.constants.RenderFlags.FLAT)

        return color, depth


    def _get_homogeneous_image_coords(self, camera, get_mesh=False):
        cam_w = camera.get_width()
        cam_h = camera.get_height()

        xx, yy = np.meshgrid( np.arange(cam_w), np.arange(cam_h) )
        coords = np.stack(
            (xx.reshape(-1), yy.reshape(-1), np.ones_like(xx).reshape(-1)),
            axis=0)

        if not get_mesh:
            return coords

        else:
            upper = np.array([[0,0],[0,1],[1,1]])
            lower = np.array([[0,0],[1,1],[1,0]])
            mesh_tri = []
            tic = time.time()
            # FIXME TODO: vectorize this double for-loop
            for i in range(0, cam_h-1):
                for j in range(0, cam_w-1):
                    c = np.array([i,j])
                    mesh_tri.append(np.ravel_multi_index((c+upper).T, (cam_h,cam_w)))
                    mesh_tri.append(np.ravel_multi_index((c+lower).T, (cam_h,cam_w)))
            mesh_tri = np.stack(mesh_tri, axis=1)
            print (time.time()-tic)
            return coords, mesh_tri

    def _create_rotation_matrix(self, theta):
        s, c = np.sin(theta), np.cos(theta)
        R = np.array([
            [  c,  0,  s ],
            [  0,  1,  0 ],
            [ -s,  0,  c ]
        ])
        return R



if __name__ == "main":

    camera = Camera("camera_front")
    camera.resize(250,400)

    vs = ViewSynthesis(camera)

    img = np.random.rand(250,400,3)
    disp = np.ones((250,400))*0.1
    depth = vs.disp_to_depth(disp)
    toc = []
    for i in np.linspace(-1,1,1000):
        tic = time.time()
        color, depth = vs.view_synthesizer(0, i, 0, img, depth=vs.depth)
        toc.append(time.time()-tic)
        print(toc[-1])
        # cv2.imshow('hi',cv2.normalize(np.log10(depth+1e-2), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))
        cv2.imshow('hi',color)
        cv2.waitKey(1)

    print("AVG: ",np.mean(toc))
