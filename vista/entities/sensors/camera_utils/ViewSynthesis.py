from typing import Dict, Optional, Tuple
from enum import Enum
import numpy as np
import pyrender

from . import CameraParams
from ....utils import transform


ZNEAR = 0.01
ZFAR = 10000


class DepthModes(Enum):
    FIXED_PLANE = 1
    INPUT_DISP = 2
    MONODEPTH = 3


class ViewSynthesis:
    def __init__(self, camera_param: CameraParams, config: Dict) -> None:
        # Parse configuration
        self._camera_param = camera_param
        self._config = config
        self._config['depth_mode'] = self._config.get('depth_mode', DepthModes.FIXED_PLANE)
        self._config['znear'] = self._config.get('znear', ZNEAR)
        self._config['zfar'] = self._config.get('zfar', ZFAR)
        self._config['use_lighting'] = self._config.get('use_lighting', True)
        self._config['ambient_light_factor'] = self._config.get('ambient_light_factor', 0.2)
        self._config['recoloring_factor'] = self._config.get('recoloring_factor', 0.5)

        # Projection and re-projection parameters
        self._K = self._camera_param.get_K()
        ### DEBUG
        self._K[0, 2] = self._camera_param.get_width() / 2.
        self._K[1, 2] = self._camera_param.get_height() / 2.
        ### DEBUG
        self._K_inv = np.linalg.inv(self._K)

        # Mesh coordinates, faces, and rays
        self._homo_coords, self._mesh_faces = self._get_homogeneous_image_coords(get_mesh=True)
        self._world_rays = np.matmul(self._K_inv, self._homo_coords)

        normal = np.reshape(self._camera_param.get_ground_plane()[:3], [1,3])
        d = self._camera_param.get_ground_plane()[3]
        k = np.divide(d, np.matmul(normal, self._world_rays))
        k[k < 0] = self._config['zfar']
        self._depth = k

        # Objects for rendering the scene
        self._renderer = pyrender.OffscreenRenderer(self._camera_param.get_width(),
                                                    self._camera_param.get_height())
        self._scene = pyrender.Scene(ambient_light=[1.,1.,1.],
                                     bg_color=[0,0,0])

        camera = pyrender.IntrinsicsCamera(fx=self._camera_param._fx,
                                           fy=self._camera_param._fy,
                                           cx=self._K[0,2],
                                           cy=self._K[1,2],
                                           znear=self._config['znear'],
                                           zfar=self._config['zfar'])
        self._camera_node = pyrender.Node(name='camera', 
                                          camera=camera)#, # DEBUG
                                        #   translation=self._camera_param.get_position()[:,0],
                                        #   rotation=self._camera_param.get_quaternion()[:,0])
        self._scene.add_node(self._camera_node)

        mesh = pyrender.Mesh([
            pyrender.Primitive(
                positions=self._world_rays.T,
                indices=self._mesh_faces.T,
                color_0=np.ones((self._world_rays.shape[1], 4)),
                mode=pyrender.constants.GLTF.TRIANGLES,
            )
        ])
        self._bg_node = pyrender.Node(name='bg', # project camera view to 3D
                                      mesh=mesh)#, # DEBUG
                                    #   translation=self._camera_param.get_position()[:,0],
                                    #   rotation=self._camera_param.get_quaternion()[:,0])
        self._scene.add_node(self._bg_node)

    def synthesize(self, trans: np.ndarray, rot: np.ndarray, img: np.ndarray, 
                   depth: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        # Update mesh vertex positions
        if self._config['depth_mode'] == DepthModes.FIXED_PLANE:
            assert depth is None
            depth = self._depth
            
            depth = depth.reshape((1, -1))
            world_coords = np.multiply(-depth, self._world_rays)
            self._bg_node.mesh.primitives[0].positions = world_coords.T
        else:
            raise NotImplementedError

        # Update mesh face colors
        colors = img[:,::-1] / 255.
        self._bg_node.mesh.primitives[0].color_0[:,:3] = colors.reshape((-1, 3))

        # Update camera pose based on the requested viewpoint
        self._camera_node.matrix = transform.vec2mat(trans, rot)

        # Render background
        color_bg, depth_bg = self._renderer.render(self._scene, \
            flags=pyrender.constants.RenderFlags.FLAT)

        # TODO: DEBUG
        color, depth = color_bg, depth_bg
        import cv2
        cv2.imwrite('test.png', color)
        import pdb; pdb.set_trace()

        return color, depth        
    
    def _get_homogeneous_image_coords(self, get_mesh=False):
        cam_w = self._camera_param.get_width()
        cam_h = self._camera_param.get_height()

        xx, yy = np.meshgrid(np.arange(cam_w), np.arange(cam_h))
        coords = np.stack((xx.reshape(-1), yy.reshape(-1), np.ones_like(xx).reshape(-1)), axis=0)

        if not get_mesh:
            return coords
        else:
            upper = np.array([[0, 0], [0, 1], [1, 1]])
            lower = np.array([[0, 0], [1, 1], [1, 0]])
            mesh_tri = []

            # FIXME TODO: vectorize this double for-loop
            for i in range(0, cam_h - 1):
                for j in range(0, cam_w - 1):
                    c = np.array([i, j])
                    mesh_tri.append(
                        np.ravel_multi_index((c + upper).T, (cam_h, cam_w)))
                    mesh_tri.append(
                        np.ravel_multi_index((c + lower).T, (cam_h, cam_w)))
            mesh_tri = np.stack(mesh_tri, axis=1)

            return coords, mesh_tri
