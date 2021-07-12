from typing import Dict, Optional, Tuple, List
from enum import Enum
import numpy as np
import pyrender

from . import CameraParams
from ....utils import transform, logging


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

        # Renderer and scene
        self._renderer = pyrender.OffscreenRenderer(self._camera_param.get_width(),
                                                    self._camera_param.get_height())
        self._scene = pyrender.Scene(ambient_light=[1.,1.,1.],
                                     bg_color=[0,0,0])

        # Camera for rendering
        camera = pyrender.IntrinsicsCamera(fx=self._camera_param._fx,
                                           fy=self._camera_param._fy,
                                           cx=self._camera_param._cx,
                                           cy=self._camera_param._cy,
                                           znear=self._config['znear'],
                                           zfar=self._config['zfar'])
        self._camera_node = pyrender.Node(name='camera', 
                                          camera=camera,
                                          translation=self._camera_param.get_position()[:,0],
                                          rotation=self._camera_param.get_quaternion()[:,0])
        self._scene.add_node(self._camera_node)

        # Mesh of background. Can add more by calling add_bg_mesh for different camera_param
        self._world_rays: Dict[str, np.ndarray] = dict()
        if self._config['depth_mode'] == DepthModes.FIXED_PLANE:
            self._depth: Dict[str, np.ndarray] = dict()
        self._bg_node: Dict[str, pyrender.Node] = dict()
        self.add_bg_mesh(self._camera_param)

    def synthesize(self, trans: np.ndarray, rot: np.ndarray, imgs: Dict[str, np.ndarray], 
                   depth: Optional[Dict[str, np.ndarray]] = None) -> Tuple[np.ndarray, np.ndarray]:
        for name, img in imgs.items():
            # Refresh meshes in renderer; otherwise mesh vertex/color won't update
            mesh = self._bg_node[name].mesh
            for prim in mesh.primitives:
                prim._unbind()
                prim._remove_from_context()
    
            if mesh in self._renderer._renderer._meshes:
                self._renderer._renderer._meshes.remove(mesh)

            # Update mesh vertex positions (this won't affect node-level translation and rotation)
            if self._config['depth_mode'] == DepthModes.FIXED_PLANE:
                depth = self._depth[name]
            else:
                raise NotImplementedError

            depth = depth.reshape((1, -1))
            world_coords = np.multiply(-depth, self._world_rays[name])
            mesh.primitives[0].positions = world_coords.T

            # Update mesh face colors
            colors = img[:,::-1] / 255.
            mesh.primitives[0].color_0[:,:3] = colors.reshape((-1, 3))

        # Update camera pose based on the requested viewpoint (with camera
        # relative pose to the car as offset)
        trans += self._camera_param.get_position()[:,0]
        rot += transform.quat2euler(self._camera_param.get_quaternion()[:,0])
        self._camera_node.matrix = transform.vec2mat(trans, rot)

        # Render background
        color_bg, depth_bg = self._renderer.render(self._scene, \
            flags=pyrender.constants.RenderFlags.FLAT)

        ### DEBUG
        logging.warning('Only using background image for rendering')
        color, depth = color_bg, depth_bg
        # import cv2
        # cv2.imwrite('test.png', imgs['camera_front'])
        # cv2.imwrite('test2.png', color)
        # import pdb; pdb.set_trace()
        ### DEBUG

        return color, depth
    
    def add_bg_mesh(self, camera_param: CameraParams) -> None:
        # Projection and re-projection parameters
        K = camera_param.get_K().copy()
        K_inv = np.linalg.inv(K)

        # Mesh coordinates, faces, and rays
        name = camera_param.name
        homo_coords, mesh_faces = self._get_homogeneous_image_coords(get_mesh=True)
        self._world_rays[name] = np.matmul(K_inv, homo_coords)

        # Get depth for ground plane assumption
        if self._config['depth_mode'] == DepthModes.FIXED_PLANE:
            normal = np.reshape(camera_param.get_ground_plane()[:3], [1,3])
            d = camera_param.get_ground_plane()[3]
            k = np.divide(d, np.matmul(normal, self._world_rays[name]))
            if camera_param == self._camera_param: 
                # NOTE: hacky way to make image from the main camera have fronter order
                logging.debug('Hacky way to make main camera image have fronter order')
                k[k < 0] = self._config['zfar'] / 10.1
            else:
                k[k < 0] = self._config['zfar'] / 10. # should be smaller than actual zfar
            self._depth[camera_param.name] = k

        # Add mesh to scene (fix node level translation and rotation)
        mesh = pyrender.Mesh([
            pyrender.Primitive(
                positions=self._world_rays[name].T,
                indices=mesh_faces.T,
                color_0=np.ones((self._world_rays[name].shape[1], 4)),
                mode=pyrender.constants.GLTF.TRIANGLES,
            )
        ])
        self._bg_node[name] = pyrender.Node(name='bg_{}'.format(name), # project camera view to 3D
                                            mesh=mesh,
                                            translation=camera_param.get_position()[:,0],
                                            rotation=camera_param.get_quaternion()[:,0])
        self._scene.add_node(self._bg_node[name])
    
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
            logging.debug('Homogeneous coordinate computation not vectorized')
            for i in range(0, cam_h - 1):
                for j in range(0, cam_w - 1):
                    c = np.array([i, j])
                    mesh_tri.append(
                        np.ravel_multi_index((c + upper).T, (cam_h, cam_w)))
                    mesh_tri.append(
                        np.ravel_multi_index((c + lower).T, (cam_h, cam_w)))
            mesh_tri = np.stack(mesh_tri, axis=1)

            return coords, mesh_tri

    @property
    def bg_mesh_names(self) -> List[str]:
        return self._bg_node.keys()
