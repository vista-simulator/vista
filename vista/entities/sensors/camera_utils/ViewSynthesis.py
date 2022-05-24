import os
from typing import Dict, Optional, Tuple, List
from enum import Enum
import numpy as np
import pyrender
import copy

from . import CameraParams
from ....utils import transform, logging, misc

ZNEAR = 0.01
ZFAR = 10000


class DepthModes(Enum):
    FIXED_PLANE = 1
    INPUT_DISP = 2
    MONODEPTH = 3


class ViewSynthesis:
    """ A RGB synthesizer that simulates RGB image at novel viewpoint around a pre-collected
    RGB image/video dataset. Conceptually, it (1) projects a reference 2D RGB image to 3D colored
    mesh using camera projection matrices and (approximated) depth, (2) place virtual objects
    in the scene, (3) render RGB image at novel viewpoint.

    Args:
        camera_param (CameraParams): Camera parameter object of the virtual camera.
        config (Dict): Configuration of the synthesizer.
        init_with_bg_mesh (bool): Whether to initialize with background mesh; default
            is set to ``True``.

    """
    DEFAULT_CONFIG = {
        'depth_mode': DepthModes.FIXED_PLANE,
        'znear': ZNEAR,
        'zfar': ZFAR,
        'use_lighting': False,
        'recoloring_factor': 0.5,
    }

    def __init__(self,
                 camera_param: CameraParams,
                 config: Dict,
                 init_with_bg_mesh: Optional[bool] = True) -> None:
        # Parse configuration
        self._camera_param = camera_param
        config = misc.merge_dict(config, self.DEFAULT_CONFIG)
        self._config = config
        if isinstance(self._config['depth_mode'], str):
            self._config['depth_mode'] = getattr(DepthModes,
                                                 self._config['depth_mode'])

        # Renderer and scene
        self._renderer = pyrender.OffscreenRenderer(
            self._camera_param.get_width(), self._camera_param.get_height())
        self._scene = pyrender.Scene(ambient_light=[1., 1., 1.],
                                     bg_color=[0, 0, 0])

        # Camera for rendering
        cam_w = camera_param.get_width()
        cam_h = camera_param.get_height()
        camera = pyrender.IntrinsicsCamera(
            fx=self._camera_param._fx,
            fy=self._camera_param._fy,
            cx=cam_w / 2.,  # NOTE: assume calibrated camera
            cy=cam_h / 2.,
            znear=self._config['znear'],
            zfar=self._config['zfar'])
        self._camera_node = pyrender.Node(
            name='camera',
            camera=camera,
            translation=self._camera_param.get_position()[:, 0],
            rotation=self._camera_param.get_quaternion()[:, 0])
        self._scene.add_node(self._camera_node)

        # Mesh of background. Can add more by calling add_bg_mesh for different camera_param
        self._world_rays: Dict[str, np.ndarray] = dict()
        if self._config['depth_mode'] == DepthModes.FIXED_PLANE:
            self._depth: Dict[str, np.ndarray] = dict()
        self._bg_node: Dict[str, pyrender.Node] = dict()
        if init_with_bg_mesh:
            self.add_bg_mesh(self._camera_param)

        # Scene for non-background objects and object nodes
        self._object_scene = pyrender.Scene(ambient_light=[1., 1., 1.],
                                            bg_color=[0, 0, 0])
        self._object_nodes = dict()

    def synthesize(
        self,
        trans: np.ndarray,
        rot: np.ndarray,
        imgs: Dict[str, np.ndarray],
        depth: Optional[Dict[str, np.ndarray]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """ Synthesize RGB image at the novel viewpoint specified by ``trans`` and
        ``rot`` with respect to the nominal viewpoint that corresponds to a set of
        RGB images and depth maps.

        Args:
            trans (np.ndarray): Translation vector.
            rot (np.ndarray): Rotation vector in Euler angle.
            imgs (Dict[str, np.ndarray]): A set of images (potentially from multiple camera).
            depth (Dict[str, np.ndarray]): A set of depth maps corresponding to ``imgs``.

        Returns:
            Returns a tuple (``array_1``, ``array_2``), where ``array_1`` is the
            synthesized RGB image and ``array_2`` is the corresponding depth image.

        """

        for name, img in imgs.items():
            # Need this otherwise will cause memory leak
            if os.environ.get('PYOPENGL_PLATFORM', None) != 'egl':
                self._bg_node[name].mesh = copy.deepcopy(
                    self._bg_node[name].mesh)

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
            colors = img[:, ::-1] / 255.
            mesh.primitives[0].color_0[:, :3] = colors.reshape((-1, 3))

        # Update camera pose based on the requested viewpoint (with camera
        # relative pose to the car as offset)
        trans += self._camera_param.get_position()[:, 0]
        rot += transform.quat2euler(self._camera_param.get_quaternion()[:, 0])
        self._camera_node.matrix = transform.vec2mat(trans, rot)

        # Render background
        color_bg, _ = self._renderer.render(
            self._scene, flags=pyrender.constants.RenderFlags.FLAT)

        # Render scene object
        self._object_scene.clear()

        light = pyrender.DirectionalLight(
            [255, 255, 255], self.config['directional_light_intensity'])
        self._object_scene.add(light)

        self._object_scene.add_node(self._camera_node)

        for name, object_node in self._object_nodes.items():
            self._object_scene.add_node(object_node)

        color_objects, depth_objects = self._renderer.render(
            self._object_scene)
        color_objects = color_objects[:, :, ::
                                      -1]  # TODO: not sure why we need this

        # Overlay
        mask = (depth_objects != 0.).astype(np.uint8)[:, :, None]

        if mask.sum() != 0:  # recoloring
            color_bg_mean = color_bg.mean(0).mean(0)
            color_objects_mean = (color_objects *
                                  mask).sum(0).sum(0) / mask.sum()
            color_objects = color_objects + (
                color_bg_mean -
                color_objects_mean) * self.config['recoloring_factor']
            color_objects = np.clip(color_objects, 0, 255).astype(np.uint8)
        else:  # agent out-of-view
            pass

        color = (1 - mask) * color_bg + mask * color_objects

        return color, depth

    def update_object_node(self, name: str, mesh: pyrender.Mesh,
                           trans: np.ndarray, quat: np.ndarray) -> None:
        """ Update the virtual object in the scene.

        Args:
            name (str): Name of the virtual object.
            mesh (pyrender.Mesh): Mesh of the virtual object.
            trans (np.ndarray): Translation vector.
            quat (np.ndarray): Quaternion vector.

        """
        self._object_nodes[name] = pyrender.Node(name=name,
                                                 mesh=mesh,
                                                 translation=trans,
                                                 rotation=quat)

    def add_bg_mesh(self, camera_param: CameraParams) -> None:
        """ Add background mesh to the scene based on camera projection and the
        initial depth. The color of the mesh will be updated at every ``synthesize``
        call and if not using ground-plane depth approximation, the geometry of
        the mesh will be also updated.

        Args:
            camera_param (CameraParams): Camera parameter of the virtual camera.

        """
        # Projection and re-projection parameters
        K = camera_param.get_K().copy()
        logging.debug('Hacky way to handle unrectified image')
        cam_w = camera_param.get_width()
        cam_h = camera_param.get_height()
        K[0, 2] = cam_w / 2.
        K[1, 2] = cam_h / 2.
        K_inv = np.linalg.inv(K)

        # Mesh coordinates, faces, and rays
        name = camera_param.name
        homo_coords, mesh_faces = self._get_homogeneous_image_coords(
            camera_param, get_mesh=True)
        self._world_rays[name] = np.matmul(K_inv, homo_coords)

        # Get depth for ground plane assumption
        if self._config['depth_mode'] == DepthModes.FIXED_PLANE:
            normal = np.reshape(camera_param.get_ground_plane()[:3], [1, 3])
            d = camera_param.get_ground_plane()[3]
            k = np.divide(d, np.matmul(normal, self._world_rays[name]))
            k[k < 0] = self._config[
                'zfar'] / 10.  # should be smaller than actual zfar
            if camera_param == self._camera_param:
                # NOTE: hacky way to make image from the main camera have fronter order
                logging.debug(
                    'Hacky way to make main camera image have fronter order')
                k = k / 1.1
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
        self._bg_node[name] = pyrender.Node(
            name='bg_{}'.format(name),  # project camera view to 3D
            mesh=mesh,
            translation=camera_param.get_position()[:, 0],
            rotation=camera_param.get_quaternion()[:, 0])
        self._scene.add_node(self._bg_node[name])

    def _get_homogeneous_image_coords(self, camera_param, get_mesh=False):
        cam_w = camera_param.get_width()
        cam_h = camera_param.get_height()

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
        """ Names of all background meshes in the scene. """
        return self._bg_node.keys()

    @property
    def object_nodes(self) -> List[pyrender.Node]:
        """ Pyrender nodes of all virtual objects added to the scene. """
        return self._object_nodes

    @property
    def config(self) -> Dict:
        """ Configuration of the view synthesizer. """
        return self._config
