from typing import List, Optional
import os
import copy
import numpy as np
from PIL import Image
import trimesh
import pyrender
from vista.utils import transform


class MeshLib(object):
    """ Handle all meshes of actors (as opposed to scene/background itself) in the scene.
    It basically reads in all meshes with .obj extension, calibrates the meshes such that
    they are centered at the origin, and convert them to ``pyrender.Mesh`` for later usage.
    Note that this class is written specifically for a set of meshes (`carpack01`) and thus
    if there is a custom set of meshes, you would need to change how to read from the root
    directory and to calibrate the mesh.

    Args:
        root_dirs (List[str]): A list of root directories that contains meshes.

    Raises:
        AttributeError: If there is no mesh in the directory.

    In the source code, there is a ``main`` function that uses this class independently that
    performs rendering on meshes in pyrender.

    """

    FORMATS = ['.obj']

    def __init__(self, root_dirs: List[str]):
        root_dirs = [root_dirs
                     ] if not isinstance(root_dirs, list) else root_dirs
        fpaths = []
        for root_dir in root_dirs:
            root_dir = os.path.abspath(os.path.expanduser(root_dir))
            for sdir in os.listdir(root_dir):
                sdir = os.path.join(root_dir, sdir)
                if not os.path.isdir(sdir):
                    continue
                fpath = os.listdir(sdir)
                for fm in self.FORMATS:  # elements in the front has higher priority
                    fp = [
                        _fp for _fp in fpath if os.path.splitext(_fp)[-1] == fm
                    ]
                    if len(fp) == 1:
                        fpath = fp[0]
                        break
                if isinstance(fpath, list):
                    print(f'Cannot find mesh with supported format in {sdir}')
                    continue
                fpath = os.path.join(sdir, fpath)
                fpaths.append(fpath)

        tmeshes = dict()
        for i, fpath in enumerate(fpaths):
            try:
                tm = trimesh.load(fpath)
                tm = list(
                    tm.geometry.values())  # convert from scene to trimesh
                tm, mesh_dim = self._calibrate_tm(tm)
                source = os.path.basename(
                    os.path.dirname(os.path.dirname(fpath)))
                body_images = dict()
                for color in [
                        'Black', 'Blue', 'Green', 'Red', 'White', 'Yellow',
                        'Grey'
                ]:
                    img_path = os.path.splitext(fpath)[0] + color + '.png'
                    if os.path.exists(img_path):
                        img = Image.open(img_path)
                        body_images[color] = img
                extra = {'body_images': body_images}
                tmeshes[i] = {
                    'fpath': fpath,
                    'tmesh': tm,
                    'mesh_dim': mesh_dim,
                    'source': source,
                    'extra': extra,
                }
            except:
                print(f'Failed to load/process mesh {fpath}')
        if len(tmeshes) == 0:
            raise AttributeError("No mesh in the directory")

        self._fpaths = fpaths
        self._tmeshes = tmeshes
        self._agents_meshes = []
        self._agents_meshes_dim = []
        self._mesh_node = pyrender.Node()

    def reset(self, n_agents: int, random: Optional[bool] = True) -> None:
        """ Reset agents meshes by sampling ``n_agents`` meshes from the mesh library.

        Args:
            n_agents (int): Number of agents.
            random (bool): Whether to randomly sample ``n_agents`` meshes from
                the entire mesh library; default is set to ``True``.

        """
        idcs = np.random.choice(
            self.n_tmeshes, n_agents) if random else np.arange(self.n_tmeshes)

        self._agents_meshes = []
        self._agents_meshes_dim = []
        for idx in idcs:
            mesh = self._tmesh2mesh(self._tmeshes[idx])
            self._agents_meshes.append(mesh)
            self._agents_meshes_dim.append(self._tmeshes[idx]['mesh_dim'])

    def _tmesh2mesh(self, tm):
        # NOTE:cannot make a copy to keep the original tmesh intact otherwise will cause memory leak
        tm_list = tm['tmesh']

        # randomization in tmesh object
        # wheel, glass, optic, body = range(len(tm_list))
        body = np.argmax([v.triangles.shape[0] for v in tm_list
                          ])  # NOTE: hacky way to get body mesh
        color = np.random.choice(list(tm['extra']['body_images'].keys()))

        tm_list[body].visual.material.image = tm['extra']['body_images'][color]
        Ns = np.random.randint(10, 300)  # specular highlight (0-1000)
        tm_list[body].visual.material.kwargs.update(Ns=Ns)

        # convert mesh color from rgb to bgr
        for i in range(len(tm_list)):
            img = tm_list[i].visual.material.image
            if img is not None:
                tm_list[i].visual.material.image = Image.merge(
                    'RGB',
                    img.split()[::-1])

        # randomization in mesh object
        mesh = pyrender.Mesh.from_trimesh(tm_list)
        intensity = np.random.uniform(
            0., 1.)  # don't change base color; only change intensity

        mesh.primitives[body].material.baseColorFactor = np.array([intensity] *
                                                                  3 + [1.])
        mesh.primitives[body].material.metallicFactor = np.random.uniform(
            0.9, 1.0)
        mesh.primitives[body].material.roughnessFactor = np.random.uniform(
            0.0, 0.3)

        return mesh

    def _calibrate_tm(self, tm):
        # calibrate mesh; shift and scale assuming car dimension (width, length) is roughly (2, 4)
        all_pts = np.concatenate([np.array(_tm.vertices) for _tm in tm],
                                 axis=0)
        pts_min = all_pts.min(0)
        pts_max = all_pts.max(0)
        pts_range = pts_max - pts_min
        pts_midpoint = (pts_min + pts_max) / 2.
        xzy_shift = [-pts_midpoint[0], -pts_min[1],
                     -pts_midpoint[2]]  # on the ground
        rot = [0., np.pi, 0.]  # calibrate heading
        scale = 1. / (pts_range[0] / 2)  # assume all car width = 2
        for i in range(len(tm)):
            tr = transform.vec2mat(xzy_shift, rot)
            tm[i].apply_transform(tr)
            tm[i].apply_scale(scale)
        mesh_dim = [pts_range[0] * scale, pts_range[2] * scale]

        return tm, mesh_dim

    @property
    def fpaths(self) -> List[str]:
        """ Paths to all meshes. """
        return self._fpaths

    @property
    def tmeshes(self) -> List:
        """ A list of trimesh objects. """
        return self._tmeshes

    @property
    def n_tmeshes(self) -> int:
        """ Number of trimeshes. """
        return len(self._tmeshes)

    @property
    def agents_meshes(self) -> List[pyrender.Mesh]:
        """ A list of meshes for all agents. """
        return self._agents_meshes

    @property
    def agents_meshes_dim(self) -> List[List[float]]:
        """ The dimensions (width, length) of agents' meshes. """
        return self._agents_meshes_dim


def main():
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    import cv2
    import argparse
    import pdb
    from vista.entities.sensors.camera_utils import CameraParams

    parser = argparse.ArgumentParser(description='Run meshlib test.')
    parser.add_argument('--mesh-dir',
                        type=str,
                        nargs='+',
                        help='Directory to meshes.')
    parser.add_argument('--out-dir',
                        type=str,
                        help='Directory to rendered outputs.')
    parser.add_argument('--rig-path', type=str, help='Path to the rig file.')
    parser.add_argument('--repeat',
                        type=int,
                        default=1,
                        help='Number of repeat to render the same mesh.')
    parser.add_argument('--show',
                        action='store_true',
                        default=False,
                        help='Show image.')
    parser.add_argument('--dump-single',
                        action='store_true',
                        default=False,
                        help='Dump every single image.')
    parser.add_argument('--dump-merged',
                        action='store_true',
                        default=False,
                        help='Dump merged image.')
    args = parser.parse_args()

    if not os.path.isdir(args.out_dir) and (args.dump_single
                                            or args.dump_merged):
        os.makedirs(args.out_dir)

    mesh_lib = MeshLib(args.mesh_dir)

    scene = pyrender.Scene(ambient_light=[.5, .5, .5],
                           bg_color=[255, 255, 255])
    light = pyrender.DirectionalLight([255, 255, 255], 5)
    scene.add(light)

    camera = CameraParams(args.rig_path, 'camera_front')
    camera.resize(200, 320)
    render_camera = pyrender.IntrinsicsCamera(fx=camera._fx,
                                              fy=camera._fy,
                                              cx=camera._cx,
                                              cy=camera._cy,
                                              znear=0.01,
                                              zfar=100000)
    scene.add(render_camera)

    renderer = pyrender.OffscreenRenderer(camera.get_width(),
                                          camera.get_height())

    all_colors = []
    for r in range(args.repeat):
        mesh_lib.reset(mesh_lib.n_meshes, random=False)
        all_colors.append([])
        for i in range(mesh_lib.n_meshes):
            trans = [0.5, -0.6, 3.2]
            theta = np.deg2rad(50)
            rot = np.array([0, np.sin(theta / 2.), 0, np.cos(theta / 2.)])
            mesh_node = mesh_lib.get_mesh_node(i, trans, rot)
            scene.add_node(mesh_node)

            color, depth = renderer.render(scene)
            if args.show:
                cv2.imshow('rendered RGB', color)
                cv2.waitKey(0)
            all_colors[-1].append(color)

            if args.dump_single:
                fpath = os.path.basename(os.path.dirname(
                    mesh_lib.fpaths[i])) + f'_{r:02d}.png'
                fpath = os.path.join(args.out_dir, fpath)
                cv2.imwrite(fpath, color)

            scene.remove_node(mesh_node)

    if False:  # row image
        row_imgs = [np.concatenate(vl, axis=0) for vl in all_colors]
        merged_img = np.concatenate(row_imgs, axis=1)
    else:
        row_imgs = [np.concatenate(vl, axis=1) for vl in all_colors]
        merged_img = np.concatenate(row_imgs, axis=0)
    if args.dump_merged:
        fpath = os.path.join(args.out_dir, 'merged.png')
        cv2.imwrite(fpath, merged_img)


if __name__ == '__main__':
    main()
