import os
import copy
import numpy as np
from PIL import Image
import trimesh
import pyrender


class MeshLib(object):
    """Handle all meshes of actors (as opposed to scene itself) in the scene."""

    FORMATS = ['.obj']

    def __init__(self, root_dirs):
        root_dirs = [root_dirs] if not isinstance(root_dirs, list) else root_dirs
        fpaths = []
        for root_dir in root_dirs:
            root_dir = os.path.abspath(os.path.expanduser(root_dir))
            for sdir in os.listdir(root_dir):
                sdir = os.path.join(root_dir, sdir)
                if not os.path.isdir(sdir):
                    continue
                fpath = os.listdir(sdir)
                for fm in self.FORMATS: # elements in the front has higher priority
                    fp = [_fp for _fp in fpath if os.path.splitext(_fp)[-1] == fm]
                    if len(fp) == 1:
                        fpath = fp[0]
                        break
                if isinstance(fpath, list):
                    print('Cannot find mesh with supported format in {}'.format(sdir))
                    continue
                fpath = os.path.join(sdir, fpath)
                fpaths.append(fpath)

        tmeshes = dict()
        for i, fpath in enumerate(fpaths):
            try:
                tm = trimesh.load(fpath)
                tm = list(tm.geometry.values()) # convert from scene to trimesh
                tm, mesh_dim = self.calibrate_tm(tm)
                if os.path.basename(os.path.dirname(os.path.dirname(fpath))) == 'carpack01':
                    source = 'carpack01'
                    body_images = dict()
                    for color in ['Black', 'Blue', 'Green', 'Red', 'White', 'Yellow', 'Grey']:
                        img_path = os.path.splitext(fpath)[0] + color + '.png'
                        if os.path.exists(img_path):
                            img = Image.open(img_path)
                            body_images[color] = img
                    extra = {'body_images': body_images}
                else:
                    source = 'default'
                    extra = dict()
                tmeshes[i] = {
                    'fpath': fpath,
                    'tmesh': tm,
                    'mesh_dim': mesh_dim,
                    'source': source,
                    'extra': extra,
                }
            except:
                print('Failed to load/process mesh {}'.format(fpath))
        if len(tmeshes) == 0:
            raise AttributeError("No mesh in the directory")
        
        self.fpaths = fpaths
        self.tmeshes = tmeshes
        self.n_meshes = len(self.tmeshes)

    def get_mesh_node(self, agent_idx, trans, rot):
        # convert to OpenGL coordinates
        trans[2] *= -1 # (x, z, y) -> (x, z, -y)
        rot[3] *= -1 # negate theta in quaternion
        mesh_node = pyrender.Node(name='agent_{}'.format(agent_idx), 
                                  mesh=self.agents_meshes[agent_idx],
                                  translation=trans,
                                  rotation=rot)
        return mesh_node

    def reset(self, n_agents, random=True):
        n_tmeshes = len(self.tmeshes)
        idcs = np.random.choice(n_tmeshes, n_agents) if random else np.arange(n_tmeshes)
        
        self.agents_meshes = []
        self.agents_meshes_dim = []
        for idx in idcs:
            mesh = self.tmesh2mesh(self.tmeshes[idx])
            self.agents_meshes.append(mesh)
            self.agents_meshes_dim.append(self.tmeshes[idx]['mesh_dim'])
    
    def tmesh2mesh(self, tm):
        if tm['source'] == 'carpack01':
            # make a copy to keep the original tmesh intact
            tm_list = copy.deepcopy(tm['tmesh'])

            # randomization in tmesh object
            # wheel, glass, optic, body = range(len(tm_list))
            body = np.argmax([v.triangles.shape[0] for v in tm_list]) # NOTE: hacky way to get body mesh
            color = np.random.choice(list(tm['extra']['body_images'].keys()))
            tm_list[body].visual.material.image = tm['extra']['body_images'][color]
            Ns = np.random.randint(10, 300) # specular highlight (0-1000)
            tm_list[body].visual.material.kwargs.update(Ns=Ns)

            # convert mesh color from rgb to bgr
            for i in range(len(tm_list)):
                img = tm_list[i].visual.material.image
                if img is not None:
                    tm_list[i].visual.material.image = Image.merge('RGB', img.split()[::-1])

            # randomization in mesh object
            mesh = pyrender.Mesh.from_trimesh(tm_list)
            intensity = np.random.uniform(0., 1.) # don't change base color; only change intensity
            mesh.primitives[body].material.baseColorFactor = np.array([intensity]*3 + [1.])
            mesh.primitives[body].material.metallicFactor = np.random.uniform(0.9, 1.0)
            mesh.primitives[body].material.roughnessFactor = np.random.uniform(0.0, 0.3)
        else:
            mesh = pyrender.Mesh.from_trimesh(tm['tmesh'])

            # convert mesh color from rgb to bgr
            for i in range(len(mesh.primitives)):
                mesh.primitives[i].material.baseColorFactor[:3] = mesh.primitives[i].material.baseColorFactor[:3][::-1]

        return mesh
    
    def calibrate_tm(self, tm):
        # calibrate mesh; shift and scale assuming car dimension (width, length) is roughly (2, 4)
        all_pts = np.concatenate([np.array(_tm.vertices) for _tm in tm], axis=0)
        pts_min = all_pts.min(0)
        pts_max = all_pts.max(0)
        pts_range = pts_max - pts_min
        pts_midpoint = (pts_min + pts_max) / 2.
        xzy_shift = [-pts_midpoint[0], -pts_max[1], -pts_midpoint[2]] # zero-height is camera height
        scale = 1. / (pts_range[0] / 2.) # car width = 2, didn't check car length = 4
        for i in range(len(tm)):
            tm[i].apply_translation(xzy_shift)
            tm[i].apply_scale(scale)
        mesh_dim = [pts_range[0] * scale, pts_range[2] * scale]

        return tm, mesh_dim

    def get_yaw_transform(self, yaw):
        c, s = np.cos(yaw), np.sin(yaw)
        rot = np.array([
            [c, 0, -s, 0],
            [0, 1,  0, 0],
            [s, 0,  c, 0],
            [0, 0,  0, 1]
        ])
        return rot


def main():
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    import cv2
    import argparse
    from vista.util import Camera

    parser = argparse.ArgumentParser(description='Run meshlib test.')
    parser.add_argument(
        '--mesh-dir',
        type=str,
        nargs='+',
        help='Directory to meshes.')
    parser.add_argument(
        '--out-dir',
        type=str,
        help='Directory to rendered outputs.')
    parser.add_argument(
        '--repeat',
        type=int,
        default=1,
        help='Number of repeat to render the same mesh.')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='Show image.')
    parser.add_argument(
        '--dump-single',
        action='store_true',
        default=False,
        help='Dump every single image.')
    parser.add_argument(
        '--dump-merged',
        action='store_true',
        default=False,
        help='Dump merged image.')
    args = parser.parse_args()

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    mesh_lib = MeshLib(args.mesh_dir)

    scene = pyrender.Scene(ambient_light=[.5, .5, .5], bg_color=[255, 255, 255])
    light = pyrender.DirectionalLight([255, 255, 255], 5)
    scene.add(light)

    camera = Camera('camera_front')
    camera.resize(250, 400)
    render_camera = pyrender.IntrinsicsCamera(
        fx=camera._fx,
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
            trans = [-0.5, 0.8, 3.5]
            rot = np.array([0, 1, 0, np.deg2rad(30)])
            rot = rot / np.linalg.norm(rot)
            mesh_node = mesh_lib.get_mesh_node(i, trans, rot)
            scene.add_node(mesh_node)

            color, depth = renderer.render(scene)
            if args.show:
                cv2.imshow('rendered RGB', color)
                cv2.waitKey(0)
            all_colors[-1].append(color)

            if args.dump_single:
                fpath = os.path.basename(os.path.dirname(mesh_lib.fpaths[i])) + '_{:02d}.png'.format(r)
                fpath = os.path.join(args.out_dir, fpath)
                cv2.imwrite(fpath, color)

            scene.remove_node(mesh_node)
    
    row_imgs = [np.concatenate(vl, axis=0) for vl in all_colors]
    merged_img = np.concatenate(row_imgs, axis=1)
    if args.dump_merged:
        fpath = os.path.join(args.out_dir, 'merged.png')
        cv2.imwrite(fpath, merged_img)


if __name__ == '__main__':
    main()
