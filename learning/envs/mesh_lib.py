import os
import copy
import numpy as np
import trimesh
import pyrender


class MeshLib(object):
    """Handle all meshes of actors (as oppose to scene itself) in the scene."""

    FORMATS = ['.obj']

    def __init__(self, root_dir):
        root_dir = os.path.abspath(os.path.expanduser(root_dir))
        fpaths = []
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

        meshes = []
        meshes_dim = []
        for fpath in fpaths:
            try:
                tm = trimesh.load(fpath)
                mesh, mesh_dim = self.preprocess(tm)
                meshes.append(mesh)
                meshes_dim.append(mesh_dim)
            except:
                print('Failed to load/process mesh {}'.format(fpath))
        if len(meshes) == 0:
            raise AttributeError("No mesh in the directory")
        
        self.fpaths = fpaths
        self.meshes = meshes
        self.meshes_dim = meshes_dim

    def get_mesh_node(self, agent_id, trans, rot):
        # convert to OpenGL coordinates
        trans[2] *= -1 # (x, z, y) -> (x, z, -y)
        rot[3] *= -1 # negate theta in quaternion
        mesh_node = pyrender.Node(name='agent_{}'.format(agent_id), 
                                  mesh=self.agents_meshes[agent_id],
                                  translation=trans,
                                  rotation=rot)
        return mesh_node

    def reset(self, n_agents):  
        idcs = np.random.choice(len(self.meshes), n_agents)
        # make copies for mesh instances otherwise conflicts in pyrender mesh primitives vaid
        self.agents_meshes = [copy.deepcopy(self.meshes[idx]) for idx in idcs]
        self.agents_meshes_dim = [copy.copy(self.meshes_dim[idx]) for idx in idcs]

    def preprocess(self, tm):
        """ preprocess mesh """
        tm = list(tm.geometry.values()) # for trimesh.scene

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

        # convert trimesh to pyrender mesh
        mesh = pyrender.Mesh.from_trimesh(tm)

        # convert mesh color from rgb to bgr
        for i in range(len(mesh.primitives)):
            mesh.primitives[i].material.baseColorFactor[:3] = mesh.primitives[i].material.baseColorFactor[:3][::-1]

        return mesh, mesh_dim

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
    import cv2
    import argparse
    from vista.util import Camera

    parser = argparse.ArgumentParser(description='Run meshlib test.')
    parser.add_argument(
        'mesh_dir',
        type=str,
        help='Directory to meshes.')
    parser.add_argument(
        'out_dir',
        type=str,
        help='Directory to rendered outputs.')
    args = parser.parse_args()

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    mesh_lib = MeshLib(args.mesh_dir)

    scene = pyrender.Scene(ambient_light=[.1, .1, .1])
    light = pyrender.DirectionalLight([255, 255, 255], 10)
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

    for i, mesh in enumerate(mesh_lib.meshes):
        mesh_node = pyrender.Node(mesh=mesh, translation=[0,0,-5])
        scene.add_node(mesh_node)

        color, depth = renderer.render(scene)
        if False:
            cv2.imshow('rendered RGB', color)
            cv2.waitKey(0)

        fpath = os.path.basename(os.path.dirname(mesh_lib.fpaths[i])) + '.png'
        fpath = os.path.join(args.out_dir, fpath)
        cv2.imwrite(fpath, color)

        scene.remove_node(mesh_node)


if __name__ == '__main__':
    main()
