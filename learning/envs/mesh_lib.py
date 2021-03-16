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
            tm = trimesh.load(fpath)
            mesh, mesh_dim = self._temp_process(tm)
            meshes.append(mesh)
            meshes_dim.append(mesh_dim)
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

    def _temp_process(self, tm):
        """[TEMPORARY] hacky way to handle the bugatti mesh"""
        tm = list(tm.geometry.values()) # for trimesh.scene

        # remove unwanted scene component
        if True:
            tm.remove(tm[-1]) 
            tm.remove(tm[-1]) 
            tm.remove(tm[34]) 
            tm.remove(tm[22])

            new_tm = []
            for _tm in tm:
                pts = np.array(_tm.vertices)
                if np.all(pts.min(0) > -10) and np.all(pts.max(0) < 10):
                    new_tm.append(_tm)
            tm = new_tm

        # calibrate mesh; shift and scale assuming car dimension (width, length) is roughly (2, 4)
        all_pts = np.concatenate([np.array(_tm.vertices) for _tm in tm], axis=0)
        pts_min = all_pts.min(0)
        pts_max = all_pts.max(0)
        pts_range = pts_max - pts_min
        pts_midpoint = (pts_min + pts_max) / 2.
        xzy_shift = [-pts_midpoint[0], -pts_max[1], -pts_midpoint[2]] # zero-height is camera height
        scale = 1. / (pts_range[0] / 2.) # car width = 2, didn't check car length = 4
        rot = self.get_yaw_transform(0.15) # compensate the mesh that is a bit oriented toward left
        for i in range(len(tm)):
            tm[i].apply_translation(xzy_shift)
            tm[i].apply_scale(scale)
            tm[i].apply_transform(rot)
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