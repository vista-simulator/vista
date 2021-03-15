import os
import copy
import numpy as np
import trimesh
import pyrender


class MeshLib(object):
    """Handle all meshes of actors (as oppose to scene itself) in the scene."""

    FORMATS = ['.obj']

    def __init__(self, root_dir):
        fpaths = []        
        for sdir in os.listdir(root_dir):
            sdir = os.path.join(root_dir, sdir)
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
        for fpath in fpaths:
            tm = trimesh.load(fpath)
            meshes.append(self._temp_process(tm))
        if len(meshes) == 0:
            raise AttributeError("No mesh in the directory")
        
        self.fpaths = fpaths
        self.meshes = meshes

    def get_mesh_node(self, agent_id, trans, rot):
        mesh_node = pyrender.Node(name='agent_{}'.format(agent_id), 
                                  mesh=self.agents_meshes[agent_id],
                                  translation=trans,
                                  rotation=rot)
        return mesh_node

    def reset(self, n_agents):  
        idcs = np.random.choice(len(self.meshes), n_agents)
        # make copies for mesh instances otherwise conflicts in pyrender mesh primitives vaid
        self.agents_meshes = [copy.deepcopy(self.meshes[idx]) for idx in idcs]

    def _temp_process(self, tm):
        """[TEMPORARY] hacky way to handle the bugatti mesh"""
        tm = list(tm.geometry.values()) # for trimesh.scene

        # remove unwanted scene component
        tm.remove(tm[-1]) 
        tm.remove(tm[-1]) 
        tm.remove(tm[34]) 
        tm.remove(tm[22])

        # calibrate mesh
        for i in range(len(tm)):
            tm[i].apply_translation([0.,-4.,-4.0])
            tm[i].apply_scale(0.1)

        mesh = pyrender.Mesh.from_trimesh(tm)

        # convert mesh color from rgb to bgr
        for i in range(len(mesh.primitives)):
            mesh.primitives[i].material.baseColorFactor[:3] = mesh.primitives[i].material.baseColorFactor[:3][::-1]

        return mesh