import os
import sys
import shutil
from termcolor import colored


root_dir = os.path.expanduser(sys.argv[1])
out_root_dir = os.path.expanduser(sys.argv[2])
ckpt_name = 'model_deepknight.pkl'
config_name = 'config.yaml'
config = """dataset:
  sensors: [fcamera] #navimap
  min_distance: 3 
  max_distance: null
  camera_size: [200, 320] # [35, 155] = size after roi cropping
  voxel_size: 0.2
  navimap_size: [50, 50]
  lookaheads: [0]

model:
  extractors:
    fcamera:
      name: convnet_ma
      standardize: False
  estimator:
    name: deterministic_ma

deployment: 
  ignore_future: True
  fusion: uniform
"""

def replace_weird_str(s):
    return s.replace('\'', '').replace(',', '').replace(' ', '').replace('[', '').replace(']', '')

for subdname in os.listdir(root_dir):
    subdir = os.path.join(root_dir, subdname)
    for ssubdname in [_v for _v in os.listdir(subdir) if 'PPO' in _v]:
        ssubdir = os.path.join(subdir, ssubdname)
        n_valid_ckpt = 0
        for ckpt_dname in os.listdir(ssubdir):
            if 'checkpoint' in ckpt_dname:
                ckpt_dir = os.path.join(ssubdir, ckpt_dname)
                ckpt_path = os.path.join(ckpt_dir, ckpt_name)
                if os.path.exists(ckpt_path):
                    out_dir = os.path.join(out_root_dir, *replace_weird_str(ssubdir).split('/')[-3:-1], ckpt_dname)
                    if not os.path.isdir(out_dir):
                        os.makedirs(out_dir)
                    out_path = os.path.join(out_dir, ckpt_name)
                    shutil.copy(ckpt_path, out_path)
                    print('Copy to {}'.format(out_path))

                    if True:
                        if 'sTrue' in ssubdir:
                            if 'standardize: False' in config:
                                config = config.replace('standardize: False', 'standardize: True')
                        else:
                            if 'standardize: True' in config:
                                config = config.replace('standardize: True', 'standardize: False')

                    config_out_path = os.path.join(out_dir, config_name)
                    with open(config_out_path, 'w') as f:
                        f.write(config)            
                    n_valid_ckpt += 1
        if n_valid_ckpt == 0:
            colored('No valid checkpoint found in {}'.format(ssubdir), 'red')
