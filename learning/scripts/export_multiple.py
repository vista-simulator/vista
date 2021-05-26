import os
import sys
import subprocess
from termcolor import colored


root_dir = os.path.expanduser(sys.argv[1])
ckpt_dname = 'checkpoint_94'
for subdname in os.listdir(root_dir):
    subdir = os.path.join(root_dir, subdname)
    for ssubdname in [_v for _v in os.listdir(subdir) if 'PPO' in _v]:
        ssubdir = os.path.join(subdir, ssubdname)
        if ckpt_dname in os.listdir(ssubdir):
            ckpt_dir = os.path.join(ssubdir, ckpt_dname)
            ckpt_path = os.path.join(ckpt_dir, ckpt_dname.replace('_', '-'))

            subprocess.call(['python', '-m', 'misc.export_model', ckpt_path, '--export', '--to-deepknight'])
            # subprocess.call(['ls', os.path.join(ckpt_dir, 'model_deepknight.pkl')])
        else:
            colored('[!!!!] Cannot find {} in {}'.format(ckpt_dname, ssubdir), 'red')