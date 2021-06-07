import os
import sys
import subprocess
from termcolor import colored


root_dir = os.path.expanduser(sys.argv[1])
ckpt_dname = 'checkpoint_000094' # 'checkpoint_94'
ckpt_dname_alias = '_'.join(ckpt_dname.split('_')[:-1] + [str(int(ckpt_dname.split('_')[-1]))])
for subdname in os.listdir(root_dir):
    subdir = os.path.join(root_dir, subdname)
    for ssubdname in [_v for _v in os.listdir(subdir) if 'PPO' in _v]:
        ssubdir = os.path.join(subdir, ssubdname)
        has_ckpt = ckpt_dname in os.listdir(ssubdir)
        has_ckpt_alias = ckpt_dname_alias in os.listdir(ssubdir)
        if has_ckpt or has_ckpt_alias:
            ckpt_dname_or_alias = ckpt_dname if has_ckpt else ckpt_dname_alias
            ckpt_dir = os.path.join(ssubdir, ckpt_dname_or_alias)
            ckpt_path = os.path.join(ckpt_dir, ckpt_dname_or_alias.replace('_', '-'))
            if not os.path.exists(ckpt_path):
                ckpt_path = ckpt_path.split('/')
                ckpt_path[-1] = 'checkpoint-' + str(int(ckpt_path[-1].split('-')[-1]))
                ckpt_path = os.path.join('/', *ckpt_path)
                assert os.path.exists(ckpt_path)

            subprocess.call(['python', '-m', 'misc.export_model', ckpt_path, '--export', '--to-deepknight'])
            # subprocess.call(['ls', os.path.join(ckpt_dir, 'model_deepknight.pkl')])
        else:
            colored('[!!!!] Cannot find {} in {}'.format(ckpt_dname, ssubdir), 'red')