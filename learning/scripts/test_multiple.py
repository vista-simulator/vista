import os
import sys
import subprocess
from termcolor import colored


root_dir = os.path.expanduser(sys.argv[1])
ckpt_dname = 'checkpoint_000313' # 'checkpoint_94'
ckpt_dname_alias = '_'.join(ckpt_dname.split('_')[:-1] + [str(int(ckpt_dname.split('_')[-1]))])
for subdname in os.listdir(root_dir):
    subdir = os.path.join(root_dir, subdname)
    if not os.path.isdir(subdir):
        continue
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

            if 'pd0.3-sc0.1-cadc{low:-0.07high:0.07}' in ckpt_path or 'pd0.3-sc0.1-cadc{low:-0.3high:0.3}' in ckpt_path: # DEBUG
                continue

            subprocess.call(['python', 'test.py', ckpt_path, '--eval-config', 'config/local/eval.yaml', 
                             '--save-rollout', '--episodes', '1000', '--save-dir-suffix', 'final'])

            # rollout_path = os.path.join(ckpt_dir, 'results_overlap0.05_roadwidth5', 'rollout.pkl')
            # subprocess.call(['python', 'misc/simple_metrics.py', rollout_path, '--task', 'Overtaking'])
        else:
            colored('[!!!!] Cannot find {} in {}'.format(ckpt_dname, ssubdir), 'red')