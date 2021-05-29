import os
import re
import yaml
import copy
import itertools
import collections.abc
from functools import partial, partialmethod
from importlib import import_module
import numpy as np
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog

import envs
import models
from . import callbacks


def load_yaml(fpath):
    """ Load yaml file with scientifc notation.
        Ref: https://stackoverflow.com/questions/30458977/yaml-loads-5e-6-as-string-and-not-a-number """
    with open(fpath) as f:
        loader = yaml.SafeLoader
        loader.add_implicit_resolver(
            u'tag:yaml.org,2002:float',
            re.compile(u'''^(?:
            [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$''', re.X),
            list(u'-+0123456789.'))
        data = yaml.load(f, Loader=loader)
    return data


def get_latest_checkpoint(ckpt_root_dir):
    ckpt_dirnames = [v for v in os.listdir(ckpt_root_dir) if 'checkpoint' in v]
    ckpt_nums = [int(v.split('_')[-1]) for v in ckpt_dirnames]
    latest_ckpt_dirname = ckpt_dirnames[np.argmax(ckpt_nums)]
    latest_ckpt = os.path.join(ckpt_root_dir, latest_ckpt_dirname, latest_ckpt_dirname.replace('_', '-'))
    assert os.path.exists(latest_ckpt)
    return latest_ckpt


def update_dict(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def resume_from_latest_ckpt(exp, exp_name):
    exp_dir = os.path.join(exp['local_dir'], exp_name)
    assert os.path.isdir(exp_dir)
    trial_dirs = []
    for v in os.listdir(exp_dir):
        v = os.path.join(exp_dir, v)
        if os.path.isdir(v):
            trial_dirs.append(v)
    trial_time = ['_'.join(v.split('_')[-2:]) for v in trial_dirs]
    latest_trial_dir = trial_dirs[np.argmax(trial_time)]
    ckpt_dirnames = [v for v in os.listdir(latest_trial_dir) if 'checkpoint' in v]
    latest_ckpt_dirname = ckpt_dirnames[np.argmax([int(v.split('_')[-1]) for v in ckpt_dirnames])]
    latest_ckpt = os.path.join(latest_trial_dir, latest_ckpt_dirname, latest_ckpt_dirname.replace('_', '-'))
    assert os.path.exists(latest_ckpt)
    exp['restore'] = latest_ckpt


def update_by_job_array_exp(exp, job_array_module, job_array_task_id):
    job_array_mod = import_module('job_array.{}'.format(job_array_module))
    job_array_exp = getattr(job_array_mod, 'job_array_exp')
    search_method = getattr(job_array_mod, 'search_method')
    if search_method == 'grid':
        job_array_exp_flat_list = []
        for k, v in job_array_exp.items():
            assert isinstance(v, list)
            task_exp_list = []
            for vv in v:
                task_exp_list.append([k, vv])
            job_array_exp_flat_list.append(task_exp_list)
        all_task_exp = list(itertools.product(*job_array_exp_flat_list))
    elif search_method == 'cartesian':
        n_task = np.unique([len(v) for v in list(job_array_exp.values())])
        assert len(n_task) == 1, 'Every list should have the same length'
        n_task = n_task[0]
        all_task_exp = []
        for i in range(n_task):
            all_task_exp.append([])
            for k, v in job_array_exp.items():
                all_task_exp[-1].append([k, v[i]])
    else:
        raise NotImplementedError('Unrecognized search method {}'.format(search_method))
    task_exp = all_task_exp[job_array_task_id]
    exp_name = []
    def update_weird_str(_s):
        return _s.replace('\'', '').replace(',', '').replace(' ', '').replace('[', '').replace(']', '').replace('~','').replace('/','_')
    for v in task_exp:
        set_dict_value_by_str(exp, v[0], v[1])
        exp_name.append(''.join([vv[0] for vv in v[0].split(':')[-1].split('_')]) + update_weird_str(str(v[1])))
    exp_name = '-'.join(exp_name)
    
    return exp, exp_name


def get_dict_value_by_str(data, string, delimiter=':'):
    """ Recursively access nested dict with string,
        e.g., get_dict_value_by_str(d, 'a:b') := d['a']['b']. """
    assert isinstance(data, dict)
    keys = string.split(delimiter)
    if len(keys) == 1:
        return data[keys[0]]
    else:
        string = ':'.join(keys[1:])
        return get_dict_value_by_str(data[keys[0]], string, delimiter)


def set_dict_value_by_str(data, string, value, delimiter=':'):
    """ Recursively access nested dict with string and set value,
        e.g., set_dict_value_by_str(d, 'a:b', 1) := d['a']['b'] = 1. """
    assert isinstance(data, dict)
    keys = string.split(delimiter)
    if len(keys) == 1:
        data[keys[0]] = value
        return
    else:
        string = ':'.join(keys[1:])
        set_dict_value_by_str(data[keys[0]], string, value, delimiter)


def register_custom_env(env_name):
    """ Register custom environment in rllib. """
    def _env_creator_base(env_config, env_cls):
        env_config = copy.copy(env_config) # keep original env_config intact
        _wrappers = env_config.pop('wrappers')
        _wrappers_config = env_config.pop('wrappers_config')
        _env = env_cls(**env_config)
        for _wrapper in _wrappers:
            if _wrappers_config and _wrapper in _wrappers_config.keys():
                _wrapper_config = _wrappers_config[_wrapper]
            else:
                _wrapper_config = dict()
            _wrapper = getattr(envs.wrappers, _wrapper)
            _env = _wrapper(_env, **_wrapper_config)
        return _env
    _env_creator = partial(_env_creator_base, env_cls=getattr(envs, env_name))
    register_env(env_name, _env_creator)
    return _env_creator


def register_custom_model(model_config):
    """ Register custom model in rllib. """
    try: # TODO: hacky fix for checkpoints that doesn't save this attribute because of pop
        action_dist_config = model_config['custom_model_config']['custom_action_dist_config']
    except:
        try:
            action_dist_config = model_config.pop('custom_action_dist_config')
        except:
            print('[!!!!!!!!!!!!!!!!!!!!] Use hardcoded action distribution config')
            action_dist_config = {'low': [-5., -15.], 'high': [5., 15.]}
    ActDist = getattr(models, model_config['custom_action_dist'])
    ActDist = partialclass(ActDist, **action_dist_config)
    ModelCatalog.register_custom_action_dist(model_config['custom_action_dist'], ActDist)

    if 'custom_model' not in model_config.keys():
        return 
    # model_config = copy.copy(model_config) # don't modify original model config
    Model = getattr(models, model_config['custom_model'])
    ModelCatalog.register_custom_model(model_config['custom_model'], Model)


def set_callbacks(exp, agent_ids):
    """ Set callbacks to a callback class by string. """
    _callbacks = getattr(callbacks, exp['config']['callbacks'])
    if 'callbacks_config' in exp['config']['multiagent'].keys():
        kwargs = exp['config']['multiagent']['callbacks_config']
    else:
        kwargs = dict()
    _callbacks = partial(_callbacks, agent_ids=agent_ids, **kwargs)
    exp['config']['callbacks'] = _callbacks


def partialclass(cls, *args, **kwargs):
    class NewCls(cls):
        __init__ = partialmethod(cls.__init__, *args, **kwargs)

    return NewCls