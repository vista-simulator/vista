import re
import yaml
import copy
from functools import partial, partialmethod
from importlib import import_module
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
    if 'custom_model' not in model_config.keys():
        return 
    Model = getattr(models, model_config['custom_model'])
    ModelCatalog.register_custom_model(model_config['custom_model'], Model)

    action_dist_config = model_config.pop('custom_action_dist_config')
    ActDist = getattr(models, model_config['custom_action_dist'])
    ActDist = partialclass(ActDist, **action_dist_config)
    ModelCatalog.register_custom_action_dist(model_config['custom_action_dist'], ActDist)


def set_callbacks(exp, agent_ids):
    """ Set callbacks to a callback class by string. """
    _callbacks = getattr(callbacks, exp['config']['callbacks'])
    _callbacks = partial(_callbacks, agent_ids=agent_ids)
    exp['config']['callbacks'] = _callbacks


def partialclass(cls, *args, **kwargs):
    class NewCls(cls):
        __init__ = partialmethod(cls.__init__, *args, **kwargs)

    return NewCls