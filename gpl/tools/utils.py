import os
import re
import yaml
import time
import collections.abc
import pprint
from collections import deque
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from attrdict import AttrDict


class Logger:
    def __init__(self, logdir, write_mode='w', with_tensorboard=True):
        self._text_writer = open(os.path.join(logdir, 'results.txt'),
                                 write_mode)
        if with_tensorboard:
            self._tf_writer = SummaryWriter(logdir)
        self._logdir = logdir
        self._scalars = {'default': {}}
        self._timer = {'default': {}}
        self._accum_timer = {'default': {}}

    def write(self, iter, group='default'):
        text = [f'iter#{iter}']
        for name, value in self._scalars[group].items():
            if hasattr(self, '_tf_writer'):
                self._tf_writer.add_scalar(f'{group}/{name}', value, iter)
            text.append(f'{name}: {value:.4f}')

        for name, value_list in self._accum_timer[group].items():
            value = np.mean(value_list)
            if hasattr(self, '_tf_writer'):
                self._tf_writer.add_scalar(f'{group}/{name}', value, iter)
            text.append(f't/{name}: {value:.4f}')

        self.print('  '.join(text))

    def print(self, data):
        if not isinstance(data, str):
            data = pprint.pformat(data)
        self._text_writer.write(data + '\n')
        self._text_writer.flush()
        print(data)

    def scalar(self, name, value, group=None):
        group = 'default' if group is None else group
        self._scalars[group][name] = float(value)

    def tic(self, name, group=None):
        group = 'default' if group is None else group
        self._timer[group][name] = [time.time()]

    def toc(self, name, group=None):
        group = 'default' if group is None else group
        assert len(
            self._timer[group][name]) == 1, f'Should call tic({name}) first'
        self._timer[group][name].append(time.time())

        if not name in self._accum_timer[group].keys():
            self._accum_timer[group][name] = deque(maxlen=500)
        tic, toc = self._timer[group][name]
        self._accum_timer[group][name].append(toc - tic)

    def create_group(self, group):
        self._scalars[group] = {}
        self._timer[group] = {}
        self._accum_timer[group] = {}

    def close(self):
        self._text_writer.close()
        if hasattr(self, '_tf_writer'):
            self._tf_writer.close()


def save_checkpoint(fpath, n_iter, model, optimizer):
    torch.save(
        {
            'n_iter': n_iter,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, fpath)


def load_checkpoint(fpath, model, optimizer=None, load_optim=True):
    ckpt = torch.load(fpath)  #, map_location=torch.device('cpu'))
    model.load_state_dict(ckpt['model'])
    if load_optim:
        optimizer.load_state_dict(ckpt['optimizer'])
    return ckpt['n_iter']


def preprocess_config(config):
    for k in ['dataset:trace_paths', 'val_dataset:trace_paths']:
        v = get_dict_value_by_str(config, k)
        if isinstance(v, str):
            set_dict_value_by_str(config, k, validate_path(v))
        elif isinstance(v, list):
            set_dict_value_by_str(config, k, [validate_path(vv) for vv in v])
        else:
            raise NotImplementedError(
                f'Unrecognized dict value {v} to be modified')


def validate_path(path):
    """ Handle '~', '$' and relative path. """
    valid_path = ['/'] if path.startswith('/') else []
    for v in path.split('/'):
        if v.startswith('$'):
            v = v[1:]
            assert v in os.environ, f'Remember to set ${v}'
            v = os.environ[v]
        valid_path.append(v)
    valid_path = os.path.join(*valid_path)
    valid_path = os.path.abspath(os.path.expanduser(valid_path))
    return valid_path


def load_yaml(fpath):
    """ Load yaml file with scientifc notation.
        Ref: https://stackoverflow.com/questions/30458977/yaml-loads-5e-6-as-string-and-not-a-number """
    with open(fpath) as f:
        loader = yaml.SafeLoader
        loader.add_implicit_resolver(
            u'tag:yaml.org,2002:float',
            re.compile(
                u'''^(?:
            [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$''', re.X), list(u'-+0123456789.'))
        data = yaml.load(f, Loader=loader)
    return AttrDict(data)


def update_dict(d, u):
    """ Recursively update a python dictionary """
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d


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
