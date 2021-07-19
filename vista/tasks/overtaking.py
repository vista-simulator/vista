import numpy as np

from .base import Base


class Overtaking:
    DEFAULT_CONFIG = {
        'reward_type': 'non-crash',
        'maximal_rotation': np.pi / 10.,
    }

    def __init__(self, **kwargs):
        super(Overtaking, self).__init__(**kwargs)
