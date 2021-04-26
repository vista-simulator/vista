from ray.rllib.agents.registry import ALGORITHMS


def _import_toy_ppo():
    from .toy_ppo import ToyPPOTrainer, DEFAULT_CONFIG
    return ToyPPOTrainer, DEFAULT_CONFIG


def _import_guided_ppo():
    from .guided_ppo import GuidedPPOTrainer, DEFAULT_CONFIG
    return GuidedPPOTrainer, DEFAULT_CONFIG


CUSTOM_ALGORITHMS = {
    'ToyPPO': _import_toy_ppo,
    'GuidedPPO': _import_guided_ppo
}


def get_trainer_class(alg: str, return_config=False) -> type:
    if alg in ALGORITHMS.keys():
        class_ = ALGORITHMS[alg]()
    elif alg in CUSTOM_ALGORITHMS:
        class_, config = CUSTOM_ALGORITHMS[alg]()
    else:
        raise NotImplementedError

    if return_config:
        return class_, config
    return class_