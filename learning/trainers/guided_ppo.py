import os
import sys
import gym
import numpy as np
from typing import Dict, List, Type, Union
import torch
from torch.nn import functional as F

from ray.rllib.evaluation.postprocessing import compute_gae_for_sample_batch, \
Postprocessing
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import TensorType, TrainerConfigDict
from ray.rllib.utils.torch_ops import apply_grad_clipping, \
    explained_variance, sequence_mask

from ray.rllib.agents.ppo import PPOTorchPolicy, PPOTrainer, DEFAULT_CONFIG

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from policies.exported_policy import ExportedPolicy


def custom_loss_fn(
        policy: Policy, model: ModelV2,
        dist_class: Type[TorchDistributionWrapper],
        train_batch: SampleBatch) -> Union[TensorType, List[TensorType]]:
    """Constructs the loss for Proximal Policy Objective.
    Args:
        policy (Policy): The Policy to calculate the loss for.
        model (ModelV2): The Model to calculate the loss for.
        dist_class (Type[ActionDistribution]: The action distr. class.
        train_batch (SampleBatch): The training data.
    Returns:
        Union[TensorType, List[TensorType]]: A single loss tensor or a list
            of loss tensors.
    """
    logits, state = model.from_batch(train_batch, is_training=True)
    curr_action_dist = dist_class(logits, model)

    # RNN case: Mask away 0-padded chunks at end of time axis.
    if state:
        B = len(train_batch["seq_lens"])
        max_seq_len = logits.shape[0] // B
        mask = sequence_mask(
            train_batch["seq_lens"],
            max_seq_len,
            time_major=model.is_time_major())
        mask = torch.reshape(mask, [-1])
        num_valid = torch.sum(mask)

        def reduce_mean_valid(t):
            return torch.sum(t[mask]) / num_valid

    # non-RNN case: No masking.
    else:
        mask = None
        reduce_mean_valid = torch.mean

    prev_action_dist = dist_class(train_batch[SampleBatch.ACTION_DIST_INPUTS],
                                  model)

    logp_ratio = torch.exp(
        curr_action_dist.logp(train_batch[SampleBatch.ACTIONS]) -
        train_batch[SampleBatch.ACTION_LOGP])
    action_kl = prev_action_dist.kl(curr_action_dist)
    mean_kl = reduce_mean_valid(action_kl)

    curr_entropy = curr_action_dist.entropy()
    mean_entropy = reduce_mean_valid(curr_entropy)

    # inference of guidance policy 
    obs_filter = policy.guidance.obs_filter
    if type(obs_filter).__name__ == 'MeanStdFilter':
        x = train_batch['obs'][:,-obs_filter.shape[0]:] # NOTE: assume obs list = [vis_obs, state_obs, wrappers_obs]
        if obs_filter.demean:
            x = x - torch.Tensor(obs_filter.rs.mean).to(x) # TODO: np2torch should be somewhere else
        if obs_filter.destd:
            x = x / (torch.Tensor(obs_filter.rs.std).to(x) + 1e-8) # TODO: np2torch should be somewhere else
        if obs_filter.clip:
            x = np.clip(x, -obs_filter.clip, obs_filter.clip)
    elif type(obs_filter).__name__ == 'NoFilter':
        g_obs_shape = 0
        for ospace in policy.guidance.model.obs_space:
            g_obs_shape += np.prod(ospace.shape)
        x = train_batch['obs'][:,-g_obs_shape:]
    else:
        raise NotImplementedError('Unrecognized obs filter of guidance policy {}'.format(obs_filter))
    _, _, act_info = policy.guidance.compute_action(x, None, None, input_mode=1)
    guidance_action_dist = dist_class(act_info['action_dist_inputs'], model)

    # compute advantage; otherwise will cause [KeyError: 'advantages'] later on
    advantage = train_batch[Postprocessing.ADVANTAGES]

    # define policy loss with guidance policy
    criterion = policy.guidance.guidance_criterion
    if criterion == 'simple_kl':
        guidance_kl = guidance_action_dist.kl(curr_action_dist)
        guidance_kl *= -1 # NOTE: compensate for the negative sign in surrogate loss later
        surrogate_loss = guidance_kl 
    elif criterion == 'kl_with_ppo':
        guidance_kl = guidance_action_dist.kl(curr_action_dist)
        guidance_kl *= -1 # NOTE: compensate for the negative sign in surrogate loss later
        surrogate_loss = torch.min(
            guidance_kl,
            guidance_kl * torch.clamp(
                logp_ratio, 1 - policy.config["clip_param"],
                1 + policy.config["clip_param"]))
    elif criterion == 'simple_reverse_kl':
        guidance_reverse_kl = curr_action_dist.kl(guidance_action_dist)
        guidance_reverse_kl *= -1 # NOTE: compensate for the negative sign in surrogate loss later
        surrogate_loss = guidance_reverse_kl 
    elif criterion == 'l2':
        guidance_act_dist_inp = torch.Tensor(act_info['action_dist_inputs']).to(logits)
        surrogate_loss = F.mse_loss(logits, guidance_act_dist_inp)
        surrogate_loss *= -1 # NOTE: compensate for the negative sign in surrogate loss later
    elif criterion == 'l1':
        guidance_act_dist_inp = torch.Tensor(act_info['action_dist_inputs']).to(logits)
        surrogate_loss = F.l1_loss(logits, guidance_act_dist_inp)
        surrogate_loss *= -1 # NOTE: compensate for the negative sign in surrogate loss later
    elif criterion == 'smooth_l1':
        guidance_act_dist_inp = torch.Tensor(act_info['action_dist_inputs']).to(logits)
        surrogate_loss = F.smooth_l1_loss(logits, guidance_act_dist_inp)
        surrogate_loss *= -1 # NOTE: compensate for the negative sign in surrogate loss later
    elif criterion == 'cross_entropy':
        guidance_kl = guidance_action_dist.kl(curr_action_dist)
        guidance_ent = guidance_action_dist.entropy()
        surrogate_loss = guidance_kl + guidance_ent
        surrogate_loss *= -1 # NOTE: compensate for the negative sign in surrogate loss later
    elif criterion == 'reverse_cross_entropy':
        guidance_reverse_kl = curr_action_dist.kl(guidance_action_dist)
        curr_act_dist_ent = curr_action_dist.entropy()
        surrogate_loss = guidance_reverse_kl + curr_act_dist_ent
        surrogate_loss *= -1 # NOTE: compensate for the negative sign in surrogate loss later
    else:
        raise ValueError('Unrecognized criterion {}'.format(criterion))
    surrogate_loss = policy.guidance.guidance_coef * surrogate_loss
    mean_policy_loss = reduce_mean_valid(-surrogate_loss)

    if policy.config["use_gae"]:
        prev_value_fn_out = train_batch[SampleBatch.VF_PREDS]
        value_fn_out = model.value_function()
        vf_loss1 = torch.pow(
            value_fn_out - train_batch[Postprocessing.VALUE_TARGETS], 2.0)
        vf_clipped = prev_value_fn_out + torch.clamp(
            value_fn_out - prev_value_fn_out, -policy.config["vf_clip_param"],
            policy.config["vf_clip_param"])
        vf_loss2 = torch.pow(
            vf_clipped - train_batch[Postprocessing.VALUE_TARGETS], 2.0)
        vf_loss = torch.max(vf_loss1, vf_loss2)
        mean_vf_loss = reduce_mean_valid(vf_loss)
        total_loss = reduce_mean_valid(
            -surrogate_loss + policy.kl_coeff * action_kl +
            policy.config["vf_loss_coeff"] * vf_loss -
            policy.entropy_coeff * curr_entropy)
    else:
        mean_vf_loss = 0.0
        total_loss = reduce_mean_valid(-surrogate_loss +
                                       policy.kl_coeff * action_kl -
                                       policy.entropy_coeff * curr_entropy)

    # Store stats in policy for stats_fn.
    policy._total_loss = total_loss
    policy._mean_policy_loss = mean_policy_loss
    policy._mean_vf_loss = mean_vf_loss
    policy._vf_explained_var = explained_variance(
        train_batch[Postprocessing.VALUE_TARGETS],
        policy.model.value_function())
    policy._mean_entropy = mean_entropy
    policy._mean_kl = mean_kl

    return total_loss


def custom_before_loss_init(policy: Policy, obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 config: TrainerConfigDict) -> None:
    # default ppo process
    from ray.rllib.agents.ppo.ppo_torch_policy import setup_mixins
    setup_mixins(policy, obs_space, action_space, config)

    # load guidance policy
    ckpt_path = config['guidance_ckpt_path']
    policy.guidance = ExportedPolicy(ckpt_path)
    policy.guidance.model.cuda()

    policy.guidance.guidance_criterion = config['guidance_criterion']
    policy.guidance.guidance_coef = config['guidance_coef']

    # make sure no obs filter otherwise may have conflict with obs filter in guidance policy
    assert config['observation_filter'] == 'NoFilter'


def custom_extra_action_out_fn(
        policy: Policy, input_dict: Dict[str, TensorType],
        state_batches: List[TensorType], model: ModelV2,
        action_dist: TorchDistributionWrapper) -> Dict[str, TensorType]:
    # default ppo process
    from ray.rllib.agents.ppo.ppo_torch_policy import vf_preds_fetches
    out = vf_preds_fetches(policy, input_dict, state_batches, model, action_dist)

    return out


policy_cls = PPOTorchPolicy
DEFAULT_CONFIG['guidance_ckpt_path'] = None
DEFAULT_CONFIG['guidance_criterion'] = 'simple_kl'
DEFAULT_CONFIG['guidance_coef'] = 1.0
GuidedPPOPolicy = policy_cls.with_updates(
    loss_fn=custom_loss_fn, 
    before_loss_init=custom_before_loss_init,
    extra_action_out_fn=custom_extra_action_out_fn)
GuidedPPOTrainer = PPOTrainer.with_updates(
    name="GuidedPPO", 
    get_policy_class=lambda _: GuidedPPOPolicy,
    default_config=DEFAULT_CONFIG)