VISTA.. _advanced_usage-reinforcement_learning:

Reinforcement Learning
======================

Based on what we discussed in the previous section for guided policy learning where we actively
generate data with off-human-trajectory initialization and a privileged controller for correction,
we can consider the further extreme of exploiting the "activeness" of applying VISTA on a passive
dataset. This leads to reinforcement learning (RL), where an agent is allowed to interact with the
environment, collect training data itself, and learn to achieve good task performance. Here we show
how to use VISTA to define a RL environment for learning a policy.

Implementation-wise, there are four major components to be defined, namely the observations, environment
dynamics, reward function, and terminal condition. Observations are simply sensor data and environment
dynamics is how the ego car moves after applying some control commands, which is already embedded in
the vehicle state update (and collision if multi-agent scenario is considered). Thus, we need to further
define the reward function and terminal conditions. For example, if we consider lane following, ::

    def step(self, action, dt=1 / 30.):
        # Step agent and get observation
        agent = self._world.agents[0]
        action = np.array([action[agent.id][0], agent.human_speed])
        agent.step_dynamics(action, dt=dt)
        agent.step_sensors()
        observations = agent.observations

        # Define terminal condition
        done, info_from_terminal_condition = self.config['terminal_condition'](
            self, agent.id)

        # Define reward
        reward, _ = self.config['reward_fn'](self, agent.id,
                                             **info_from_terminal_condition)

        # Get info
        # ...

        # Pack output
        observations, reward, done, info = map(
            self._append_agent_id, [observations, reward, done, info])

        return observations, reward, done, info

, where ``self.config['terminal_condition']`` can be defined as when the car is off the lane (too
far away from the lane center) or the car heading deviates too much from the road curvature. Note that apart
from being the terminal condition for the lane following task, the above-mentioned two constraints
should be satisfied since VISTA only allows for high-fidelity synthesis locally around the original
passive dataset. ::

    def default_terminal_condition(task, agent_id, **kwargs):
        agent = [_a for _a in task.world.agents if _a.id == agent_id][0]

        def _check_out_of_lane():
            road_half_width = agent.trace.road_width / 2.
            return np.abs(agent.relative_state.x) > road_half_width

        def _check_exceed_max_rot():
            maximal_rotation = np.pi / 10.
            return np.abs(agent.relative_state.theta) > maximal_rotation

        out_of_lane = _check_out_of_lane()
        exceed_max_rot = _check_exceed_max_rot()
        done = out_of_lane or exceed_max_rot or agent.done
        other_info = {
            'done': done,
            'out_of_lane': out_of_lane,
            'exceed_max_rot': exceed_max_rot,
        }

        return done, other_info

We can define a very simple reward function that encourages survival (not going off the lane or
exceeding some rotation with respect to the road curvature) by simply checking whether the current
step is terminated. ::

    def default_reward_fn(task, agent_id, **kwargs):
        """ An example definition of reward function. """
        reward = -1 if kwargs['done'] else 0  # simply encourage survival

        return reward, {}

Please check :ref:`lane_following.py <api_lane_following>` for more details. The implementation
roughly follows `OpenAI Gym <https://gym.openai.com/>`_ interface with ``reset`` and ``step`` functions.
However, there are still other attributes or functions to be implemented like ``action_space``,
``observation_space``, ``render``, etc, which may require objects from ``gym``.
