.. _advanced_usage-guided_policy_learning:

Guided Policy Learning
======================

Here we demonstrate how to leverage the power of local synthesis around a passive dataset using
VISTA, which leads to a learning framework called guided policy learning (GPL). In contrast to
imitation learning (IL), GPL actively samples sensor data that is around but different from the original
dataset and couples it with control commands (labels in supervised imitation learning) that aims to
correct the agent to the nominal human trajectory in the original passive dataset. Overall, GPL can
be seen as a data augmentation version of IL that tries to improve robustness of the model within
some deviation of demonstration (the passive dataset).

Similar to IL, we first initialize VISTA simulator and a sampler. There are two major differences
during reset. First, we always initialize the ego agent some distance away from the human trajectory
(demonstration) to actively create scenarios to be corrected from. This is specified by ``initial_dynamics_fn``.
Second, we need a controller that provides ground truth control commands to correct toward the
demonstration. The controller is allowed to have access to privileged information (e.g., lane
boundaries, ado cars' poses, etc) as it is only used to provide guidance for the policy learning
during training time. ::

    self._world = vista.World(self.trace_paths, self.trace_config)
    self._agent = self._world.spawn_agent(self.car_config)
    self._camera = self._agent.spawn_camera(self.camera_config)
    self._world.reset({self._agent.id: self.initial_dynamics_fn})
    self._sampler = RejectionSampler()

    self._privileged_controller = get_controller(self.privileged_control_config)

Next, we implement a data generator that produces a training dataset "around" the original passive
dataset used for imitation learning. This encourages the policy to correct itself toward the
demonstration (human trajectories). ::

    # Data generator from simulation
    self._snippet_i = 0
    while True:
        # reset simulator
        if self._agent.done or self._snippet_i >= self.snippet_size:
            self._world.reset({self._agent.id: self.initial_dynamics_fn})
            self._snippet_i = 0

        # privileged control
        curvature, speed = self._privileged_controller(self._agent)

        # step simulator
        sensor_name = self._camera.name
        img = self._agent.observations[sensor_name] # associate action t with observation t-1
        action = np.array([curvature, speed])
        self._agent.step_dynamics(action)

        val = curvature
        sampling_prob = self._sampler.get_sampling_probability(val)
        if self._rng.uniform(0., 1.) > sampling_prob: # reject
            self._snippet_i += 1
            continue
        self._sampler.add_to_history(val)

        # preprocess and produce data-label pairs
        img = transform_rgb(img, self._camera, self.train)
        label = np.array([curvature]).astype(np.float32)

        self._snippet_i += 1

        yield {'camera': img, 'target': label}

As shown above, the only difference from imitation learning is to run a privileged controller that
produces the correct control commands for states deviated from the human trajectories. Thus, we get
data of the agent initially deviating away from the demonstration but gradually converging to the human
trajectories. At test time with closed-loop control settings, such a training scheme allows the policy
to correct itself from drifting away due to compounding error. For more details, please check
``examples/advanced_usage/gpl_rgb_dataset.py``.
