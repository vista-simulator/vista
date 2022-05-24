.. _advanced_usage-imitation_learning:

Imitation Learning
==================

Here we show an example of using VISTA to learn a neural-network-based policy with the most basic
variant of imitation learning (i.e., behavior cloning). Since sensor data synthesis is not required
in imitation learning, we show how to extract a passive dataset upon which the data-driven simulation
is built with VISTA's interface. The high-level idea is to construct a world containing an ego-agent
in VISTA and run the agent with human control command in the passive dataset without doing sensor
data synthesis to avoid redundant computation.

First, we initialize VISTA world with an agent attached with a sensor (we use RGB camera here for
illustration). Note that it's often useful to implement an additional sampler since balanced training
data distribution is often of great significance for supervised learning. ::

    self._world = vista.World(self.trace_paths, self.trace_config)
    self._agent = self._world.spawn_agent(self.car_config)
    self._camera = self._agent.spawn_camera(self.camera_config)
    self._world.reset()
    self._sampler = RejectionSampler() # data sampler

Then, we can implement a data generator that runs indefinitely to produce a training dataset. ::

    # Data generator from simulation
    self._snippet_i = 0
    while True:
        # reset simulator
        if self._agent.done or self._snippet_i >= self.snippet_size:
            self._world.reset()
            self._snippet_i = 0

        # step simulator
        sensor_name = self._camera.name
        img = self._agent.observations[sensor_name] # associate action t with observation t-1
        self._agent.step_dataset(step_dynamics=False)

        # rejection sampling
        val = self._agent.human_curvature
        sampling_prob = self._sampler.get_sampling_probability(val)
        if self._rng.uniform(0., 1.) > sampling_prob:
            self._snippet_i += 1
            continue
        self._sampler.add_to_history(val)

        # preprocess and produce data-label pairs
        img = transform_rgb(img, self._camera, self.train)
        label = np.array([self._agent.human_curvature]).astype(np.float32)

        self._snippet_i += 1

        yield {'camera': img, 'target': label}

The implementation is straightforward. After resetting the simulator (the pointer to the passive
dataset is randomly initialized), we step through the dataset to get the next frame by calling
``agent.step_dataset``, followed by a rejection sampling to balance the steering control command
(``human_curvature``). Finally, we preprocess sensor data and construct data-label pairs for training.
Note that we usually set a maximum snippet size to make sure that each snippet, a series of data
from the start (a reset) to the termination (another reset), won't last indefinitely and training
data can have sufficient diversity. Also, to ensure the i.i.d. data distribution that is required for
stochastic gradient descent, the data stream (``yield {'camera': img, 'target': label}``
) is connected to a buffer with shuffling. For more details, please check ``examples/advanced_usage/il_rgb_dataset.py``.
