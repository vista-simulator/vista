from typing import List, Dict, Tuple, Optional, Callable
import numpy as np

from .Trace import Trace
from ..entities.agents import Car
from ..utils import logging


class World:
    """ This class specifies the :class:`World` where all entities (i.e., agents and sensors) 
    in VISTA lives in. The is built upon one or multiple traces pre-collected from the real world. 
    Users can spawn agents (:class:`Car`) in the :class:`World`, where all agents will be visible 
    while rendering sensory measurement. At reset, the :class:`World` randomly sample an pointer
    in the dataset (conceptually retrieving a frame within a trace) as the initial condition for
    the simulation.

    Args:
        trace_paths (List(str)): A list of paths to traces
        trace_config (Dict): Configuration of traces

    Example usage::

        >>> world = vista.World(trace_paths, trace_config)
        >>> agent = world.spawn_agent(agent_config)
        >>> world.reset()

    """
    def __init__(
        self, trace_paths: List[str],
        trace_config: Optional[Dict] = dict()) -> None:
        # A list of traces that define the world
        self._traces: List[Trace] = [
            Trace(trace_path, trace_config) for trace_path in trace_paths
        ]

        # A list of agents within this world. Agents start in a single
        # trace but can be teleported between traces, since they are all in
        # the same world.
        self._agents: List[Car] = []

        # Random seed; will also assign the same seed to traces
        self.set_seed(0)

    def spawn_agent(self, config: Dict) -> Car:
        """ Spawn an agent in this world.

        Args:
            config (Dict): Configuration of the agent.

        Returns:
            Car: The agent being spawned.

        """
        agent = Car(world=self, car_config=config)
        logging.info('Spawn an agent ({})'.format(agent.id))
        self._agents.append(agent)

        return agent

    def reset(
        self, initial_dynamics_fn: Optional[Dict[str,
                                                 Callable]] = dict()) -> None:
        """ Reset the world. This includes (1) sample a new anchor point from the real-world
        dataset to be simulated from and (2) reset states for all agents.

        Args:
            initial_dynamics_fn (Dict[str, Callable]):
                A dict mapping agent names to a function that initialize agents poses.

        """
        logging.info('World reset')

        # Sample a new trace and a new location at the sampled trace
        new_trace_index, new_segment_index, new_frame_index = \
            self.sample_new_location()

        # Reset agents
        for agent in self.agents:
            agent.reset(new_trace_index, new_segment_index, new_frame_index,
                        initial_dynamics_fn.get(agent.id, None))

    def sample_new_location(self) -> Tuple[int, int, int]:
        """ Sample a pointer to the dataset for simulation.

        Returns:
            Return a tuple (``int_a``, ``int_b``, ``int_c``), where ``int_a`` is trace index,
            ``int_b`` is segment index, and ``int_c`` is frame index.

        """
        new_trace_index = self.sample_new_trace_index()
        trace = self.traces[new_trace_index]

        new_segment_index = trace.find_segment_reset()

        # note that this is the index of Trace.good_frames instead of element
        new_frame_index = trace.find_frame_reset(new_segment_index)

        return new_trace_index, new_segment_index, new_frame_index

    def sample_new_trace_index(self) -> int:
        """ Sample a new trace index based on number of frames in a trace.

        Returns:
            int: An index to specify which trace to be simulated from.

        """
        trace_reset_probs = np.zeros(len(self.traces))
        for i, trace in enumerate(self.traces):
            trace_reset_probs[i] = trace.num_of_frames
        trace_reset_probs /= np.sum(trace_reset_probs)

        new_trace_index = self._rng.choice(trace_reset_probs.shape[0],
                                           p=trace_reset_probs)

        return new_trace_index

    def set_seed(self, seed) -> None:
        """ Set random seed.

        Args:
            seed (int): Random seed.

        """
        self._seed = seed
        self._rng = np.random.default_rng(self.seed)
        [t.set_seed(seed) for t in self.traces]

    @property
    def seed(self) -> int:
        """ Random seed for sampling pointer to the dataset during :meth:`reset`. """
        return self._seed

    @property
    def traces(self) -> List[Trace]:
        """ All trace attached to the :class:`World`. """
        return self._traces

    @property
    def agents(self) -> List[Car]:
        """ All agents that live in the :class:`World`. """
        return self._agents

    def __repr__(self) -> str:
        return '<{}> '.format(self.__class__.__name__) + \
               '#traces: {} '.format(len(self.traces)) + \
               '#agents: {} '.format(len(self.agents))
