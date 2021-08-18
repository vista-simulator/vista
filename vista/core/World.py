from typing import List, Dict, Tuple, Optional
import numpy as np

from .Trace import Trace
from ..entities.agents import Car
from ..utils import logging


class World:
    def __init__(
        self, trace_paths: List[str],
        trace_config: Optional[Dict] = dict()) -> None:
        """ Instantiate a World object.

        Args:
            trace_paths (List(str)): a list of paths to traces
            trace_config (Dict): configuration of traces
        """
        # A list of traces that define the world
        self._traces: List[Trace] = [
            Trace(trace_path, trace_config) for trace_path in trace_paths
        ]

        # A list of agents within this world. Agents start in a single
        # trace but can be teleported between traces, since they are all in
        # the same world.
        self._agents: List[Car] = []

    def spawn_agent(self, config: Dict) -> Car:
        """ Spawn an agent in this world.

        Args:
            config (Dict): configuration of the agent

        Returns:
            Car: the agent being spawned
        """
        agent = Car(world=self, car_config=config)
        logging.info('Spawn an agent ({})'.format(agent.id))
        self._agents.append(agent)

        return agent

    def reset(self) -> None:
        logging.info('World reset')

        # Sample a new trace and a new location at the sampled trace
        new_trace_index, new_segment_index, new_frame_index = \
            self.sample_new_location()

        # Reset agents
        for agent in self.agents:
            agent.reset(new_trace_index, new_segment_index, new_frame_index)

    def sample_new_location(self) -> Tuple[int, int, int]:
        """ Sample a pointer to the dataset for simulation.

        Args:
            None

        Returns:
            int: trace index
            int: segment index
            int: frame index (note that this is the index of Trace.good_frames
                 instead of element)
        """
        new_trace_index = self.sample_new_trace_index()
        new_trace_index = 0 # DEBUG
        trace = self.traces[new_trace_index]

        new_segment_index = trace.find_segment_reset()
        new_segment_index = 0 # DEBUG

        new_frame_index = trace.find_frame_reset(new_segment_index)
        new_frame_index = 100 # DEBUG

        return new_trace_index, new_segment_index, new_frame_index

    def sample_new_trace_index(self) -> int:
        """ Sample a new trace index based on number of frames in a trace.

        Args:
            None

        Returns:
            int: an index for which trace to be simulated from
        """
        trace_reset_probs = np.zeros(len(self.traces))
        for i, trace in enumerate(self.traces):
            trace_reset_probs[i] = trace.num_of_frames
        trace_reset_probs /= np.sum(trace_reset_probs)

        new_trace_index = np.random.choice(trace_reset_probs.shape[0],
                                           p=trace_reset_probs)

        return new_trace_index

    @property
    def traces(self) -> List[Trace]:
        return self._traces

    @property
    def agents(self) -> List[Car]:
        return self._agents

    def __repr__(self) -> str:
        return '<{}> '.format(self.__class__.__name__) + \
               '#traces: {} '.format(len(self.traces)) + \
               '#agents: {} '.format(len(self.agents))
