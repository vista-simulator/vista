from typing import *

from .Trace import *
from ..entities.agents import *


class World:
    def __init__(self, trace_paths: List[str]) -> None:

        # A list of traces that define the world
        self.traces = [Trace(trace_path) for trace_path in trace_paths]

        # A list of agents within this world. Agents start in a single trace
        # but can be teleported between traces, since they are all in the
        # same world.
        self.agents = []

    def spawn_agent(self) -> Car:
        agent = Car(world=self)
        self.agents.append(agent)
        return agent

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def sample_new_location(self):
        new_trace_index = self.sample_new_trace_index()
        trace = self.traces[new_trace_index]

        # Assume every trace only has one segment (need to fix this and split traces with > 1 segment)
        current_segment_index = trace.find_segment_reset()
        curv_reset_probs = trace.get_curv_reset_probs(
            current_segment_index)  # Get new curv reset probs for it

        # Reset favoring places with higher curvatures
        new_frame_index = trace.find_curvature_reset(curv_reset_probs)

        new_frame_index = 220 # DEBUG

        return new_trace_index, current_segment_index, new_frame_index

    def sample_new_trace_index(self):
        trace_reset_probs = np.zeros(len(self.traces))
        for i, trace in enumerate(self.traces):
            trace_reset_probs[i] = trace.num_of_frames
        trace_reset_probs /= np.sum(trace_reset_probs)
        new_trace = np.random.choice(trace_reset_probs.shape[0],
                                     size=1,
                                     p=trace_reset_probs)
        return new_trace[0]
