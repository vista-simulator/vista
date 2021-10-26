.. _api_documentation-tasks:

Tasks
=====

This module demonstrates how to build environments of various tasks with Vista.
We follow roughly OpenAI gym interface for reinforcement learning setting here (with
member functions of the environments like ``reset``, ``step`` and return values like
``observation``, ``reward``, ``done``, ``info``).

.. autoclass:: vista.tasks.lane_following.LaneFollowing
    :members:
    :inherited-members:

.. autoclass:: vista.tasks.multi_agent_base.MultiAgentBase
    :members:
    :inherited-members:
