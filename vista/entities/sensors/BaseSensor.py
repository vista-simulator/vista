from abc import abstractmethod

from ..Entity import Entity


class BaseSensor(Entity):
    def __init__(self) -> None:
        super(BaseSensor, self).__init__()
