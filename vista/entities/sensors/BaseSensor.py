from typing import Dict, Optional, Any

from ..Entity import Entity


class BaseSensor(Entity):
    def __init__(self,
                 attach_to: Entity,
                 config: Optional[Dict] = None) -> None:
        super(BaseSensor, self).__init__()

        self._parent = attach_to
        self._config = config
        self._name = config['name']

    def capture(self, timestamp: float) -> Any:
        raise NotImplementedError

    def update_scene_object(self, name: str, scene_object: Any,
                            pose: Any) -> None:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self._name
