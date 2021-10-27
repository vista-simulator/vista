from typing import Dict, Optional, Any

from ..Entity import Entity
from ...utils import misc


class BaseSensor(Entity):
    """ Base class of all sensors.

    Args:
        attach_to (Entity): A car to be attached to.
        config (dict): Configuration of the sensor.

    """
    DEFAULT_CONFIG = {}

    def __init__(self,
                 attach_to: Entity,
                 config: Optional[Dict] = None) -> None:
        super(BaseSensor, self).__init__()

        self._parent = attach_to
        config = misc.merge_dict(config, self.DEFAULT_CONFIG)
        self._config = config
        self._name = config['name']

    def capture(self, timestamp: float, **kwargs) -> Any:
        """ Run sensor synthesis based on current timestamp and transformation
        between the novel viewpoint to be simulated and the nominal viewpoint from
        the pre-collected dataset.

        Args:
            timestamp (float): Timestamp that allows to retrieve a pointer to
                the dataset for data-driven simulation (synthesizing point cloud
                from real LiDAR sweep).

        """
        raise NotImplementedError

    def update_scene_object(self, name: str, scene_object: Any,
                            pose: Any) -> None:
        """ Update object in the scene for rendering. This is only used
        when we put virtual objects in the scene.

        Args:
            name (str): Name of the scene object.
            scene_object (Any): The scene object.
            pose (Any): The pose of the scene object.

        """
        raise NotImplementedError

    @property
    def name(self) -> str:
        """ The name of the sensor. """
        return self._name
