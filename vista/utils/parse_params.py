from typing import Optional, List, Tuple
import os
import numpy as np
import xml.etree.ElementTree as ET
import pathlib


def ignore_case(tree):
    t = ET.tostring(tree)
    t = t.lower()
    return ET.fromstring(t)


class ParamsFile:
    """ Helper object for processing a dataset parameters file. Every trace
    has a dedicated parameters file (``params.xml``) stored within it which
    contains calibration and other information on each of the sensors so we can
    reconstruct and simulate novel views.

    Args:
        path (str): Path to params.xml that specifies sensor parameters.

    """
    def __init__(self, path: str):
        self.path = path
        tree = ET.parse(self.path)
        self.root = ignore_case(tree.getroot())

    def parse_camera(self, name: str = None):
        """ Returns a parsed dictionary of all keys/values corresponding to the
        camera sensor with specified ``name``. If no ``name`` is given, then
        the first camera in the params file is used.

        Args:
            name (str): name of the camera (e.g. ``camera_front``). If not
            provided, then return the first camera.

        Returns:
            (tuple): tuple containing:
                (dict) of key -> value pairs for the specified camera sensor.
                (str): name of the sensor (useful if ``name`` was not specied
                as input)
        """
        return self._parse_sensor(sensor="camera", name=name)

    def parse_lidar(self, name: str = None):
        """ Returns a parsed dictionary of all keys/values corresponding to the
        lidar sensor with specified ``name``. If no ``name`` is given, then
        the first lidar in the params file is used.

        Args:
            name (str): name of the lidar (e.g. ``lidar_3d``). If not
            provided, then return the first lidar.

        Returns:
            (tuple): tuple containing:
                (dict) of key -> value pairs for the specified lidar sensor.
                (str): name of the sensor (useful if ``name`` was not specied
                as input)
        """
        return self._parse_sensor(sensor="lidar", name=name)

    def _parse_sensor(self, sensor: str, name: str = None):
        xml_sensors = self.root.findall(f'sensors/{sensor}')
        names = [cam.get('name') for cam in xml_sensors]
        sensors = dict(zip(names, xml_sensors))

        if name:
            if name not in sensors.keys():
                raise ValueError(
                    f'{name} not a valid name in {self.path}. ' + \
                    f'Only found sensors ({names})')
        else:
            # default to the first camera
            # TODO: should default to the closest camera (most overlapping FoV)
            name = names[0]

        self.name = name
        sensor = sensors[self.name]
        xml_props = sensor.findall('property')

        pname = [p.get('name') for p in xml_props]
        pvalue = []
        for p in xml_props:
            value = p.get('value')
            if "," in value:
                value = np.array([v for v in value.split(",")])
            else:
                value = np.array(value)

            try:
                value = value.astype(np.float32)
                if value.size == 1:
                    value = float(value)
            except ValueError:
                if value.size == 1:
                    value = str(value)
                else:
                    value = list(value)

            pvalue.append(value)

        props = dict(zip(pname, pvalue))
        return props, name
