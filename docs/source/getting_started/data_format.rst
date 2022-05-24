.. _getting_started-data_format:

Data Format
===========

Vista simulation requires two sets of data: **sensor data** and **odometry data**. Sensor data
contains a timestamp file and a sensor data file, and odometry data contains minimal information
to step vehicle dynamics. Every data file has their corresponding ROS timestamp, which allows to be
associated across each other for data-driven simulation.

Sensor Data
-----------

The general concept of data-driven simulation in Vista is to synthesize locally around precollected
sensor data. In order to achieve that, we need to assoicate current states of the simulator to some
point in the dataset. We use ROS timestamp as an universal identifier across different modalities of
data. Given a vehicle state associated with some ROS timestamp, we need to retrieve from the dataset
a piece of sensor data (e.g., a RGB frame or a LiDAR sweep) for synthesis. Thus, sensor data contains
two types of data with file name as sensor name specified in Vista,

    * A csv file that maps indices of the sensor (e.g., frame number for RGB camera or sweep number
      for LiDAR) to an universal ROS timestamp that can be shared across different sensors.
    * A major sensor data file contains the actual sensor data. This file may have different format
      across different sensors, e.g., <camera-namne>.avi for RGB camera and <lidar-name>.h5 for LiDAR.

:ref:`MultiSensor <api_multi_sensor>` is a class that handles sensor data retrieval for all sensors.
It handles the mapping between the universal (ROS) timestamp shared throughout the entire simulator
and sensor-specific pointer to data stream of each sensor. Also note that the ``master_sensor`` in
this class provide a reference timestamp for stepping vehicle dynamics.

Odometry Data
-------------

Vista depends on timestamped IMU and speed data to step vehicle dynamics. At a high level,
we extract speed feedback and IMU data (or more specifically, yaw rate) and feed
them into vehicle dynamics model, as shown in :ref:`Trace <api_trace>`. ::

    from scipy.interpolate import interp1d

    def _get_states_func(self):
        # Read from dataset`
        speed = np.genfromtxt(os.path.join(self._trace_path,
            TopicNames.speed + '.csv'), delimiter=',')
        imu = np.genfromtxt(os.path.join(self._trace_path,
            TopicNames.imu + '.csv'), delimiter=',')

        # Obtain function representation of speed
        f_speed = interp1d(speed[:, 0], speed[:, 1], fill_value='extrapolate')

        # Obtain function representation of curvature
        timestamps = imu[:, 0]
        yaw_rate = imu[:, 6]
        curvature = yaw_rate / np.maximum(f_speed(timestamps), 1e-10)
        good_curvature_inds = np.abs(curvature) < 1 / 3.
        f_curvature = interp1d(timestamps[good_curvature_inds],
            curvature[good_curvature_inds], fill_value='extrapolate')
        
        return f_speed, f_curvature

Basically, the odometry data should contain minimal information to trace out human trajectories in
the dataset. In our case, we are using ``curvature`` (equivalent information to steering command)
and ``speed``, which is required in stepping the vehicle dynamics (bicycle model). Note that we are
using control command feedback instead of control command to eliminate the effect of imperfect low-
level controller, tire slippage, and etc. Another thing to remark is that in some use cases, we have
access to more accurate localization like differential GPS, which can provide better estimate of 
vehicle state. Overall, there are several choices that can be used as odometry data with different
accessibility and accuracy.
