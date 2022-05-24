class TopicNames:
    """ This class specifies (ROS) topic names of each sensors.
    """
    camera_front = 'camera_front'
    camera_wing_left = 'camera_wing_left'
    camera_wing_right = 'camera_wing_right'
    camera_left = 'camera_left'
    camera_right = 'camera_right'
    cameras = [
        camera_front, camera_wing_right, camera_wing_right, camera_left,
        camera_right
    ]

    steering = 'steering_can'
    imu = 'imu'
    odometry = 'odom'
    gps = 'gps'
    gps_heading = 'gps_heading'
    speed = 'speed'
    distance = 'distance'
    lidar_3d = 'lidar_3d'

    BADVAL = ['BADVAL']  # for denoting topics we don't know yet

    # Master sensor to align all others with:
    master_topic = camera_front
