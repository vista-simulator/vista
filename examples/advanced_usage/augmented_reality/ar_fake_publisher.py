""" Simulate playing a rosbag and a timestamped video. """
import os
import time
import argparse
import numpy as np
import rospy
import rosbag
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, Imu, NavSatFix
from ffio import FFReader


class RosbagVideoPublisher(object):
    def __init__(self, bag_path, video_path, video_timestamp_path):
        self._bag = rosbag.Bag(bag_path)
        self._topics = ['/lexus/oxts/gps/fix', '/lexus/oxts/imu/data']
        self._pubs = dict()
        topic_infos = self._bag.get_type_and_topic_info()[1]
        for topic in self._topics:
            topic_info = topic_infos[topic]
            topic_type = globals()[topic_info[0].split('/')[-1]]
            pub = rospy.Publisher(topic, topic_type, queue_size=1)
            self._pubs[topic] = pub

        self._stream = FFReader(video_path)
        self._timestamps = np.genfromtxt(video_timestamp_path,
                                         delimiter=',',
                                         skip_header=1,
                                         dtype=np.float64)
        self._stream.read()
        self._camera_name = video_path.split('/')[-1].split('.')[0]
        self._pub_video = rospy.Publisher(
            f'camera_array/{self._camera_name}/image_raw', Image, queue_size=1)
        self._cvbr = CvBridge()

    def publish(self):
        prev_t = None
        for topic, msg, t in self._bag.read_messages(topics=self._topics):
            tic = time.time()
            self._pubs[topic].publish(msg)
            if prev_t is not None:
                time_gap = t.to_sec() - prev_t.to_sec()
            else:
                time_gap = 0.
            prev_t = t
            if t.to_sec() >= self._timestamps[int(self._stream.frame_num), 1]:
                msg_img = self._cvbr.cv2_to_imgmsg(self._stream.image)
                msg_img.header.seq = self._stream.frame_num
                msg_img.header.stamp.secs = int(
                    self._timestamps[int(self._stream.frame_num), 1])
                msg_img.header.stamp.nsecs = int(
                    1e9 * (self._timestamps[int(self._stream.frame_num), 1] -
                           msg_img.header.stamp.secs))
                msg_img.header.frame_id = self._camera_name
                self._pub_video.publish(msg_img)
                self._stream.read()
            toc = time.time()
            idle_time = max(0, time_gap - (toc - tic))
            time.sleep(idle_time)


def main(args):
    node_name = 'vista_ar_fake_publisher_node'
    rospy.init_node(node_name, anonymous=False)
    if args.video_timestamp_path is None:
        args.video_timestamp_path = os.path.splitext(
            args.video_path)[0] + '.csv'
    rosbag_video_publisher = RosbagVideoPublisher(args.bag_path,
                                                  args.video_path,
                                                  args.video_timestamp_path)
    rosbag_video_publisher.publish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create fake publisher for Vista')
    parser.add_argument('--bag-path',
                        type=str,
                        required=True,
                        help='Path to rosbag')
    parser.add_argument('--video-path',
                        type=str,
                        required=True,
                        help='Path to video')
    parser.add_argument(
        '--video-timestamp-path',
        type=str,
        default=None,
        help='Path to timestamp of the video;' +
        'default to the same path to the video but with .csv extension')
    args = parser.parse_args()
    main(args)
