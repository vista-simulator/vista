import os
import time
import argparse
import rosbag
from skvideo.io import FFmpegWriter

from sensor_msgs.msg import PointCloud2
import ros_numpy
import sensor_msgs.point_cloud2 as pc2

from utils import *
from vista.entities.sensors.lidar_utils import Pointcloud
from vista.core.Display import plot_pointcloud, fig2img


def main():
    # Parse arguments and config
    parser = argparse.ArgumentParser(description='Convert event bag to video')
    parser.add_argument('--bag-path',
                        type=str,
                        default=None,
                        required=True,
                        help='Path to event camera data bag')
    args = parser.parse_args()
    args.bag_path = validate_path(args.bag_path)

    # Init plot and set config
    car_width = 2
    car_length = 5

    fig, ax = plt.subplots(1, 1, figsize=(10,10))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    ax.set_xticks([])
    ax.set_yticks([])
    scat = None

    # Convert
    bag = rosbag.Bag(args.bag_path)
    topic_info = bag.get_type_and_topic_info()[1]

    fps = topic_info['/lexus/velodyne_points'].frequency
    video_path = '.'.join(args.bag_path.split('.')[:-1] + ['mp4'])
    video_writer = FFmpegWriter(video_path, inputdict={'-r': str(fps)},
                                outputdict={'-r': str(fps),
                                            '-c:v': 'libx264',
                                            '-pix_fmt': 'yuv420p'})

    total_msgs = bag.get_message_count(['/lexus/velodyne_points'])
    for topic, msg, t in tqdm(bag.read_messages(topics=['/lexus/velodyne_points']), total=total_msgs):
        try:
            msg.__class__ = PointCloud2
            pc = ros_numpy.numpify(msg)
            xyz = np.stack([pc['x'], pc['y'], pc['z']], axis=1)
            pcd = Pointcloud(xyz, pc['intensity'])
            
            ax, scat = plot_pointcloud(pcd[::4], # subsample for vis
                                       ax=ax,
                                       color_by="z",
                                       max_dist=30.,
                                       car_dims=None, # don't draw car
                                       cmap="nipy_spectral",
                                       scat=scat,
                                       s=4)
            frame = fig2img(fig)
            video_writer.writeFrame(frame) # NOTE: flip x, y due to video writer
        except KeyboardInterrupt:
            video_writer.close()
    video_writer.close()


if __name__ == '__main__':
    main()