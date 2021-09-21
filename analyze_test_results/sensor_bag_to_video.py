import os
from re import I
import time
import argparse

from numpy.core.numeric import outer
import rosbag
from skvideo.io import FFmpegWriter
from scipy.interpolate import interp1d
from shapely.geometry import box as Box
from shapely import affinity
from descartes import PolygonPatch
from matplotlib import cm, patches
import cv2

from utils import *
from vista.core.Display import fig2img


def main():
    # Parse arguments and config
    parser = argparse.ArgumentParser(description='Convert event bag to video')
    parser.add_argument('--bag-path',
                        type=str,
                        default=None,
                        required=True,
                        help='Path to event camera data bag')
    parser.add_argument('--devens-road-path',
                        type=str,
                        default=None,
                        help='Path to the pickle file that stores Devens road')
    args = parser.parse_args()
    args.bag_path = validate_path(args.bag_path)

    car_width = 2
    car_length = 5
    car_color = list(list(cm.get_cmap('Set1').colors)[0]) + [0.6]

    # Read bag
    data, topic_info = read_rosbag(args.bag_path, return_topic_info=True)
    gps = fetch_gps(data)
    yaws = fetch_yaw(data)
    yaw_f = interp1d(yaws[:,0], yaws[:,1], fill_value='extrapolate')

    # Init video
    subsample_factor = 10
    fps = topic_info['/lexus/oxts/gps/fix'].frequency / float(subsample_factor)
    video_path = '.'.join(args.bag_path.split('.')[:-1] + ['mp4'])
    video_writer = FFmpegWriter(video_path, inputdict={'-r': str(fps)}, outputdict={'-r': str(fps), '-c:v': 'libx264'})

    # Read data for top-down view (sensor bag / devens road)
    fig, ax = plt.subplots(1, 1)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    ax.set_xticks([])
    ax.set_yticks([])

    if args.devens_road_path is not None:
        args.devens_road_path = validate_path(args.devens_road_path)
        devens_road = load_devens_road(args.devens_road_path)
        inner_roads = {k: v for k, v in devens_road.items() if 'inner' in k}
        outer_roads = {k: v for k, v in devens_road.items() if 'inner' not in k}

    # Plot car moving
    artists = dict()
    for i, (t, x, y) in tqdm(enumerate(gps), total=gps.shape[0]):
        try:
            if i % subsample_factor != 0:
                continue

            # compute top-down view origin based on 
            xy = np.array([x, y])

            use_road_center_as_origin = False
            if use_road_center_as_origin: # NOTE: will cause flickring
                inner_dist = [np.linalg.norm(r - xy, axis=1) for n, r in inner_roads.items()]
                inner_road_idx = np.argmin([np.min(v) for v in inner_dist])
                inner_frame_idx = np.argmin(inner_dist[inner_road_idx])
                inner_closest = list(inner_roads.values())[inner_road_idx][inner_frame_idx]

                outer_dist = [np.linalg.norm(r - xy, axis=1) for n, r in outer_roads.items()]
                outer_road_idx = np.argmin([np.min(v) for v in outer_dist])
                outer_frame_idx = np.argmin(outer_dist[outer_road_idx])
                outer_closest = list(outer_roads.values())[outer_road_idx][outer_frame_idx]

                origin = (inner_closest + outer_closest) / 2.
            else:
                origin = xy

            # get car's poly
            yaw = yaw_f(t)
            cos = np.cos(-yaw)
            sin = np.sin(-yaw)
            R = np.array([[cos, -sin], [sin, cos]])

            rotated_xy = np.matmul(R, (xy - origin))
            poly = Box(rotated_xy[0] - car_width / 2., rotated_xy[1] - car_length / 2.,
                       rotated_xy[0] + car_width / 2., rotated_xy[1] + car_length / 2.)
            patch = PolygonPatch(poly, fc=car_color, ec=car_color, zorder=2)
            update_patch(ax, artists, 'patch:car', patch)

            # plot road
            rotated_devens_road = dict()
            for name, road in devens_road.items():
                rotated_devens_road[name] = np.matmul(road - origin, R.T)
            plot_devens_road(rotated_devens_road, [fig, ax], linewidth=2, color='w', buffer=1., lns=artists)

            ax.set_xlim(-30, +30)
            ax.set_ylim(-30, +30)
            
            # save to video
            img = fig2img(fig)
            video_writer.writeFrame(img)
        except KeyboardInterrupt:
            video_writer.close()
    video_writer.close()

    
def update_patch(ax, artists, name, patch):
    if name in artists.keys():
        artists[name].remove()
    ax.add_patch(patch)
    artists[name] = patch


if __name__ == '__main__':
    main()