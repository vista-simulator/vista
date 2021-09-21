import os
import time
import argparse
import rosbag
from skvideo.io import FFmpegWriter

import genpy
import struct

import std_msgs.msg
from prophesee_event_msgs.msg import Event, EventArray

from utils import *
from vista.core.Display import events2frame


_struct_I = genpy.struct_I
def _get_struct_I():
    global _struct_I
    return _struct_I
_struct_2H = None
def _get_struct_2H():
    global _struct_2H
    if _struct_2H is None:
        _struct_2H = struct.Struct("<2H")
    return _struct_2H
_struct_2I = None
def _get_struct_2I():
    global _struct_2I
    if _struct_2I is None:
        _struct_2I = struct.Struct("<2I")
    return _struct_2I
_struct_3I = None
def _get_struct_3I():
    global _struct_3I
    if _struct_3I is None:
        _struct_3I = struct.Struct("<3I")
    return _struct_3I
_struct_B = None
def _get_struct_B():
    global _struct_B
    if _struct_B is None:
        _struct_B = struct.Struct("<B")
    return _struct_B


def parse_raw_message(raw_msg, mode=0):
    # Parse header
    data_str = raw_msg[1]
    header = std_msgs.msg.Header()
    end = 0
    start = end
    end += 12
    (header.seq, header.stamp.secs, header.stamp.nsecs,) = _get_struct_3I().unpack(data_str[start:end])
    start = end
    end += 4
    (length,) = _struct_I.unpack(data_str[start:end])
    start = end
    end += length
    header.frame_id = data_str[start:end].decode('utf-8', 'rosmsg')
    start = end
    end += 8
    (height, width,) = _get_struct_2I().unpack(data_str[start:end])
    start = end
    end += 4
    (length,) = _struct_I.unpack(data_str[start:end])
    # Parse events
    if mode == 0:
        start = end
        xy_bstr = b''.join([data_str[start + 13*i:start + 13*i + 4] for i in range(0, length, 1)])
        xys = np.frombuffer(xy_bstr, dtype='<2H')
        xs = xys[:,0]
        ys = xys[:,1]
        ps = np.frombuffer(data_str[start+12::13], dtype=np.bool)
        events = [xs, ys, ps]
    else: # slow
        xs, ys, ps = [], [], []
        for i in range(0, length):
            val1 = Event()
            _x = val1
            start = end
            end += 4
            (_x.x, _x.y,) = _get_struct_2H().unpack(data_str[start:end])
            _v4 = val1.ts
            _x = _v4
            start = end
            end += 8
            (_x.secs, _x.nsecs,) = _get_struct_2I().unpack(data_str[start:end])
            start = end
            end += 1
            (val1.polarity,) = _get_struct_B().unpack(data_str[start:end])
            val1.polarity = bool(val1.polarity)
            xs.append(val1.x)
            ys.append(val1.y)
            ps.append(val1.polarity)
        events = [xs, ys, ps]
    return header, (height, width), events


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

    # Convert
    bag = rosbag.Bag(args.bag_path)
    topic_info = bag.get_type_and_topic_info()[1]

    fps = topic_info['/prophesee/camera/cd_events_buffer'].frequency
    double_check = False
    video_path = '.'.join(args.bag_path.split('.')[:-1] + ['mp4'])
    video_writer = FFmpegWriter(video_path, inputdict={'-r': str(fps)},
                                outputdict={'-r': str(fps),
                                            '-c:v': 'libx264',
                                            '-pix_fmt': 'yuv420p'})

    total_msgs = bag.get_message_count(['/prophesee/camera/cd_events_buffer'])
    for topic, msg_raw, t in tqdm(bag.read_messages(topics=['/prophesee/camera/cd_events_buffer'], raw=True), total=total_msgs):
        try:
            header, (cam_h, cam_w), events = parse_raw_message(msg_raw)
            events = np.stack([events[1], events[0], np.zeros_like(events[2]), events[2]], axis=1)

            if double_check:
                _, _, events_check = parse_raw_message(msg_raw, mode=1)
                events_check = np.stack([events_check[1], events_check[0], 
                                         np.zeros_like(events_check[2]), events_check[2]], axis=1)
                events_check = events_check.copy()

                print(np.mean(events_check[:,0] == events[:,0]), 
                      np.mean(events_check[:,1] == events[:,1]), 
                      np.mean(events_check[:,2] == events[:,2]))

            pos_polarity_mask = events[:,3] > 0
            pos_events = events[pos_polarity_mask]
            neg_events = events[~pos_polarity_mask]
            events = [[pos_events], [neg_events]]
            frame = events2frame(events, cam_h, cam_w)
            video_writer.writeFrame(frame[:,:,::-1])
        except KeyboardInterrupt:
            video_writer.close()
    video_writer.close()


if __name__ == '__main__':
    main()