import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import cv2
import multiprocessing
import os
import pickle
from tqdm import tqdm
from pathlib import Path

from vista.entities.sensors.lidar_utils import LidarSynthesis

# Parse Arguments
parser = argparse.ArgumentParser(
    description='Preprocess lidar data to be compatible with VISTA')
parser.add_argument('-i',
                    '--input',
                    type=str,
                    required=True,
                    help='Path to the trace to prepare')
parser.add_argument('-j',
                    '--jobs',
                    type=int,
                    default=multiprocessing.cpu_count(),
                    help='Threads')
args = parser.parse_args()

synthesizer = LidarSynthesis()
f_in = h5py.File(os.path.join(args.input, "lidar_3d.h5"), "r")
f_xyz = f_in['xyz']
f_intensity = f_in['intensity']
f_timestamp = f_in['timestamp']
n_total = f_timestamp.shape[0]

f_out = h5py.File(os.path.join(args.input, "lidar_3d_vista_new.h5"), "w")
d_timestamp = f_out.create_dataset(name="timestamp", data=f_timestamp[:])
d_pcd = f_out.create_dataset(name="pcd",
                             shape=(n_total, f_xyz.shape[1], 5),
                             chunks=(1, f_xyz.shape[1], 5),
                             dtype=np.float16)
d_depth = f_out.create_dataset(name="d_depth",
                               shape=(n_total, synthesizer._dims[1],
                                      synthesizer._dims[0], 1),
                               chunks=(1, synthesizer._dims[1],
                                       synthesizer._dims[0], 1),
                               dtype=np.float16)
d_int = f_out.create_dataset(name="d_int",
                             shape=(n_total, synthesizer._dims[1],
                                    synthesizer._dims[0], 1),
                             chunks=(1, synthesizer._dims[1],
                                     synthesizer._dims[0], 1),
                             dtype=np.uint8)
d_mask = f_out.create_dataset(name="mask",
                              shape=(n_total, synthesizer._dims[1],
                                     synthesizer._dims[0], 1),
                              chunks=(1, synthesizer._dims[1],
                                      synthesizer._dims[0], 1),
                              dtype=np.bool)


def preprocess_scan(i):
    scan = f_xyz[i]
    intensity = f_intensity[i]
    timestamp = f_timestamp[i]

    valid_points = (scan.sum(1) != 0)

    dist = np.linalg.norm(scan, ord=2, axis=1)[:, np.newaxis]
    scan = np.concatenate((scan, dist, intensity), axis=1)

    scan_valid = scan[valid_points]

    mask, s_depth, s_int = synthesizer.pcd2sparse(scan_valid, fill=[-1, 3, 4])

    depth = np.expand_dims(synthesizer.sparse2dense(s_depth), -1)
    intensity = np.expand_dims(synthesizer.sparse2dense(s_int), -1)
    mask = ~np.isnan(np.expand_dims(mask, -1))

    return (scan, depth, intensity, mask)


print(f"Preprocessing LiDAR data with {args.jobs} parallel threads")
with tqdm(total=n_total) as pbar:
    # Split all data into chunks to process in parallel before saving
    for chunk in np.array_split(range(n_total), n_total // 200):
        results = []
        # for i in tqdm(chunk):
        #     results.append(preprocess_scan(i))

        # Process a chunk of data and save until storing
        with multiprocessing.Pool(args.jobs) as p:
            for result in p.imap_unordered(preprocess_scan, chunk):
                results.append(result)
                pbar.update()

        # Save results to disk
        scan, depth, intensity, mask = zip(*results)
        d_pcd[chunk] = scan
        d_depth[chunk] = depth
        d_int[chunk] = intensity
        d_mask[chunk] = mask

f_out.close()
