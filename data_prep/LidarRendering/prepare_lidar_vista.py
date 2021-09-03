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

from vista.entities.sensors.lidar_utils import LidarSynthesis, Pointcloud, Point

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

synthesizer = LidarSynthesis(load_model=False)
f_in = h5py.File(os.path.join(args.input, "lidar_3d.h5"), "r")
f_xyz = f_in['xyz']
f_intensity = f_in['intensity']
f_timestamp = f_in['timestamp']
n_all_data = f_timestamp.shape[0]
subsample = 10
n_total = len(range(0, n_all_data, subsample))

f_out = h5py.File(os.path.join(args.input, "lidar_3d_vista.h5"), "w")
d_timestamp = f_out.create_dataset(name="timestamp", data=f_timestamp[:])
# d_pcd = f_out.create_dataset(name="pcd",
#                              shape=(n_total, f_xyz.shape[1], 5),
#                              chunks=(1, f_xyz.shape[1], 5),
#                              dtype=np.float32)
d_depth_orig = f_out.create_dataset(name="d_depth_orig",
                                    shape=(n_total, synthesizer._dims[1],
                                           synthesizer._dims[0], 1),
                                    chunks=(1, synthesizer._dims[1],
                                            synthesizer._dims[0], 1),
                                    dtype=np.float32)
d_depth_trans = f_out.create_dataset(name="d_depth_trans",
                                     shape=(n_total, synthesizer._dims[1],
                                            synthesizer._dims[0], 1),
                                     chunks=(1, synthesizer._dims[1],
                                             synthesizer._dims[0], 1),
                                     dtype=np.float32)
d_int_orig = f_out.create_dataset(name="d_int_orig",
                                  shape=(n_total, synthesizer._dims[1],
                                         synthesizer._dims[0], 1),
                                  chunks=(1, synthesizer._dims[1],
                                          synthesizer._dims[0], 1),
                                  dtype=np.uint8)
d_int_trans = f_out.create_dataset(name="d_int_trans",
                                   shape=(n_total, synthesizer._dims[1],
                                          synthesizer._dims[0], 1),
                                   chunks=(1, synthesizer._dims[1],
                                           synthesizer._dims[0], 1),
                                   dtype=np.uint8)
# d_int = f_out.create_dataset(name="d_int",
#                              shape=(n_total, synthesizer._dims[1],
#                                     synthesizer._dims[0], 1),
#                              chunks=(1, synthesizer._dims[1],
#                                      synthesizer._dims[0], 1),
#                              dtype=np.uint8)
d_mask_orig = f_out.create_dataset(name="mask_orig",
                                   shape=(n_total, synthesizer._dims[1],
                                          synthesizer._dims[0], 1),
                                   chunks=(1, synthesizer._dims[1],
                                           synthesizer._dims[0], 1),
                                   dtype=bool)
d_mask_trans = f_out.create_dataset(name="mask_trans",
                                    shape=(n_total, synthesizer._dims[1],
                                           synthesizer._dims[0], 1),
                                    chunks=(1, synthesizer._dims[1],
                                            synthesizer._dims[0], 1),
                                    dtype=bool)


def preprocess_scan(i, cutoff=2.5):
    scan = f_xyz[i]
    intensity = f_intensity[i]
    timestamp = f_timestamp[i]

    pcd = Pointcloud(scan, intensity)
    pcd = pcd[pcd.dist > cutoff]

    # Transform point cloud
    uniform = np.random.uniform
    t = np.array(
        [uniform(-1.5, 1.5),
         uniform(-1.5, 1.5),
         uniform(-0.25, 0.25)])
    theta = uniform(0, 2 * np.pi)
    c, s = (np.cos(theta), np.sin(theta))
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    pcd_trans = pcd.transform(R, t)
    # scan_trans = scan_valid @ R + t

    # dist_valid = np.linalg.norm(scan_valid, ord=2, axis=1)[:, np.newaxis]
    # scan_valid = np.concatenate((scan_valid, dist_valid, int_valid), axis=1)
    # scan_padded = np.zeros((scan.shape[0], 5))
    # scan_padded[:scan_valid.shape[0]] = scan_valid

    # dist_trans = np.linalg.norm(scan_trans, ord=2, axis=1)[:, np.newaxis]
    # scan_trans = np.concatenate((scan_trans, dist_trans, int_valid), axis=1)
    # scan_trans_padded = np.zeros((scan.shape[0], 5))
    # scan_trans_padded[:scan_trans.shape[0]] = scan_trans

    sparse = synthesizer.pcd2sparse(pcd,
                                    channels=(Point.DEPTH, Point.INTENSITY,
                                              Point.MASK))
    s_depth, s_int, mask = (sparse[:, :, i] for i in range(3))
    mask = ~np.isnan(np.expand_dims(mask, -1))
    depth = synthesizer.sparse2dense(s_depth, method="linear")
    depth = np.expand_dims(depth, -1)
    intensity = synthesizer.sparse2dense(s_int, method="linear")
    intensity = np.expand_dims(intensity, -1).astype(np.uint8)

    sparse_trans = synthesizer.pcd2sparse(pcd_trans,
                                          channels=(Point.DEPTH,
                                                    Point.INTENSITY,
                                                    Point.MASK))
    s_depth_trans, s_int_trans, mask_trans = (sparse_trans[:, :, i]
                                              for i in range(3))
    mask_trans = ~np.isnan(np.expand_dims(mask_trans, -1))
    occlusions = synthesizer.cull_occlusions_np(s_depth_trans)
    s_depth_trans[occlusions[:, 0], occlusions[:, 1]] = np.nan
    s_int_trans[occlusions[:, 0], occlusions[:, 1]] = np.nan

    depth_trans = synthesizer.sparse2dense(s_depth_trans, method="linear")
    depth_trans_ext = synthesizer.sparse2dense(s_depth_trans, method="nearest")
    depth_trans[depth_trans < cutoff] = depth_trans_ext[depth_trans < cutoff]
    depth_trans = np.expand_dims(depth_trans, -1)

    int_trans = synthesizer.sparse2dense(s_int_trans, method="linear")
    int_trans_ext = synthesizer.sparse2dense(s_int_trans, method="nearest")
    int_trans[int_trans < cutoff] = int_trans_ext[int_trans < cutoff]
    int_trans = np.expand_dims(int_trans, -1).astype(np.uint8)

    # cv2.imshow('hi1', mask.astype(np.float32))
    # cv2.imshow('hi2', depth / 70.)
    # cv2.imshow('hi3', intensity * 5)
    # cv2.imshow('hi4', mask_trans.astype(np.float32))
    # cv2.imshow('hi5', depth_trans / 70.)
    # cv2.imshow('hi6', int_trans * 5)
    # cv2.waitKey(1)

    return (mask, depth, intensity, mask_trans, depth_trans, int_trans)


with tqdm(total=n_total) as pbar:
    # Split all data into chunks to process in parallel before saving
    chunks = np.array_split(range(0, n_all_data, subsample), n_total // 200)
    ichunks = np.array_split(range(n_total), n_total // 200)
    print(f"Preprocessing LiDAR data with {args.jobs} parallel threads in " +
          f"{len(chunks)} chunks")
    for ichunk, chunk in zip(ichunks, chunks):
        results = []
        # for i in tqdm(chunk):
        #     results.append(preprocess_scan(i))

        # Process a chunk of data and save until storing
        with multiprocessing.Pool(args.jobs) as p:
            for result in p.imap(preprocess_scan, chunk):
                results.append(result)
                pbar.update()

        # Save results to disk
        mask_o, depth_o, int_o, mask_t, depth_t, int_t = zip(*results)
        d_mask_orig[ichunk] = mask_o
        d_depth_orig[ichunk] = depth_o
        d_int_orig[ichunk] = int_o
        d_mask_trans[ichunk] = mask_t
        d_depth_trans[ichunk] = depth_t
        d_int_trans[ichunk] = int_t

f_out.close()
