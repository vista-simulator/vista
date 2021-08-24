import imp
import importlib.resources as pkg_resources
import numpy as np
from scipy import interpolate
from typing import Tuple, Optional
import tensorflow as tf
import warnings

from vista import resources
from vista.utils import transform, logging
from .Pointcloud import Pointcloud, Point


class LidarSynthesis:
    def __init__(self,
                 yaw_res=0.1,
                 pitch_res=0.1,
                 yaw_fov=(-180, 180),
                 pitch_fov=(-21.0, 15.0),
                 load_model=True):

        self._res = np.array([yaw_res, pitch_res], dtype=np.float32)
        self._fov = np.array([yaw_fov, pitch_fov], dtype=np.float32)
        self._fov_rad = self._fov * np.pi / 180.

        self._dims = (self._fov[:, 1] - self._fov[:, 0]) / self._res
        self._dims = np.ceil(self._dims).astype(np.int)[:, np.newaxis]

        # integer arrays for indexing
        self._grid_idx = np.meshgrid(np.arange(0, self._dims[0]),
                                     np.arange(0, self._dims[1]))

        self._grid_angles = np.meshgrid(
            np.linspace(self._fov[1, 0], self._fov[1, 0], self._dims[1, 0]),
            np.linspace(self._fov[0, 0], self._fov[0, 0], self._dims[0, 0]))
        self._grid_pitch, self._grid_yaw = (self._grid_angles[0].flatten(),
                                            self._grid_angles[1].flatten())

        xyLen = np.cos(self._grid_pitch)
        self.rays = np.array([ \
            xyLen * np.cos(self._grid_yaw),
            xyLen * np.sin(-self._grid_yaw),
            np.sin(-self._grid_pitch)])

        r = 1
        offsets = tf.meshgrid(tf.range(-r, r + 1), tf.range(-r, r + 1))
        offsets = tf.reshape(tf.stack(offsets, axis=-1), (-1, 2))
        offsets = offsets[tf.math.reduce_any(offsets != 0, axis=1)]
        self.offsets = tf.cast(tf.expand_dims(offsets, 0), tf.int32)

        self.avg_mask = np.load(
            str(pkg_resources.files(resources) / "Lidar/avg_mask.npy"))
        self.avg_mask = self.avg_mask[:, :, 0]

        # path = pkg_resources.files(resources) / "Lidar/LidarRenderModel.tf"
        # self.render_model = None
        # if path.is_dir() and load_model:
        #     logging.debug(f"Loading Lidar model from {path}")
        #     with open(path / "config", "r") as f:
        #         config = f.read().replace('\n', '')
        #         config = eval(config)
        #
        #     obj = imp.load_source("model", str(path / "model.py"))
        #     self.render_model = obj.LidarRenderModel(**config)
        #     self.render_model.build((1, *config["input_shape"]))
        #     self.render_model.load_weights(str(path / "weights.h5"))

        path = pkg_resources.files(resources) / "Lidar/LidarFiller2.h5"
        self.render_model = None
        if path.is_file() and load_model:
            logging.debug(f"Loading Lidar model from {path}")
            self.render_model = tf.keras.models.load_model(
                str(path), custom_objects={"exp": tf.math.exp}, compile=False)

    def synthesize(
            self,
            trans: np.ndarray,
            rot: np.ndarray,
            pcd: np.ndarray,
            return_as_pcd: Optional[bool] = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """ Apply rigid transformation to a dense pointcloud and return new
        dense representation or sparse pointcloud. """
        # points = pcd[:, :3].T

        # Rigid transform of points
        R = transform.rot2mat(rot)
        pcd = pcd.transform(R, trans)
        # R = transform.vec2mat(trans, rot)
        # new_points = np.matmul(R[:3, :3], points) + R[:3, 3][:, np.newaxis]

        # Update distances and intensities (FIXME: intensity placeholder zeros)
        # new_dist = np.linalg.norm(new_points, ord=2, axis=0, keepdims=True)
        # new_int = np.zeros_like(new_dist)  # TODO: fixme
        # new_pcd = np.concatenate((new_points, new_dist, new_int), axis=0).T

        # Convert from new pointcloud to dense image
        sparse = self.pcd2sparse(pcd,
                                 channels=(Point.DEPTH, Point.INTENSITY,
                                           Point.MASK))

        # Find occlusions and cull them from the rendering
        occlusions = self.cull_occlusions(sparse[:, :, 0])
        sparse[occlusions[:, 0], occlusions[:, 1]] = np.nan

        # Densify the image before masking
        dense = self.sparse2dense(sparse, method="nn")

        # Sample the image to simulate active LiDAR using neural masking
        return self.dense2pcd(dense) if return_as_pcd else dense

    def pcd2sparse(self, pcd, channels=Point.DEPTH):
        """ Convert from pointcloud to sparse image in polar coordinates.
        Fill image with specified features of the data (-1 = binary) """

        if not isinstance(channels, list) and not isinstance(channels, tuple):
            channels = [channels]

        # Compute the values to fill and the indicies where to fill
        values = [pcd.get(channel) for channel in channels]
        values = np.stack(values, axis=1)
        inds = self._compute_sparse_inds(pcd)

        # Re-order to fill points with smallest depth last
        order = np.argsort(pcd.dist)[::-1]
        values = values[order]
        inds = inds[:, order]

        # Creat the image and fill it
        img = np.empty((self._dims[1, 0], self._dims[0, 0], len(channels)),
                       np.float32)
        img.fill(np.nan)
        img[-inds[1], inds[0], :] = values
        return img

    @tf.function
    def cull_occlusions(self, sparse, depth_slack=0.1):

        # Coordinates where we have depth samples
        coords = tf.cast(tf.where(sparse > 0), tf.int32)

        # Grab the depths we have at these sparse locations
        # TODO: combine this gather_nd with the gather_nd below (only one
        # call is necessary)
        depths = tf.gather_nd(sparse, coords)

        # At each location, also compute coordinate for all of its neighbors
        samples = tf.expand_dims(coords, 1) + self.offsets  # (N, M, 2)

        # Collect the samples in each neighborhood
        if tf.test.is_gpu_available():
            # gather_nd on GPU will not throw error on out-of-bounds indicies;
            # however, it returns 0.0 at these locations. So we need to set
            # these to nan manually after.
            neighbor_depth = tf.gather_nd(sparse, samples)
            neighbor_depth = tf.where(neighbor_depth == 0.0, np.nan,
                                      neighbor_depth)

        else:
            # gather_nd on CPU will throw error on out-of-bounds indicies.
            # so before calling we need to clip the indicies to min/max bounds
            samples = tf.stack([
                tf.clip_by_value(samples[:, :, 0], 0, sparse.shape[0] - 1),
                tf.clip_by_value(samples[:, :, 1], 0, sparse.shape[1] - 1)
            ], -1)
            neighbor_depth = tf.gather_nd(sparse, samples)

        # For each location, compute the average depth of all neighbors
        valid = ~tf.math.is_nan(neighbor_depth)
        neighbor_depth = tf.where(valid, neighbor_depth, 0.0)
        avg_depth = (tf.reduce_sum(neighbor_depth, axis=1) /
                     tf.reduce_sum(tf.cast(valid, tf.float32), axis=1))

        # Estimate if the location is occluded by measuring if its depth
        # greater than its surroundings (i.e. if it is behind its surroundings)
        # Some amound of slack can be added here to allow for edge cases.
        occluded = (depths - depth_slack) > avg_depth

        # Create a new sparse image by replacing all occluded coordinates by
        # empty depth values (nan)
        occluded_coords = coords[occluded]
        # num_occluded = tf.shape(occluded_coords)[0]
        # nans = tf.fill((num_occluded, ), np.nan)
        # sparse = tf.tensor_scatter_nd_update(sparse, occluded_coords, nans)
        #
        # return sparse
        return occluded_coords

    def cull_occlusions_np(self, sparse, depth_slack=0.1):
        coords = np.array(np.where(sparse > 0)).T
        depths = sparse[coords[:, 0], coords[:, 1]]

        samples = np.expand_dims(coords, 1) + self.offsets.numpy()
        samples[:, :, 0] = np.clip(samples[:, :, 0], 0, sparse.shape[0] - 1)
        samples[:, :, 1] = np.clip(samples[:, :, 1], 0, sparse.shape[1] - 1)

        neighbor_depth = sparse[samples[:, :, 0], samples[:, :, 1]]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            avg_depth = np.nanmean(neighbor_depth, axis=1)

        my_depth = sparse[coords[:, 0], coords[:, 1]]

        # point is valid if it is closer than the average depth around it
        occluded = (my_depth - depth_slack) > avg_depth
        occluded_coords = coords[occluded]

        # remove (cull) all invalid points
        # sparse[occluded_coords[:, 0], occluded_coords[:, 1]] = np.nan
        # return sparse
        return occluded_coords

    def sparse2dense(self, sparse, method="linear"):
        """ Convert from sparse image representation of pointcloud to dense. """

        if method == "nn":
            sparse[np.isnan(sparse)] = 0.0
            sparse = sparse[np.newaxis, :, :, [0]]
            # dense = self.render_model.s2d(sparse)[0, :, :, 0].numpy()
            dense = self.render_model(sparse)[0, :, :, 0].numpy()
        else:
            # mask all invalid values
            zs = np.ma.masked_invalid(sparse)

            # retrieve the valid, non-Nan, defined values
            valid_xs = self._grid_idx[0][~zs.mask]
            valid_ys = self._grid_idx[1][~zs.mask]
            valid_zs = zs[~zs.mask]

            # generate interpolated array of values
            dense = interpolate.griddata((valid_xs, valid_ys),
                                         valid_zs,
                                         tuple(self._grid_idx),
                                         method=method)
            dense[np.isnan(dense)] = 0.0
        return dense

    def dense2pcd(self, dense_depth, dense_intensity=None):
        """ Sample mask from network and render points from mask """
        # TODO: load trained masking network and feed dense through
        # For now, simply load a mask prior from training data and sample
        mask = self.avg_mask > np.random.uniform(size=(self.avg_mask.shape))

        dist = dense_depth[mask]
        intensity = None
        if dense_intensity is not None:
            intensity = dense_intensity[mask]
        pitch, yaw = np.where(mask)

        yaw = yaw * (self._fov_rad[0, 1] - self._fov_rad[0, 0]) / \
              self._dims[0, 0] + self._fov_rad[0, 0]

        pitch = pitch * (self._fov_rad[1, 1] - self._fov_rad[1, 0]) / \
              self._dims[1, 0] + self._fov_rad[1, 0]

        xyLen = np.cos(pitch)
        rays = np.array([ \
            xyLen * np.cos(yaw),
            xyLen * np.sin(-yaw),
            np.sin(-pitch)])

        points = dist[:, np.newaxis] * rays.T
        pcd = Pointcloud(points, intensity)
        return pcd

    def _compute_sparse_inds(self, pcd):
        """ Compute the indicies on the image representation which will be
        filled for a given pointcloud """

        # project point cloud to 2D point map
        yaw = np.arctan2(-pcd.y, pcd.x)
        pitch = np.arcsin(pcd.z / pcd.dist)
        angles = np.stack((yaw, pitch))

        fov_range = self._fov_rad[:, [1]] - self._fov_rad[:, [0]]
        slope = self._dims / fov_range
        inds = slope * (angles - self._fov_rad[:, [0]])

        inds = np.floor(inds).astype(np.int)
        np.clip(inds, np.zeros((2, 1)), self._dims - 1, out=inds)

        return inds
