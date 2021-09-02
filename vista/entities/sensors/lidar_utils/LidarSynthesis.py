import imp
import importlib.resources as pkg_resources
import numpy as np
from scipy import interpolate
from typing import Tuple, Optional, Union
import tensorflow as tf

for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_virtual_device_configuration(
        gpu,
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1300)])

import warnings

from vista import resources
from vista.utils import transform, logging
from .Pointcloud import Pointcloud, Point


class LidarSynthesis:
    def __init__(self,
                 yaw_res: float = 0.1,
                 pitch_res: float = 0.1,
                 yaw_fov: Tuple[float, float] = (-180., 180.),
                 pitch_fov: Tuple[float, float] = (-21.0, 12.6),
                 culling_r: int = 1,
                 load_model: bool = True):

        ### Basic properties required for setting up the synthesizer including
        # the dimensionality and resolution of the image representation space
        self._res = np.array([yaw_res, pitch_res], dtype=np.float32)
        self._fov = np.array([yaw_fov, pitch_fov], dtype=np.float32)
        self._fov_rad = self._fov * np.pi / 180.

        self._dims = (self._fov[:, 1] - self._fov[:, 0]) / self._res
        self._dims = np.ceil(self._dims).astype(np.int)[:, np.newaxis]

        ### Culling occluded LiDAR
        # Create a list of offset coordinates within a radius R of the origin,
        # but excluding the origin itself.
        cull_axis = tf.range(-culling_r, culling_r + 1)
        offsets = tf.meshgrid(cull_axis, cull_axis)
        offsets = tf.reshape(tf.stack(offsets, axis=-1), (-1, 2))
        offsets = offsets[tf.math.reduce_any(offsets != 0, axis=1)]
        self.offsets = tf.cast(tf.expand_dims(offsets, 0), tf.int32)

        ### Rendering masks and neural network model for sparse -> dense
        rsrc_path = pkg_resources.files(resources)
        self.avg_mask = np.load(str(rsrc_path / "Lidar/avg_mask.npy"))

        self.load_model = load_model
        self.render_model = None
        path = rsrc_path / "Lidar/LidarFiller5.h5"
        if path.is_file() and load_model:
            logging.debug(f"Loading Lidar model from {path}")

            self.render_model = tf.keras.models.load_model(
                str(path),
                custom_objects={
                    "exp": tf.math.exp,
                    "tf": tf
                },
                compile=False,
            )

    def synthesize(
        self,
        trans: np.ndarray,
        rot: np.ndarray,
        pcd: np.ndarray,
    ) -> Tuple[Pointcloud, np.ndarray]:
        """ Apply rigid transformation to a dense pointcloud and return new
        dense representation or sparse pointcloud. """

        # Rigid transform of points
        R = transform.rot2mat(rot)
        pcd = pcd.transform(R, trans)

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
        new_pcd = self.dense2pcd(dense)

        return (new_pcd, dense)

    def pcd2sparse(self,
                   pcd: Pointcloud,
                   channels: Point = Point.DEPTH) -> np.ndarray:
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
    def cull_occlusions(self,
                        sparse: Union[np.ndarray, tf.Tensor],
                        depth_slack: float = 0.1) -> tf.Tensor:

        # Coordinates where we have depth samples
        coords = tf.cast(tf.where(sparse > 0), tf.int32)

        # Grab the depths we have at these sparse locations
        # TODO: combine this gather_nd with the gather_nd below (only one
        # call is necessary)
        depths = tf.gather_nd(sparse, coords)

        # At each location, also compute coordinate for all of its neighbors
        samples = tf.expand_dims(coords, 1) + self.offsets  # (N, M, 2)

        # Collect the samples in each neighborhood
        if len(tf.config.list_physical_devices('GPU')) > 0:
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

    def cull_occlusions_np(self,
                           sparse: np.ndarray,
                           depth_slack: float = 0.1) -> np.ndarray:

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

    def sparse2dense(self,
                     sparse: np.ndarray,
                     method: str = "linear") -> np.ndarray:
        """ Convert from sparse image representation of pointcloud to dense. """

        if method == "nn":
            mask = ~np.isnan(sparse)
            sparse[~mask] = 0.0
            sparse = sparse[np.newaxis]
            # dense = self.render_model.s2d(sparse)[0, :, :, 0].numpy()
            dense = self.render_model(sparse)[0].numpy()
        else:
            # mask all invalid values
            zs = np.ma.masked_invalid(sparse)

            # integer arrays for indexing
            grid_x, grid_y = np.meshgrid(np.arange(0, self._dims[0]),
                                         np.arange(0, self._dims[1]))

            # retrieve the valid, non-Nan, defined values
            valid_xs = grid_x[~zs.mask]
            valid_ys = grid_y[~zs.mask]
            valid_zs = zs[~zs.mask]

            # generate interpolated array of values
            dense = interpolate.griddata((valid_xs, valid_ys),
                                         valid_zs,
                                         tuple((grid_x, grid_y)),
                                         method=method)
            dense[np.isnan(dense)] = 0.0
        return dense

    def dense2pcd(self, dense: np.ndarray):
        """ Sample mask from network and render points from mask """
        # TODO: load trained masking network and feed dense through
        # For now, simply load a mask prior from training data and sample
        mask = self.avg_mask > np.random.uniform(size=(self.avg_mask.shape))

        dist = dense[mask, 0]
        intensity = None
        if dense.shape[-1] == 2:  # intensity dimension
            intensity = dense[mask, 1]

        pitch, yaw = np.where(mask)
        pitch, yaw = self.coords2angles(pitch, yaw)
        rays = self.angles2rays(pitch, yaw)

        points = dist[:, np.newaxis] * rays.T
        pcd = Pointcloud(points, intensity)
        return pcd

    def coords2angles(self, pitch_coords: np.ndarray,
                      yaw_coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        yaw = yaw_coords * (self._fov_rad[0, 1] - self._fov_rad[0, 0]) / \
              self._dims[0, 0] + self._fov_rad[0, 0]
        pitch = pitch_coords * (self._fov_rad[1, 0] - self._fov_rad[1, 1]) / \
              self._dims[1, 0] + self._fov_rad[1, 1]
        return pitch, yaw

    def angles2rays(self, pitch: np.ndarray, yaw: np.ndarray) -> np.ndarray:
        xyLen = np.cos(pitch)
        rays = np.array([ \
            xyLen * np.cos(yaw),
            xyLen * np.sin(yaw),
            np.sin(pitch)])
        return rays

    def _compute_sparse_inds(self, pcd: Pointcloud) -> np.ndarray:
        """ Compute the indicies on the image representation which will be
        filled for a given pointcloud """

        # project point cloud to 2D point map
        yaw = np.arctan2(pcd.y, pcd.x)
        pitch = np.arcsin(pcd.z / pcd.dist)
        angles = np.stack((yaw, pitch))

        fov_range = self._fov_rad[:, [1]] - self._fov_rad[:, [0]]
        slope = self._dims / fov_range
        inds = slope * (angles - self._fov_rad[:, [0]])

        inds = np.floor(inds).astype(np.int)
        np.clip(inds, np.zeros((2, 1)), self._dims - 1, out=inds)

        return inds
