import imp
import importlib.resources as pkg_resources
import numpy as np
from scipy import interpolate
from typing import Tuple, Optional
import tensorflow as tf

from vista import resources
from vista.utils import transform, logging


class LidarSynthesis:
    def __init__(self,
                 yaw_res=0.1,
                 pitch_res=0.1,
                 yaw_fov=(-180, 180),
                 pitch_fov=(-21.0, 15.0)):

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

        path = pkg_resources.files(resources) / "Lidar/LidarRenderModel.tf"
        self.render_model = None
        if path.is_dir():
            logging.debug(f"Loading Lidar model from {path}")
            with open(path / "config", "r") as f:
                config = f.read().replace('\n', '')
                config = eval(config)

            obj = imp.load_source("model", str(path / "model.py"))
            self.render_model = obj.LidarRenderModel(**config)
            self.render_model.build((1, *config["input_shape"]))
            self.render_model.load_weights(str(path / "weights.h5"))
            # get the config, load the class manually and fill the weights

    def synthesize(
            self,
            trans: np.ndarray,
            rot: np.ndarray,
            pcd: np.ndarray,
            return_as_pcd: Optional[bool] = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """ Apply rigid transformation to a dense pointcloud and return new
        dense representation or sparse pointcloud. """
        import time
        tic = time.time()
        # Sample dense pcd to return sparse pcd
        # TODO: For now, use the sparse pointcloud -- eventually replace with
        # the two lines below which compute points from the dense depths
        points = pcd[:, :3].T

        # dist = d_depth.reshape(-1, order="F")
        # points = self.rays * dist

        # Rigid transform of points
        R = transform.vec2mat(trans, rot)
        new_points = np.matmul(R[:3, :3], points) + R[:3, 3][:, np.newaxis]

        # Update distances and intensities (FIXME: intensity placeholder zeros)
        new_dist = np.linalg.norm(new_points, ord=2, axis=0, keepdims=True)
        new_int = np.zeros_like(new_dist)  # TODO: fixme
        new_pcd = np.concatenate((new_points, new_dist, new_int), axis=0).T

        # Convert from new pointcloud to dense image
        transformed_sparse = self.pcd2sparse(new_pcd, fill=3)
        tic2 = time.time()
        transformed_dense = self.sparse2dense(transformed_sparse, method="nn")
        logging.warning(f"{time.time() - tic2} {time.time() - tic}")

        # Sample the image to simulate active LiDAR using neural masking
        if return_as_pcd:
            transformed_pcd = self.dense2pcd(transformed_dense)
            return transformed_pcd
        else:
            return transformed_dense

    def pcd2sparse(self, pcd, fill=3):
        """ Convert from pointcloud to sparse image in polar coordinates.
        Fill image with specified axis of the data (-1 = binary)

        Args:
            pcd (np.array): array of shape (num_points, 5), with columns
                            (x, y, z, dist, intensity)
            fill (int or List[int]): which channel to fill the sparse image
                            with. `channel=-1` fills a binary mask

        """
        # Convert from pointcloud (x, y, z) to sparse polar image (yaw, pitch)
        x, y, z, dist, intensity = np.split(pcd, 5, axis=1)

        inds = self._compute_sparse_inds(x, y, z, dist)

        def create_image(channel):
            img = np.empty((self._dims[1, 0], self._dims[0, 0]))
            img.fill(np.nan)
            values = (pcd[:, channel].ravel()) if (channel != -1) else 1
            img[-inds[1], inds[0]] = values
            return img

        if type(fill) is list:
            outputs = [create_image(channel) for channel in fill]
        else:
            outputs = create_image(fill)

        return outputs

    def sparse2dense(self, sparse, method="linear"):
        """ Convert from sparse image representation of pointcloud to dense. """

        if method == "nn":
            sparse[np.isnan(sparse)] = 0.0
            sparse = sparse[np.newaxis, :, :, np.newaxis]
            dense = self.render_model.s2d(sparse)[0, :, :, 0].numpy()

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

    def dense2pcd(self, dense):
        """ Sample mask from network and render points from mask """
        # TODO: load trained masking network and feed dense through
        mask = self.render_model.d2mask(dense[np.newaxis, :, :, np.newaxis])
        mask = mask.numpy()[0, :, :, 0]
        thresh = np.percentile(mask, 80)
        mask = mask > thresh

        dist = dense[mask]
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
        return points

    def _compute_sparse_inds(self, x, y, z, dist):
        """ Compute the indicies on the image representation which will be
        filled for a given pointcloud geometric data `(x, y, z, dist)` """

        # project point cloud to 2D point map
        yaw = np.arctan2(-y, x)
        pitch = np.arcsin(z / dist)
        angles = np.concatenate((yaw, pitch), axis=1).T

        fov_range = self._fov_rad[:, [1]] - self._fov_rad[:, [0]]
        slope = self._dims / fov_range
        inds = slope * (angles - self._fov_rad[:, [0]])

        inds = np.floor(inds).astype(np.int)
        np.clip(inds, np.zeros((2, 1)), self._dims - 1, out=inds)

        return inds

    # def _euler2rot(self, euler):
    #     euler = np.reshape(euler, -1)
    #     s, c = (np.sin(euler), np.cos(euler))
    #     sx, sy, sz = s
    #     cx, cy, cz = c
    #
    #     Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    #     Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    #     Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    #
    #     return Rx @ Ry @ Rz
