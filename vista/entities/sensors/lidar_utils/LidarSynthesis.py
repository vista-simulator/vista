import numpy as np
from typing import Tuple, Optional
from scipy import interpolate


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

    def synthesize(
            self,
            trans: np.ndarray,
            rot: np.ndarray,
            dense_pcd: np.ndarray,
            return_as_pcd: Optional[bool] = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """ Apply rigid transformation to a dense pointcloud and return new
        dense representation or sparse pointcloud. """

        # Sample dense pcd to return sparse pcd
        dist = img_.reshape(-1, order="F")

        points = self.rays * dist
        new_points = np.matmul(R, points) + t[:, np.newaxis]

        new_dist = np.linalg.norm(new_points, ord=2, axis=0)
        new_pcd = np.stack((new_points, new_dist), axis=0).T

        transformed = self.sparse2dense(self.pcd2sparse(new_pcd, fill=3))
        if return_as_pcd:
            transformed = self.dense2pcd(transformed)
        return transformed

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
        pass

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
