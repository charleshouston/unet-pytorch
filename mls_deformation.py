### Class for deformation using Moving Least Squares in 3D.

from typing import Tuple, Dict
import numpy as np
import numpy.linalg
import scipy as sp
import scipy.interpolate


def multidot(*args: np.ndarray) -> np.ndarray:
    """Extends numpy.dot to more than two arguments."""
    ret = args[0]
    for a in args[1:]:
        ret = np.dot(ret, a)
    return ret


class MLSDeformation(object):
    """Deform 3D image by Moving Least Squares [Zhu and Gortler, 2007]."""

    def __init__(self, img_shape: Tuple[int], ps: np.ndarray, alpha: float,
                 interp_grid_spacing: int=1):
        """Initialisation.

        Args:
            img_shape: Shape of image array to be deformed.
            ps: Row vector of control points.
            alpha: Weighting parameter in MLS.
            interp_grid_spacing: Spacing (in pixels) between uniform grid points
                for bilinear interpolation approximation. Defaults to 1 (no
                interpolation).
        """
        self.img_shape = img_shape
        self.img_coords = self.generate_grid_coords(img_shape, 1)
        self.ps = ps
        self.alpha = alpha
        self.interp_grid_spacing = interp_grid_spacing

    def __call__(self, sample: Dict[str, np.ndarray],
                 qs: np.ndarray=None, seed: int=None) -> np.ndarray:
        """Deforms image by rigid transformation.

        If `self.interp_grid_spacing` is > 1 then the transformation is
        calculated on a regular grid and bilinear interpolation used to
        map each coordinate in the image.

        Args:
            sample: Image and mask array to be transformed.
            qs: Displaced control points corresponding to `self.ps`.
            seed: Random number generator seed.

        Returns:
            Deformed sample.

        Raises:
            ValueError: If `self.ps` and `qs` do not have same shape.
        """
        if seed is not None:
            np.random.seed(seed)

        img, mask = sample['image'], sample['mask']

        if qs is not None:
            if self.ps.shape != qs.shape:
                raise ValueError('ps and qs should be the same shape.'
                                 'Got ps.shape: {}'
                                 'Got qs.shape: {}'
                                 .format(ps.shape, qs.shape))
        else:
            qs = (np.random.normal(0, 4, size=(self.ps.shape[0], 3))
                  + self.ps)

        grid_deformed = self._deform_grid(qs)

        img_vals = img[self.grid[:,0],
                       self.grid[:,1],
                       self.grid[:,2]]
        img_deformed = (sp.interpolate.griddata(grid_deformed,
                                                img_vals,
                                                self.img_coords)
                        .reshape(img.shape)
                        .astype(np.uint8))
        mask_vals = mask[self.grid[:,0],
                         self.grid[:,1],
                         self.grid[:,2]]
        mask_deformed = (sp.interpolate.griddata(grid_deformed,
                                                 mask_vals,
                                                 self.img_coords)
                         .reshape(mask.shape)
                         .astype(np.uint8))
        return {'image': img_deformed, 'mask': mask_deformed}

    def _compute_weights(self, v: np.ndarray) -> np.ndarray:
        """Compute weights for centroid of control points."""
        weights = np.asarray([1./(np.linalg.norm(p-v)**(2*self.alpha))
                              for p in self.ps])
        weights[np.isinf(weights)] = 1. # in case v is on a control point
        return weights

    def _compute_weighted_centroid(self, pts: np.ndarray,
                                    weights: np.ndarray) -> np.ndarray:
        """Compute weighted centroid of a vector of coordinates."""
        numerator = np.dot(weights, pts)
        denominator = np.sum(weights)
        return numerator / denominator

    @staticmethod
    def generate_grid_coords(img_shape: Tuple[int],
                             spacing: np.ndarray) -> np.ndarray:
        """Creates spaced grid coordinates."""
        x, y, z = [range(0, s, spacing)
                   for s in img_shape]
        grid = np.meshgrid(x, y, z)
        return np.vstack(map(np.ravel, grid)).T

    def _deform_coord(self, v: np.ndarray) -> np.ndarray:
        """Deforms single coordinate using displaced control points."""
        weights = self._compute_weights(v)
        p_centroid = self._compute_weighted_centroid(self.ps, weights)
        q_centroid = self._compute_weighted_centroid(self.qs, weights)
        PQ_t = np.sum([w * np.outer(self.ps[i] - p_centroid,
                                    self.qs[i] - q_centroid)
                       for i, w in enumerate(weights)], axis=0)
        U, S, V_t = np.linalg.svd(PQ_t)
        return multidot(V_t.T, U.T, v - p_centroid) + q_centroid

    def _deform_grid(self, qs: np.ndarray) -> np.ndarray:
        """Deform regular grid using displaced control points.

        Args:
            qs: Displaced control points corresponding to `self.ps`.

        Returns:
            Deformed grid coordinates.

        Raises:
            ValueError: If `self.ps` and `qs` do not have same shape.
        """

        if not hasattr(self, 'grid'):
            self.grid = self.generate_grid_coords(self.img_shape,
                                                  self.interp_grid_spacing)

        if self.ps.shape != qs.shape:
            raise ValueError('ps and qs should be the same shape.'
                             'Got ps.shape: {}'
                             'Got qs.shape: {}'
                             .format(ps.shape, qs.shape))
        self.qs = qs
        grid_deformed = np.asarray([self._deform_coord(v)
                                    for v in self.grid])
        return grid_deformed
