### Class for deformation using Moving Least Squares in 3D.

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


def meshgrid2(*arrs: np.ndarray) -> np.ndarray:
    """Extends numpy.meshgrid to arbitrary dimensions."""
    arrs = tuple(reversed(arrs))
    lens = map(len, arrs)
    dim = len(arrs)
    sz = 1
    for s in lens:
        sz *= s
    ans = []
    for i, arr in enumerate(arrs):
        slc = [1]*dim
        slc[i] = lens[i]
        arr2 = np.asarray(arr).reshape(slc)
        for j, sz in enumerate(lens):
            if j != i:
                arr2 = arr2.repeat(sz, axis=j)
        ans.append(arr2)
    return ans


class MLSDeformation(object):
    """Deform 3D image by Moving Least Squares [Zhu and Gortler, 2007]."""

    def __init__(self, img: np.ndarray, ps: np.ndarray, alpha: float,
                 interp_grid_spacing: int = 1):
        """Initialisation.

        Args:
            img: Image to be deformed as numpy array.
            ps: Row vector of control points.
            alpha: Weighting parameter in MLS.
            interp_grid_spacing: Spacing (in pixels) between uniform grid points
                for bilinear interpolation approximation. Defaults to 1 (no
                interpolation).
        """
        self.img = img
        self.img_coords = self.__generate_grid_coords(1)
        self.ps = ps
        self.alpha = alpha
        self.interp_grid_spacing = interp_grid_spacing

    def __compute_weights(self, v: np.ndarray) -> np.ndarray:
        """Compute weights for centroid of control points."""
        weights = np.asarray([1./(np.linalg.norm(p-v)**(2*self.alpha))
                              for p in self.ps])
        weights[np.isinf(weights)] = 1. # in case v is on a control point
        return weights

    def __compute_weighted_centroid(self, pts: np.ndarray,
                                    weights: np.ndarray) -> np.ndarray:
        """Compute weighted centroid of a vector of coordinates."""
        numerator = np.dot(weights, pts)
        denominator = np.sum(weights)
        return numerator / denominator

    def __generate_grid_coords(self, spacing: np.ndarray) -> np.ndarray:
        """Creates spaced grid coordinates."""
        x, y, z = [range(0, s, spacing)
                   for s in self.img.shape]
        grid = np.meshgrid(x, y, z)
        return np.vstack(map(np.ravel, grid)).T

    def deform_coord(self, v: np.ndarray) -> np.ndarray:
        """Deforms single coordinate using displaced control points."""
        weights = self.__compute_weights(v)
        p_centroid = self.__compute_weighted_centroid(self.ps, weights)
        q_centroid = self.__compute_weighted_centroid(self.qs, weights)
        PQ_t = np.sum([w * np.outer(self.ps[i] - p_centroid,
                                    self.qs[i] - q_centroid)
                       for i, w in enumerate(weights)], axis=0)
        U, S, V_t = np.linalg.svd(PQ_t)
        return multidot(V_t.T, U.T, v - p_centroid) + q_centroid

    def __deform_grid(self, qs: np.ndarray) -> np.ndarray:
        """Deform regular grid using displaced control points.

        Args:
            qs: Displaced control points corresponding to `self.ps`.

        Returns:
            Deformed grid coordinates.

        Raises:
            ValueError: If `self.ps` and `qs` do not have same shape.
        """

        if not hasattr(self, 'coords'):
            self.grid = self.__generate_grid_coords(self.interp_grid_spacing)
            self.grid_vals = self.img[self.grid[:,0],
                                      self.grid[:,1],
                                      self.grid[:,2]]

        if self.ps.shape != qs.shape:
            raise ValueError('ps and qs should be the same shape.'
                             'Got ps.shape: {}'
                             'Got qs.shape: {}'
                             .format(ps.shape, qs.shape))
        self.qs = qs
        grid_deformed = np.asarray([self.deform_coord(v)
                                    for v in self.grid])
        return grid_deformed

    def deform_img(self, qs: np.ndarray) -> np.ndarray:
        """Deforms image by rigid transformation.

        If `self.interp_grid_spacing` is > 1 then the transformation is
        calculated on a regular grid and bilinear interpolation used to
        map each coordinate in the image.

        Args:
            qs: Displaced control points corresponding to `self.ps`.

        Returns:
            Deformed image.

        Raises:
            ValueError: If `self.ps` and `qs` do not have same shape.
        """
        if self.ps.shape != qs.shape:
            raise ValueError('ps and qs should be the same shape.'
                             'Got ps.shape: {}'
                             'Got qs.shape: {}'
                             .format(ps.shape, qs.shape))

        grid_deformed = self.__deform_grid(qs)
        img_deformed = sp.interpolate.griddata(grid_deformed,
                                               self.grid_vals,
                                               self.img_coords)
        return img_deformed
