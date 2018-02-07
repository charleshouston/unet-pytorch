### Classes to load and process data.

from typing import Tuple, Callable, Dict, Union
from pathlib import Path

import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision

import skimage as ski
import skimage.io
import skimage.util

import scipy as sp
import scipy.ndimage.interpolation

from sklearn.feature_extraction.image import extract_patches

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class AffineTransform3D(object):
    """Affine transform of volumetric image data."""

    def __init__(self, range_rotate: float,
                 range_zoom: Union[Tuple[float], float],
                 range_shift: Tuple[float]):
        """Initialisation.

        Transformations assume that order of the input image is in the
        format `depth, width, height`.

        Args:
            range_rotate: Random rotate range in degrees.
            range_zoom: Zoom range.
            range_shift: Fraction range for random shifts in axes.
        """
        self.range_rotate = range_rotate
        if isinstance(range_zoom, float):
            self.range_zoom = [1-range_zoom, 1+range_zoom]
        else:
            self.range_zoom = range_zoom
        self.range_shift = range_shift

    def __call__(self, sample: Dict[str, np.ndarray],
                 seed: int=None) -> Dict[str, np.ndarray]:
        """Carry out affine transformation."""
        image, mask = sample['image'], sample['mask']

        if seed is not None:
            np.random.seed(seed)

        transform_matrix = self._get_affine_transform_matrix(self.range_rotate,
                                                             self.range_zoom,
                                                             self.range_shift,
                                                             image.shape)
        final_affine_matrix = transform_matrix[:3, :3]
        final_offset = transform_matrix[:3, 3]

        image_transformed = sp.ndimage.interpolation.affine_transform(
                image,
                final_affine_matrix,
                final_offset,
                order=0,
                mode='reflect')
        mask_transformed = sp.ndimage.interpolation.affine_transform(
                mask,
                final_affine_matrix,
                final_offset,
                order=0,
                mode='reflect')

        return {'image': image_transformed, 'mask': mask_transformed}

    def _get_affine_transform_matrix(self, r: float,
                                     z: float, t: Tuple[float],
                                     shape: Tuple[int]) -> np.ndarray:
        """Create random affine transform matrix.

        Args:
            r: Rotation range in degrees.
            z: Zoom range.
            t: Shift range in pixels.

        Returns:
            Affine transform matrix.
        """
        # Rotation.
        if self.range_rotate:
            theta = np.pi / 180 * np.random.uniform(-self.range_rotate,
                                                    self.range_rotate)
        else:
            theta = 0

        # Translation.
        t = [(np.random.uniform(-shift, shift) * shape[i])
             for i, shift in enumerate(self.range_shift)]

        # Zoom.
        if np.allclose(self.range_zoom, 1.):
            z = np.asarray([1., 1., 1.])
        else:
            z = np.random.uniform(self.range_zoom[0], self.range_zoom[1], 3)

        transform_matrix = None
        if theta != 0:
            rotation_matrix = np.array([[1, 0, 0, 0],
                                        [0, np.cos(theta), -np.sin(theta), 0],
                                        [0, np.sin(theta), np.cos(theta), 0],
                                        [0, 0, 0, 1]])
            transform_matrix = rotation_matrix

        if np.any(t):
            shift_matrix = np.array([[1, 0, 0, t[0]],
                                     [0, 1, 0, t[1]],
                                     [0, 0, 1, t[2]],
                                     [0, 0, 0, 1]])
            transform_matrix = (shift_matrix if transform_matrix is None
                                             else np.dot(transform_matrix,
                                                         shift_matrix))

        if np.any(z):
            zoom_matrix = np.array([[z[0], 0, 0, 0],
                                    [0, z[1], 0, 0],
                                    [0, 0, z[2], 0],
                                    [0, 0, 0, 1]])
            transform_matrix = (zoom_matrix if transform_matrix is None
                                            else np.dot(transform_matrix,
                                                        zoom_matrix))
        return transform_matrix


class SplineDeformation(object):
    """Random spline deformation of control points in image.

    Interpolates using cubic splines from deformed control points.
    """

    def __init__(self, image_shape: Tuple[int], spacing_cpts: int=32,
                 stdev: int=4):
        """Initialisation.

        Args:
            image_shape: Shape of image to be deformed in pixels.
            spacing_cpts: Spacing between control points for deformation
                in pixels.
            stdev: Standard deviation of random deformation in pixels.
        """
        self.image_shape = image_shape
        self.spacing_cpts = spacing_cpts
        self.cpts = np.mgrid[0:image_shape[0]:spacing_cpts,
                             0:image_shape[1]:spacing_cpts,
                             0:image_shape[2]:spacing_cpts]
        self.stdev = stdev

        max_cpt = np.max(self.cpts.reshape(3, -1), axis=1)
        limits = self.image_shape / max_cpt * self.cpts.shape[1:] - 1
        coords = [np.linspace(0, limits[i], image_shape[i])
                  for i in range(3)]
        self.cpts_to_pixels = np.asarray(np.meshgrid(*coords, indexing='ij'))

        self.pixel_grid = np.mgrid[0:image_shape[0],
                                   0:image_shape[1],
                                   0:image_shape[2]]

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Perform deformation on sample."""

        image, mask = sample['image'], sample['mask']
        cpts_flat = np.reshape(self.cpts,
                               (self.cpts.shape[0], -1)).T
        displacements = np.empty_like(cpts_flat)
        for i in range(len(displacements)):
            displacements[i, :] = np.random.normal(0, self.stdev, 3)
        displacements = np.reshape(displacements.T, self.cpts.shape)

        # Interpolate from cpts to pixels using cubic splines.
        dx = sp.ndimage.interpolation.map_coordinates(
                displacements[0,:], self.cpts_to_pixels,
                order=3, mode='reflect')
        dy = sp.ndimage.interpolation.map_coordinates(
                displacements[1,:], self.cpts_to_pixels,
                order=3, mode='reflect')
        dz = sp.ndimage.interpolation.map_coordinates(
                displacements[2,:], self.cpts_to_pixels,
                order=3, mode='reflect')

        pixels_displaced = np.asarray([dx, dy, dz])

        # Displace pixels.
        indices = [self.pixel_grid[i] + pixels_displaced[i]
                   for i in range(3)]
        image_deformed = np.reshape(sp.ndimage.interpolation.map_coordinates(
                image, indices, order=3, mode='reflect'),
                image.shape)
        mask_deformed = np.reshape(sp.ndimage.interpolation.map_coordinates(
                mask, indices, order=3, mode='reflect'),
                mask.shape)
        # TODO: account for more than 2 classes
        mask_deformed = ((mask_deformed > 255/2) * 255)

        return {'image': image_deformed, 'mask': mask_deformed}


class ToTensor(object):
    """Converts ndarray to torch Tensor."""

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        image, mask = sample['image'], sample['mask']

        # Numpy format is D x H x W
        # torch expects C x D x H x W
        image = image.reshape((1,) + image.shape)
        return {'image': torch.from_numpy(image).float(),
                'mask': torch.from_numpy(mask).float()}


class MicroscopyDataset(Dataset):
    """Data of volumetric image(s) from confocal microscopy."""

    def __init__(self, data_dir: str,
                 net_size_in: Tuple[int], net_size_out: Tuple[int],
                 n_classes: int,
                 transform: Callable=None):
        """Initialisation.

        Args:
            data_dir: Path to location of image and mask files.
            net_size_in: Image input size to network.
            net_size_out: Output size from network.
            n_classes: Number of classes in output network.
            transform: Function for real time transformation of images/masks.

        Raises:
            ValueError: When net input and output dimensionality is not equal.
        """
        self.data_dir = Path.cwd() / data_dir

        if len(net_size_in) != len(net_size_out):
            raise ValueError('Network input and output shapes must have '
                             'equal dimensions.'
                             'Got input size: {}'
                             'Got output size: {}'
                             .format(len(net_size_in), len(net_size_out)))
        self.net_size_in = net_size_in
        self.net_size_out = net_size_out

        self.n_classes = n_classes
        self.transform = transform

        self.padding = [((size_in - size_out) // 2,
                         (size_in - size_out) // 2)
                        for size_in, size_out in zip(net_size_in, net_size_out)]

        self.length = self._generate_tile_images(self.net_size_in,
                                                 self.net_size_out)

        self.tensor_trns = ToTensor() #TODO: fix this quick hack.

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        filename = '{:d}.npy'.format(idx)
        img = np.load(str(self.data_dir / 'images' / filename))
        mask = np.load(str(self.data_dir / 'masks' / filename))

        sample = {'image': img, 'mask': mask}
        if self.transform:
            sample = self.transform(sample)
        sample['mask'] = ski.util.crop(sample['mask'], self.padding)
        sample['mask'] = self.sparsify(sample['mask'], self.n_classes)
        sample = self.tensor_trns(sample)
        return sample

    def _generate_tile_images(self, tile_size: Tuple[int],
                               step_size: Tuple[int]) -> int:
        """Tiles image and masks by overlap tiling and returns number created.

        Expects `data_dir` to contain at least one image and one
        corresponding mask with name suffix `_mask`. Saves tiles into
        directories called `images` and `masks` in the data folder.
        Will not recreate tiles if these folders already exist.
        Note that images are padded on sides by mirroring.

        Args:
            tile_size: Desired size for tiles, (depth, width, height).
            step_size: Desired step in each dimension.

        Returns:
            The number of tile images created.

        Raises:
            ValueError: When image and mask directories already exist but
                contain unequal number of files.
        """
        tile_counter = 0
        files = [x for x in self.data_dir.glob('**/*.tif') if x.is_file()]
        img_dir = self.data_dir / 'images'
        mask_dir = self.data_dir / 'masks'

        if not img_dir.exists() and not mask_dir.exists():
            img_dir.mkdir()
            mask_dir.mkdir()

            for f in files:
                if not f.with_name(f.stem + '_mask.tif').exists():
                    continue
                img = ski.io.imread(str(f))
                mask = ski.io.imread(str(f.with_name(f.stem + '_mask.tif')))
                img = ski.util.pad(img, self.padding, mode='reflect')
                mask = ski.util.pad(mask, self.padding, mode='reflect')

                img_tiles = extract_patches(img, self.net_size_in,
                                            self.net_size_out)
                img_tiles = img_tiles.reshape(-1, *self.net_size_in)
                mask_tiles = extract_patches(mask, self.net_size_in,
                                             self.net_size_out)
                mask_tiles = mask_tiles.reshape(-1, *self.net_size_in)

                for img_tile, mask_tile in zip(img_tiles, mask_tiles):
                    savename = '{:d}.npy'.format(tile_counter)
                    np.save(str(img_dir / savename), img_tile)
                    np.save(str(mask_dir / savename), mask_tile)
                    tile_counter += 1
        else:
            n_imgs = len([x for x in img_dir.glob('**/*.npy') if x.is_file()])
            n_masks = len([x for x in mask_dir.glob('**/*.npy') if x.is_file()])
            if n_imgs != n_masks:
                raise ValueError('Non-matching number of images and masks.')
            tile_counter = n_imgs

        return tile_counter

    def sparsify(self, arr: np.ndarray, n_classes: int) -> np.ndarray:
        """Convert multi-class mask to separate binary masks."""
        return 1*np.asarray([np.isclose(arr, j/(n_classes-1)*255)
                             for j in range(n_classes)])
