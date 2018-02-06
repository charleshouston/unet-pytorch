### Classes to load and process data.

from typing import Tuple, Callable, Dict
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import skimage as ski
import skimage.io, skimage.transform, skimage.util
from sklearn.feature_extraction.image import extract_patches
import numpy as np

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class MicroscopyDataset(Dataset):
    """Data of volumetric image(s) from confocal microscopy."""

    def __init__(self, data_dir: str,
                 net_size_in: Tuple[int], net_size_out: Tuple[int],
                 transform: Callable=None):
        """Initialisation.

        Args:
            data_dir: Path to location of image and mask files.
            net_size_in: Image input size to network.
            net_size_out: Output size from network.
            transform: Function for real time transformation of images/masks.
        """
        self.data_dir = Path(data_dir)
        self.net_size_in = net_size_in
        self.net_size_out = net_size_out
        self.transform = transform

        self.length = self._generate_tile_images(self.net_size_in,
                                                  self.net_size_out)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        filename = '{:d}.tif'.format(idx)
        img = ski.io.imread(str(self.data_dir / 'images' / filename))
        mask = ski.io.imread(str(self.data_dir / 'masks' / filename))

        sample = {'image': img, 'mask': mask}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def _generate_tile_images(self, tile_size: Tuple[int],
                               step_size: Tuple[int]) -> int:
        """Saves tiles image and mask file by overlap tile strategy.

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
                img = self.pad_img(img, self.net_size_in, self.net_size_out)

                img_tiles = extract_patches(img, self.net_size_in,
                                            self.net_size_out)
                img_tiles = img_tiles.reshape(-1, *self.net_size_in)
                mask_tiles = extract_patches(mask, self.net_size_out,
                                                self.net_size_out)
                mask_tiles = mask_tiles.reshape(-1, *self.net_size_out)

                for img_tile, mask_tile in zip(img_tiles, mask_tiles):
                    savename = '{:d}.tif'.format(tile_counter)
                    ski.io.imsave(str(img_dir / savename), img_tile)
                    ski.io.imsave(str(mask_dir / savename), mask_tile)
                    tile_counter += 1
        else:
            n_imgs = len([x for x in img_dir.glob('**/*.tif') if x.is_file()])
            n_masks = len([x for x in mask_dir.glob('**/*.tif') if x.is_file()])
            if n_imgs != n_masks:
                raise ValueError('Non-matching number of images and masks.')
            tile_counter = n_imgs

        return tile_counter

    @staticmethod
    def pad_img(self, img: np.ndarray, net_size_in: Tuple[int],
                net_size_out: Tuple[int]) -> np.ndarray:
        """Pads image according to neural net input and output sizes.

        Args:
            img: Image file to be padded.
            net_size_in: Input shape for tile to neural network.
            net_size_out: Output shape for tile to neural network.

        Returns:
            The input img after padding.

        Raises:
            ValueError: When network input and output dimensionality
                does not match.
        """

        if len(net_size_in) != len(net_size_out):
            raise ValueError('Network input and output shapes must have '
                             'equal dimensions.'
                             'Got input size: {}'
                             'Got output size: {}'
                             .format(len(net_size_in), len(net_size_out)))

        padding = [((size_in - size_out) // 2,
                    (size_in - size_out) // 2)
                   for size_in, size_out in zip(net_size_in, net_size_out)]
        return ski.util.pad(img, padding, mode='reflect')
