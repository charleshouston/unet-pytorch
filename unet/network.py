### Class to define 3D U-Net.

from typing import List, Tuple
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class UNet3D(nn.Module):
    """3D U-Net network architecture."""

    def __init__(self, n_layer: int, n_class: int, features_root: int,
                 input_size: Tuple[int], pool_size: int = 2,
                 conv_size: int = 3, deconv_size: int = 2):
        """Initialisation of network.

        Args:
            n_layer: Number of U-Net resolution steps, equivalent to number of
                analysis layers.
            n_class: Number of output classes.
            features_root: Number of features in the first layer of the network.
            input_size: Size of 3D input image to network.
            pool_size: Size and stride of the max pooling window.
            conv_size: Size of the convolution kernel.
            deconv_size: Size and stride of the deconvolution kernel.
        """
        super(UNet3D, self).__init__()
        self.n_layer = n_layer
        self.n_class = n_class
        self.features_root = features_root
        self.input_size = input_size
        self.pool_size = pool_size
        self.conv_size = conv_size
        self.deconv_size = deconv_size

        self.layers = self.__construct_layers()

        # Crop sizes for concatenation at shortcut connections
        self.dimen_diff = [self.calc_dimen_diff(i)
                           for i in range(1, self.n_layer)]

    def __construct_layers(self):
        """Instantiates layers for network.

        Returns:
            A list of layers in the network.
        """
        layers = []
        n_features = self.features_root

        # Analysis path
        for i in range(self.n_layer):
            if i == 0:
                features_in = 1 # TODO: adapt for RGB images
            else:
                features_in = n_features

            # First convolution + batch norm
            layers.append(nn.Conv3d(features_in, n_features,
                                    kernel_size=self.conv_size))
            layers.append(nn.BatchNorm3d(n_features))
            n_features *= 2

            # Second convolution + batch norm
            layers.append(nn.Conv3d(n_features//2, n_features,
                                    kernel_size=self.conv_size))
            layers.append(nn.BatchNorm3d(n_features))

        # Synthesis path
        for i in range(self.n_layer-1, 0, -1):
            # Upconvolution
            layers.append(nn.ConvTranspose3d(n_features, n_features,
                                             kernel_size=self.deconv_size,
                                             stride=self.deconv_size))

            # First convolution + batch norm
            layers.append(nn.Conv3d(n_features + n_features//2,
                                    n_features//2, kernel_size=self.conv_size))
            n_features //= 2
            layers.append(nn.BatchNorm3d(n_features))

            # Second convolution + batch norm
            layers.append(nn.Conv3d(n_features, n_features,
                                    kernel_size=self.conv_size))
            layers.append(nn.BatchNorm3d(n_features))

        # Final convolution layer
        layers.append(nn.Conv3d(n_features, self.n_class,
                                kernel_size=1))

        return layers

    def __calc_layer_dimension(self, n: int) -> List[int]:
        """Calculates the shape of a U-Net layer for shortcut connections.

        If the layer is an analysis (downward) resolution step, calculates
        the output of that layer before max pooling. If the layer is a
        synthesis step, calculates the input before the first convolution.

        Args:
            n: Layer number (first analysis layer is 1).

        Returns:
            The shape of the output Tensor.
        """
        if n > self.n_layer: # this is a synthesis path layer
            shape = self.__calc_layer_dimension(self.n_layer)
            num_operations = n - self.n_layer
            for i in range(num_operations):
                if i != 0:
                    shape -= (2 * (self.conv_size - 1))
                shape *= self.deconv_size
        else: # this is an analysis path layer
            shape = np.array(self.input_size)
            for i in range(n):
                if i != 0:
                    shape //= self.pool_size
                shape -= (2 * (self.conv_size - 1))
        return shape

    def calc_dimen_diff(self, res_step: int) -> List[int]:
        """Calculate dimension difference between up and down layers.

        The difference is the size difference (in pixels) between the
        input to the `n`th layer of the U-Net and the corresponding
        layer in the synthesis path. Used for concatenation in
        shortcut connections.

        Args:
            res_step: Resolution step of network (max is self.n_layer).

        Returns:
            A list of the shape difference in each axis.
        """
        shape_a = self.__calc_layer_dimension(res_step)
        shape_s = self.__calc_layer_dimension(2 * self.n_layer - res_step)
        return (shape_a - shape_s)

    def forward(self, x: Variable) -> Variable:
        """Forward pass through network.

        Args:
            x: Network input.

        Returns:
            The output of the network.
        """
        dw_features = []
        layer_num = 0 # variable to iterate through self.layers

        # Analysis path
        for i in range(self.n_layer):
            if i != 0:
                x = F.max_pool3d(x, self.pool_size, stride=self.pool_size)

            # First convolution + batch norm + RELU
            x = self.layers[layer_num](x)
            layer_num += 1
            x = F.relu(self.layers[layer_num](x))
            layer_num += 1

            # Second convolution + batch norm + RELU
            x = self.layers[layer_num](x)
            layer_num += 1
            x = F.relu(self.layers[layer_num](x))
            layer_num += 1

            if i != self.n_layer:
                dw_features.append(x) # save analysis layers for shortcuts later

        # Synthesis path
        for i in range(self.n_layer-1, 0, -1):
            # Upconvolution
            x = self.layers[layer_num](x)
            layer_num += 1

            # Shortcut connection
            difference = self.dimen_diff[i - 1] # size difference for shortcut
            crop = [(di // 2 + (di % 2 > 0), di // 2)
                           for di in difference]
            shortcut = dw_features[i-1][:,:,
                (crop[0][0]):(dw_features[i-1].size()[2] - crop[0][1]),
                (crop[1][0]):(dw_features[i-1].size()[3] - crop[1][1]),
                (crop[2][0]):(dw_features[i-1].size()[4] - crop[2][1])]
            x = torch.cat((shortcut, x), dim=1)

            # First convolution + batch norm + RELU
            x = self.layers[layer_num](x)
            layer_num += 1
            x = F.relu(self.layers[layer_num](x))
            layer_num += 1

            # Second convolution + batch norm + RELU
            x = self.layers[layer_num](x)
            layer_num += 1
            x = F.relu(self.layers[layer_num](x))
            layer_num += 1

        # Final convolution layer
        x = self.layers[layer_num](x)
        return x
