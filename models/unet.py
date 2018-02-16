### Class to define 3D U-Net.

from typing import List, Tuple
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from models.custom_layers import Softmax3d


class AnalysisLayer(nn.Module):
    """Module for analysis layer of U-Net architecture."""

    def __init__(self, n_features: int,
                 conv_size: int = 3,
                 first: bool = False,
                 pooling: nn.MaxPool3d = None,
                 upconv: nn.ConvTranspose3d = None):
        """Initialisation of layer.

        Args:
            n_features: Number of input features (output will be double).
            conv_size: Size of convolution kernel.
            first: Whether this is the first layer in the U-Net.
            pooling: Pooling layer (if supplied).
            upconv : Upconvolution layer (for bottom layer of U-Net).
        """
        super(AnalysisLayer, self).__init__()
        if first:
            features_in = 1 # TODO adapt for RGB images
        else:
            features_in = n_features
        self.pooling = pooling
        self.conv1 = nn.Conv3d(features_in, n_features,
                               kernel_size=conv_size)
        self.bn1 = nn.BatchNorm3d(n_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(n_features, n_features*2,
                               kernel_size=conv_size)
        self.bn2 = nn.BatchNorm3d(n_features*2)
        self.upconv = upconv

    def forward(self, x: Variable) -> Variable:
        """Forward pass through layer."""
        if self.pooling is not None:
            x = self.pooling(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        if self.upconv is not None:
            x = self.upconv(x)

        return x


class SynthesisLayer(nn.Module):
    """Module for synthesis layer of U-Net architecture."""
    def __init__(self, n_features: int, conv_size: int = 3,
                 upconv_size: int = 2, last: bool = False):
        """Initialisation.

        Args:
            n_features: Number of input features (remember shortcut layers!).
            conv_size: Size of convolution layer kernel.
            upconv_size: Size and stride of upconvolution layer kernel.
            last: Whether this is the final layer in the network.
        """
        super(SynthesisLayer, self).__init__()
        features_out = n_features // 3
        self.conv1 = nn.Conv3d(n_features, features_out,
                               kernel_size=conv_size)
        self.bn1 = nn.BatchNorm3d(features_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(features_out, features_out,
                               kernel_size=conv_size)
        self.bn2 = nn.BatchNorm3d(features_out)
        if last:
            self.upconv = None
        else:
            self.upconv = nn.ConvTranspose3d(features_out, features_out,
                                             kernel_size=upconv_size,
                                             stride=upconv_size)

    def forward(self, x: Variable) -> Variable:
        """Forward pass through layer."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        if self.upconv is not None:
            x = self.upconv(x)

        return x


class FinalLayer(nn.Module):
    """Final layer to reduce to classed pixels."""
    def __init__(self, n_features: int, n_classes: int):
        """Initilisation.

        Args:
            n_features: Number of input features.
            n_classes: Final number of classes.
        """
        super(FinalLayer, self).__init__()
        self.conv_fc = nn.Conv3d(n_features, n_classes, kernel_size=1)
        self.softmax = Softmax3d()

    def forward(self, x: Variable) -> Variable:
        """Forward pass through layer."""
        x = self.conv_fc(x)
        x = self.softmax(x)

        return x


class UNet3D(nn.Module):
    """3D U-Net network architecture."""

    def __init__(self, n_layer: int, n_class: int, features_root: int,
                 input_size: Tuple[int], pool_size: int = 2,
                 conv_size: int = 3, upconv_size: int = 2):
        """Initialisation of network.

        Args:
            n_layer: Number of U-Net resolution steps, equivalent to number of
                analysis layers.
            n_class: Number of output classes.
            features_root: Number of features in the first layer of the network.
            input_size: Size of 3D input image to network.
            pool_size: Size and stride of the max pooling window.
            conv_size: Size of the convolution kernel.
            upconv_size: Size and stride of the upconvolution kernel.
        """
        super(UNet3D, self).__init__()
        self.n_layer = n_layer
        self.n_class = n_class
        self.features_root = features_root
        self.input_size = input_size
        self.pool_size = pool_size
        self.conv_size = conv_size
        self.upconv_size = upconv_size

        self.pool = nn.MaxPool3d(kernel_size=self.pool_size,
                                 stride=self.pool_size)
        self.layers = self.__construct_layers()

        # Crop sizes for concatenation at shortcut connections.
        self.dimen_diff = [self.calc_dimen_diff(i)
                           for i in range(self.n_layer-1)]

    def __construct_layers(self) -> nn.ModuleList:
        """Instantiates layers for network.

        Returns:
            A module list of layers in the network.
        """
        n_features = self.features_root
        layers = nn.ModuleList([])

        # Analysis path
        for i in range(self.n_layer):
            if i == 0:
                layers.append(AnalysisLayer(n_features, first=True))
            elif i == self.n_layer-1:
                # lowest layer
                upconv = nn.ConvTranspose3d(n_features*2, n_features*2,
                                            kernel_size=self.upconv_size,
                                            stride=self.upconv_size)
                layers.append(AnalysisLayer(n_features, pooling=self.pool,
                                            upconv=upconv))
            else:
                layers.append(AnalysisLayer(n_features, pooling=self.pool))
            n_features *= 2

        # Synthesis path
        for i in range(self.n_layer-1, 0, -1):
            n_features += n_features // 2 # shortcut connection
            if i == 1:
                layers.append(SynthesisLayer(n_features, last=True))
            else:
                layers.append(SynthesisLayer(n_features))
            n_features //= 3

        # Final layer
        layers.append(FinalLayer(n_features, self.n_class))

        return layers

    def calc_layer_dimension(self, n: int) -> np.ndarray:
        """Calculates the shape of a U-Net layer for shortcut connections.

        If the layer is an analysis (downward) resolution step, calculates
        the output of that layer before max pooling. If the layer is a
        synthesis step, calculates the input before the first convolution.

        Args:
            n: Layer number (first analysis layer is 0).

        Returns:
            The shape of the output Tensor.
        """
        if n > self.n_layer-1: # this is a synthesis path layer
            shape = self.calc_layer_dimension(self.n_layer-1)
            num_operations = n - self.n_layer + 1
            for i in range(num_operations):
                if i != 0:
                    shape -= (2 * (self.conv_size - 1))
                shape *= self.upconv_size
        else: # this is an analysis path layer
            shape = np.array(self.input_size)
            for i in range(n+1):
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
            res_step: Resolution step of network (max is self.n_layer-1).

        Returns:
            A list of the shape difference in each axis.
        """
        shape_analysis = self.calc_layer_dimension(res_step)
        shape_synthesis = self.calc_layer_dimension(2 * (self.n_layer-1)
                                                    - res_step)
        return (shape_analysis - shape_synthesis)

    def forward(self, x: Variable) -> Variable:
        """Forward pass through network.

        Args:
            x: Network input.

        Returns:
            The output of the network.
        """
        dw_features = []
        shortcut_count = 0
        for i, layer in enumerate(self.layers):
            if i > self.n_layer-1 and i < len(self.layers)-1:
                # Concatenate shortcut connection.
                i_short = 2 * (self.n_layer-1) - i # shortcut index
                difference = self.dimen_diff[i_short]
                crop = [(di // 2 + (di % 2 > 0), di // 2)
                        for di in difference]
                shortcut = dw_features[i_short][:,:,
                    (crop[0][0]):(dw_features[i_short].size()[2] - crop[0][1]),
                    (crop[1][0]):(dw_features[i_short].size()[3] - crop[1][1]),
                    (crop[2][0]):(dw_features[i_short].size()[4] - crop[2][1])]
                x = torch.cat((shortcut, x), dim=1)
                shortcut_count += 1

            x = layer(x)

            if i < self.n_layer-1:
                # Save for shortcut connection.
                dw_features.append(x.clone())

        return x
