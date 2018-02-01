from typing import Tuple
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from unet.network import UNet3D

# Parameters and Dataloaders
input_size = (1, 48, 48, 48)

batch_size = 5
data_size = 30

class RandomDataset(Dataset):
    """Random data to test network functions."""
    def __init__(self, input_size: Tuple[int], length: int):
        """Initialisation.

        Args:
            input_size: Dimensions of input image to UNet.
            length: Number of data inputs.
        """
        self.len = length
        self.data = torch.randn(length, *input_size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),
                         batch_size=batch_size, shuffle=True)

model = UNet3D(3, 2, 32, input_size=(48,48,48))
if torch.cuda.device_count() > 1:
    print("Let's use ", torch.cuda.device_count(), " GPUs!")
    model = nn.DataParallel(model)

if torch.cuda.is_available():
    model.cuda()

for data in rand_loader:
    if torch.cuda.is_available():
        input_var = Variable(data.cuda())
    else:
        input_var = Variable(data)

    output = model.forward(input_var)
    print("Successfully processed data, output.shape: ", output.shape)
