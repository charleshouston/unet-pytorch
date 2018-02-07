from typing import Tuple
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from unet.network import UNet3D
import unet.dataloader as dl
import torchvision.transforms as tt

affine = dl.AffineTransform3D(range_rotate=360.0, range_zoom=0.0,
                              range_shift=(0.1, 0.1, 0.1))
spline = dl.SplineDeformation(image_shape=(116, 132, 132),
                                spacing_cpts=32,
                                stdev=4)
transform = tt.Compose([affine, spline])

dataset = dl.MicroscopyDataset('data/train', net_size_in=(116, 132, 132),
                               net_size_out=(28, 44, 44), n_classes=2,
                               transform=transform)

dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

model = UNet3D(4, 2, 32, input_size=(116, 132, 132))
if torch.cuda.device_count() > 1:
    print("Let's use ", torch.cuda.device_count(), " GPUs!")
    model = nn.DataParallel(model)

if torch.cuda.is_available():
    model.cuda()

for i_batch, sample_batched in enumerate(dataloader):
    if torch.cuda.is_available():
        input_var = Variable(sample_batched['image']).cuda()
    else:
        input_var = Variable(sample_batched['image'])
    print("Inputting image of type: ", type(sample_batched['image']))
    output = model.forward(input_var)
    print("Successfully processed data, output.shape: ", output.shape)

    if i_batch==3:
        break
