import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as tt

import time
import copy

from unet.network import UNet3D
import unet.dataloader as dl


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


def train_model(model: nn.Module, criterion: nn.Module,
                optimizer: nn.Module, scheduler: nn.Module,
                num_epochs: int=25) -> nn.Module:
    """Train 3D U-Net."""

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        scheduler.step()

        running_loss = 0.0
        running_corrects = 0.0

        # Iterate over data.
        for sample in dataloader:
            # Get the inputs.
            image, mask = sample['image'], sample['mask']

            if torch.cuda.is_available():
                image = Variable(image.cuda())
                mask = Variable(mask.cuda())
            else:
                image, mask = Variable(image), Variable(mask)

            # Zero the parameter gradients.
            optimizer.zero_grad()

            # Forward step.
            output = model(image)
            output = torch.t(output.permute(1, 0, 2, 3, 4)
                             .view([2, -1]))
            target = torch.t(mask.permute(1, 0, 2, 3, 4)
                             .view([2, -1]))
                             .long()[:, 1]
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # Print statistics.
            running_loss += loss.data[0]
            _, binary_output = torch.max(output, dim=1)
            running_corrects += torch.sum(output == mask) / output.size()[0]

        epoch_loss = running_loss / len(dataset)
        epoch_acc = running_corrects / len(dataset)

        print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best Acc: {:4f}'.format(best_acc))

    # Load best model weights.
    model.load_state_dict(best_model_wts)
    return model


model = UNet3D(4, 2, 32, input_size=(116, 132, 132))

if torch.cuda.is_available():
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.cuda()

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.99)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

model_ft = train_model(model, criterion, optimizer, exp_lr_scheduler)

with open('trained.pickle', 'w') as f:
    torch.save(model_ft.state_dict(), f)
