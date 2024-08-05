import torch
import torch.nn as nn
import torch.nn.functional as F
from pyiqa.utils.registry import ARCH_REGISTRY  # Ensure ARCH_REGISTRY is imported

class SE2C3Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, use_batchnorm=True):
        super(SE2C3Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.use_batchnorm = use_batchnorm
        if use_batchnorm:
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_batchnorm:
            x = self.bn(x)
        return x

@ARCH_REGISTRY.register()  # Register the architecture with the decorator
class SE2C3Net(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(SE2C3Net, self).__init__()

        # Define the architecture with more layers and complexity
        self.features = nn.Sequential(
            SE2C3Conv(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            SE2C3Conv(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            SE2C3Conv(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            SE2C3Conv(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            SE2C3Conv(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            SE2C3Conv(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            SE2C3Conv(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            SE2C3Conv(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 1 * 1, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.classifier(x)
        return x
