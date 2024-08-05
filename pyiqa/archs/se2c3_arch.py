import torch
import torch.nn as nn
import torch.nn.functional as F
from pyiqa.utils.registry import ARCH_REGISTRY  # 確保引入了 ARCH_REGISTRY

class SE2C3Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(SE2C3Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        # Apply convolution
        x = self.conv(x)
        return x

@ARCH_REGISTRY.register()  # 使用裝飾器來註冊 SE2C3Net
class SE2C3Net(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(SE2C3Net, self).__init__()
        self.layer1 = SE2C3Conv(in_channels, 64, 3)
        self.layer2 = SE2C3Conv(64, 128, 3)
        self.layer3 = SE2C3Conv(128, 256, 3)
        self.layer4 = SE2C3Conv(256, 128, 3)
        self.fc = nn.Linear(128 * 32 * 32, num_classes)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x
