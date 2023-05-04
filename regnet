import torch
import torch.nn as nn
import torch.nn.functional as F


class RegNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, bottleneck_ratio):
        super(RegNetBlock, self).__init__()

        # Calculate number of channels for bottleneck layer
        bottleneck_channels = int(round(out_channels * bottleneck_ratio))

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=stride, padding=1, groups=bottleneck_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        # Residual branch
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)

        x = self.conv3(x)
        x = self.bn3(x)

        # Shortcut branch
        shortcut = self.shortcut(residual)

        # Combine branches
        x += shortcut
        x = F.relu(x, inplace=True)

        return x


class RegNet(nn.Module):
    def __init__(self, num_blocks, num_classes, width_factor, bottleneck_ratio):
        super(RegNet, self).__init__()

        # Calculate initial number of channels
        num_channels = 32

        # Stem convolutional layer
        self.stem = nn.Sequential(
            nn.Conv2d(3, num_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True)
        )

        # Downsampling stages
        self.stage1 = self._make_stage(num_channels, width_factor, bottleneck_ratio, stride=1, num_blocks=num_blocks[0])
        num_channels *= 2
        self.stage2 = self._make_stage(num_channels, width_factor, bottleneck_ratio, stride=2, num_blocks=num_blocks[1])
        num_channels *= 2
        self.stage3 = self._make_stage(num_channels, width_factor, bottleneck_ratio, stride=2, num_blocks=num_blocks[2])
        num_channels *= 2
        self.stage4 = self._make_stage(num_channels, width_factor, bottleneck_ratio, stride=2, num_blocks=num_blocks[3])

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_channels, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
       
