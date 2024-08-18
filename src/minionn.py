import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniONN(nn.Module):
    def __init__(self, num_classes=10):
        super(MiniONN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1, bias=False)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1, bias=False)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1, bias=False)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1, bias=False)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0, stride=1, bias=False)
        self.conv7 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=1, padding=0, stride=1, bias=False)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(1024, num_classes)
        self.avgpool = nn.AvgPool2d(kernel_size=2)
        self.flatten = nn.Flatten()

    def forward(self, x):
        # MiniONN from 
        # MiniONN from MiniONN paper: 7Conv + 1FC + 2AP + 7ReLU
        
        x = self.relu(self.conv1(x))

        x = self.relu(self.conv2(x))
        x = self.avgpool(x)

        x = self.relu(self.conv3(x))

        x = self.relu(self.conv4(x))
        x = self.avgpool(x)

        x = self.relu(self.conv5(x))

        x = self.relu(self.conv6(x))

        x = self.relu(self.conv7(x))
        
        x = self.flatten(x)
        x = self.linear(x)

        # x = F.softmax(x, dim=-1)

        return x 