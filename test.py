import numpy as np
from src import *
from timm.models import create_model


if __name__ == "__main__":
    checkpoint = torch.load("output/train/20250427-184912-ResNet18-224/best.pth.tar.pth", weights_only=False)
    print(checkpoint['epoch'])
    print(checkpoint['state_dict'].keys())
    print(checkpoint['state_dict']['conv1.weight'].flatten()[0:10])
    print(checkpoint['state_dict']['convbn_first.weight'].flatten()[0:10])
# print(model)