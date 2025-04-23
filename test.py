import numpy as np
from src import *
from timm.models import create_model
model = create_model(
        'vit_6_4_32_sine',
        num_classes=200)
x = torch.randn(1, 3, 32, 32)
model(x)
# print(model)