import torch.nn as nn
import timm

class Swin(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.swin = timm.create_model('swinv2_base_window16_256.ms_in1k',
                                      pretrained=True,
                                      num_classes=num_classes)

    def forward(self, x):
        return self.swin(x)