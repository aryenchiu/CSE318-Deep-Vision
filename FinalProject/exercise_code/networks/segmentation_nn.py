"""SegmentationNN"""
import torch
import torch.nn as nn
import torchvision.models as models
from .unet_parts import *

class SegmentationNN(nn.Module):

    def __init__(self, num_classes=100, hparams=None, bilinear=False):
        super().__init__()

        self.features = models.mobilenet_v2(pretrained=False).features

        self.classifier = nn.Sequential(
            nn.Dropout2d(0.5),
            nn.Upsample(scale_factor=2, mode='bilinear'), # [1280, 16, 16]
            nn.Conv2d(1280, 512, 3, 1, 1),                # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear'),  # [512, 64, 64]
            nn.Conv2d(512, 256, 3, 1, 1),                  # [256, 64, 64]
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear'),  # [256, 256, 256]
            nn.Conv2d(256, num_classes, 1, 1)           # [100, 256, 256]
        )

        def init_weights(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight.data, 1)
                nn.init.constant_(m.bias.data, 0)

        self.features.apply(init_weights)
        self.classifier.apply(init_weights)

        # self.inc = DoubleConv(3, 64)
        # self.down1 = Down(64, 128)
        # self.down2 = Down(128, 256)
        # self.down3 = Down(256, 512)
        # factor = 2 if bilinear else 1
        # self.down4 = Down(512, 1024 // factor)

        # self.up1 = Up(1024, 512 // factor, bilinear)
        # self.up2 = Up(512, 256 // factor, bilinear)
        # self.up3 = Up(256, 128 // factor, bilinear)
        # self.up4 = Up(128, 64, bilinear)
        # self.outc = OutConv(64, num_classes)

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """

        # x1 = self.inc(x)
        # x2 = self.down1(x1)
        # x3 = self.down2(x2)
        # x4 = self.down3(x3)
        # x5 = self.down4(x4)
        # x = self.up1(x5, x4)
        # x = self.up2(x, x3)
        # x = self.up3(x, x2)
        # x = self.up4(x, x1)
        # logits = self.outc(x)

        x = self.features(x)
        x = self.classifier(x)

        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

class UNet(nn.Module):
    def __init__(self, num_classes=100, bilinear=False):
        super().__init__()

        self.inc = DoubleConv(3, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, num_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3, dropout=True)
        x5 = self.down4(x4, dropout=True)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        return logits

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
        
class DummySegmentationModel(nn.Module):

    def __init__(self, target_image):
        super().__init__()
        def _to_one_hot(y, num_classes):
            scatter_dim = len(y.size())
            y_tensor = y.view(*y.size(), -1)
            zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)
            return zeros.scatter(scatter_dim, y_tensor, 1)

        target_image[target_image == -1] = 1

        self.prediction = _to_one_hot(target_image, 100).permute(2, 0, 1).unsqueeze(0)

    def forward(self, x):
        return self.prediction.float()
