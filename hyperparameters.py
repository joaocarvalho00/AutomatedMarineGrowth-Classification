import torch
import torch.nn as nn
import cv2

def conv(in_channels, out_channels):
    """
    Convolutional layer function with Batch Normalization and Activation function,
    Conv2d -> BatchNorm -> ActivationFunction
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

def up_conv(in_channels, out_channels):
    """
    Upscaling layer
    Upscale -> ActivationFunction
    """
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
        nn.ReLU(inplace=True)
    )

def UnetLoss(preds, targets):
    """
    Loss function for UNet
    """
    loss_function = nn.CrossEntropyLoss()
    ce_loss = loss_function(preds, targets)
    acc = (torch.max(preds, 1)[1] == targets).float().mean()
    return ce_loss, acc