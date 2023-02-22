from torchvision import transforms
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import cv2

def get_transforms():
    """
    Function to normalize and transform arrays to Tensors.
    """
    return transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize(
                                 [0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225]
                                 ) # for imagenet
                             ])

def plot_stats(epochs, train_loss, train_acc, val_loss, val_acc):
    """
    Plots training and validation losses and accuracy
    """
    fig, (ax1, ax2) = plt.subplots(2)
        
    ax1.plot(list(range(1, epochs+1)), train_loss, label="train_loss")
    ax1.plot(list(range(1, epochs+1)), val_loss, label="val_loss")
    ax1.set_title("Loss")
    ax1.set_xlabel("epochs")
    ax1.legend()
    ax1.grid()

    ax2.plot(list(range(1, epochs+1)), train_acc, label="train_acc")
    ax2.plot(list(range(1, epochs+1)), val_acc, label="val_acc")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("epochs")
    ax2.legend()
    ax2.grid()

    plt.tight_layout()
    plt.show()

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
