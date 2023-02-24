from torchvision import transforms
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import cv2
import pandas as pd
from tqdm import tqdm
import numpy as np

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

def load_model():
    pass

def save_model():
    pass


def create_rgb_map_from_csv(color_map_path):
    """ 
    Inputs path to csv file with RGB encodings for each class and returns a dictionary with the encodings
    """
    
    colormap = pd.read_csv(color_map_path, sep = ", ", engine = "python")


    r = colormap["r"].to_list()
    g = colormap["b"].to_list()
    b = colormap["b"].to_list()

    colormap_dict = {}

    for i in range(0, len(r)):
        colormap_dict.update({i : [r[i], g[i] , b[i]]})
    
    return colormap_dict


def paint_mask_from_rgb_dict(rgb_dict, img):
    """
    Takes a grayscale image and maps each value (label) to it's corresponding RGB value in the rgb dictionary.

    Inputs:
        rgb_dict: python dictionary containing each label and the corresponding RGB mapping:
        img: grayscale image
    Outputs:
        rgb_img: RGB representation of img with the mappings provided in the dictionary
    """

    rgb_img = np.zeros((img.shape[0], img.shape[1], 3)).astype(int)

    for key in tqdm(rgb_dict.keys()):
        rgb_img[img == key] = rgb_dict[key]

    return rgb_img

def display_original_mask_predicted_mask(color_dict_path, model, dataset_img):
    """
    Displays a desired image, it's ground truth mask and the model's predicted mask.

    Inputs:
        color_dict_path: path to the csv file that maps each label to an RGB value.
        model: network we are using
        dataset_img: desired image
    Outputs:
        Displays the images in a separate window
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    convertTensor = transforms.ToTensor()

    input_img_tensor = convertTensor(dataset_img[0]).to(device)

    with torch.no_grad():
        output = model(input_img_tensor.unsqueeze(0)).cpu()
        mask = output.argmax(dim=1)
        mask_out = mask[0].long().squeeze().numpy()

    original_img = dataset_img[0]
    original_mask = dataset_img[1]

    color_dict = create_rgb_map_from_csv(color_dict_path)

    rgb_mask = paint_mask_from_rgb_dict(color_dict, original_mask)
    rgb_mask_out = paint_mask_from_rgb_dict(color_dict, mask_out)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    ax1.imshow(original_img)
    ax1.set_title("Original image")
    ax1.set_yticklabels([])
    ax1.set_yticklabels([])

    ax2.imshow(rgb_mask)
    ax2.set_title("Ground truth mask")
    ax2.set_yticklabels([])
    ax2.set_yticklabels([])

    ax3.imshow(rgb_mask_out)
    ax3.set_title("Predicted mask")
    ax3.set_yticklabels([])
    ax3.set_yticklabels([])

    plt.tight_layout()
    plt.show()