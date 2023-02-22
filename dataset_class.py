import torch
from torch.utils.data import Dataset
import os
import cv2
from utils import get_transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class SegmentationDataset(Dataset):
    def __init__(self, split, transforms):
      	# Specify the folder that contains images i
        self.items = os.listdir(f'../dummy_dataset/images_{split}/original')
        self.split = split
        self.transforms = transforms

    def __len__(self):
        # return the number of total samples contained in the dataset
        return len(self.items)

    def __getitem__(self, ix):
        # load the image from disk
        image = cv2.imread(f'../dummy_dataset/images_{self.split}/original/{self.items[ix]}', 1)
        image = cv2.resize(image, (224,224))

        # read the associated mask from disk in grayscale mode
        mask = cv2.imread(f'../dummy_dataset/images_{self.split}/mask/{self.items[ix][:-4]}.png', 0)
        mask = cv2.resize(mask, (224,224))
        # check to see if we are applying any transformations
        if self.transforms is not None:
            # apply the transformations to both image and its mask
            image = self.transforms(image)
            mask = self.transforms(mask)
        # grab the image path from the current index
        return image, mask

    #def choose(self): 
      #return self[randint(len(self))]

    def collate_fn(self, batch):
        ims, masks = list(zip(*batch))
        tfms = get_transforms()
        ims = torch.cat([tfms(im.copy()/255.)[None] for im in ims]).float().to(device)
        ce_masks = torch.cat([torch.Tensor(mask[None]) for mask in masks]).long().to(device)
        return ims, ce_masks