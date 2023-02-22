from torchvision import transforms


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