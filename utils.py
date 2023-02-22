from torchvision import transforms
import matplotlib.pyplot as plt


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

def plot_stats(train_loss, train_acc, val_loss, val_acc):
    
    fig, (ax1, ax2) = plt.subplots(2)
        
    ax1.plot(train_loss, label="train_loss")
    ax1.plot(val_loss, label="val_loss")
    ax1.set_title("Loss")
    ax1.set_xlabel("epochs")
    ax1.legend()

    ax2.plot(train_acc, label="train_acc")
    ax2.plot(val_acc, label="val_acc")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("epochs")
    ax2.legend()


    plt.tight_layout()
    plt.show()
