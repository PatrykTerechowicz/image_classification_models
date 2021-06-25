import torch.nn as nn
import torch.utils.data as data
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from typing import Tuple, List


def plot_prediction_bar(ax: plt.Axes, predictions: torch.Tensor, class_names: List, k: int=10):
    """Draws top k best predictions on given plt.Axes.

    Args:
        ax (plt.Axes): Axes on which to draw.
        predictions (torch.Tensor): Predictions from neural network, has to be projected to probabilities(ie. softmax) for proper plotting.
        class_names (List): Names of classes.
        k (int, optional): How much bars to plot. Defaults to 10.
    """
    argsorted = torch.argsort(predictions, descending=True)
    entropy = torch.sum(-predictions*torch.log2(predictions))
    top_predictions = predictions[argsorted[:k]]
    top_predictions_names = [class_names[arg] for arg in argsorted[:k]]
    ax.barh(top_predictions_names, top_predictions, height=0.8)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Probability")
    ax.set_title(f"Entropy: {entropy:.5f}")


def plot_im(ax: plt.Axes, image: torch.Tensor, true_class: str):
    """Plots given image on given axes. Sets title of image to true_class.

    Args:
        ax (plt.Axes): Axes on which to draw.
        image (torch.Tensor): Tensor image which has to be drawn on axes.
        true_class (str): Name of image class.
    """
    image = un_normalize(image)
    image_cpu: np.ndarray = image.cpu().detach().numpy()
    im_max = np.max(image_cpu)
    im_min = np.min(image_cpu)
    image_cpu = (image_cpu-im_min)/(im_max-im_min)
    image_cpu = np.transpose(image_cpu, [1, 2, 0])
    ax.set_title(true_class)
    ax.annotate(true_class, xy=(112, 112))
    ax.grid(False)
    ax.set_axis_off()
    ax.imshow(image_cpu)


def plot_test_sample(fig: plt.Figure, image: torch.Tensor, predictions: torch.Tensor, class_names: List[str], true_class: str, k: int=10):
    """Plot test sample image with predictions on given Figure.

    Args:
        fig (plt.Figure): Figure to draw on.
        image (torch.Tensor): Image to draw.
        predictions (torch.Tensor): Preditions of network.
        class_names (List[str]): Class names
        true_class (str): True class of given image
        k (int, optional): How many best predictions to plot as bar plot. Defaults to 10.
    """
    im_ax = fig.add_axes([0.0, 0.0, 2/3, 1.0])
    pred_ax = fig.add_axes([2/3, 1/8, 1/3, 7/8])
    plot_im(im_ax, image, true_class)
    plot_prediction_bar(pred_ax, predictions, class_names, k)
    

inv_transform = None

def un_normalize(image: torch.Tensor):
    global inv_transform
    if inv_transform is None:
        inv_transform = Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
            std=[1/0.229, 1/0.224, 1/0.255]
        )
    return inv_transform(image)
    


def get_transform(target_size=224):
    return Compose([Resize((target_size, target_size)), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def copy_batch_to_cuda(batch: Tuple[torch.Tensor, torch.Tensor]):
    sample, target = batch
    return sample.cuda(), target.cuda()


def get_model_device(model: nn.Module):
    """Returns device on which first parameters are located. We suppose that whole model is on same device.

    Args:
        model (nn.Module): model

    Returns:
        torch.device: model's device
    """
    return next(model.parameters()).device

def get_dataloader_device(dataloader: data.DataLoader):
    """Returns device on which first sample in first batch is located.

    Args:
        dataloader (data.DataLoader): a dataloader
    """
    for batch_idx, batch in enumerate(dataloader):
        sample, target = batch
        return sample[0].device
    