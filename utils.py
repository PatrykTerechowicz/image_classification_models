import torch.nn as nn
import torch.utils.data as data
import torch
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from typing import Tuple


def get_transform(target_size=224):
    return Compose([Resize(target_size), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


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
    