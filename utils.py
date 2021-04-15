import torch.nn as nn
import torch.data as data
import torch

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