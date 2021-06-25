from re import L
import torch.nn as nn
import torch
import torch.nn.functional as F
import random

from torchvision.datasets import ImageFolder, folder
from typing import Optional, Callable, Any, Tuple

class CrossEntropyLossOnBadExamples(nn.Module):
    def __init__(self):
        super(CrossEntropyLossOnBadExamples, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss(reduction="sum") # we use sum there, will count mean later
    
    def forward(self, output, target):
        B, N = output.shape
        preds = output.argmax(dim=1)
        mask = preds != target # only train on badly classified data
        random_mask = torch.rand(B) < 0.25 # also 25% chance of including in training
        mask = mask | random_mask.cuda()
        loss = self.cross_entropy(output[mask], target[mask])
        return loss/B # divide per B because we count every sample as training step

class SMSoftmaxLoss(nn.Module):
    def __init__(self, soft_magrin=0.5):
        super(SMSoftmaxLoss, self).__init__()
        self.nnl = nn.NLLLoss(reduction="mean")
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.soft_margin = soft_magrin

    def forward(self, outputs, target):
        B, C = outputs.shape
        mask = F.one_hot(target, num_classes=C)
        outputs[mask.bool()] -= self.soft_margin
        soft_out = self.log_softmax(outputs)
        return self.nnl(soft_out, target)

class SiameseImageFolder(ImageFolder):
    def __init__(
            self,
            root: str,
            samples_per_epoch: int = 15000,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = folder.default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None
    ):
        super(SiameseImageFolder, self).__init__(root, transform, target_transform, loader, is_valid_file)
        new_imgs = dict.fromkeys(range(len(self.classes)), [])
        for path, class_index in self.samples:
            new_imgs[class_index].append(path)
        new_samples = []
        for _ in range(samples_per_epoch):
            pos, neg = random.sample(range(len(self.classes)), 2)
            anchor_path, pos_path = random.sample(new_imgs[pos], 2)
            neg_path = random.choice(new_imgs[neg])
            new_samples.append((anchor_path, pos_path, neg_path, pos, neg))
        self.samples = new_samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        anchor_path, positive_path, negative_path, positive_class, negative_class = self.samples[index]
        anchor = self.loader(anchor_path)
        positive = self.loader(positive_path)
        negative = self.loader(negative_path)
        if self.transform is not None:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative, positive_class, negative_class