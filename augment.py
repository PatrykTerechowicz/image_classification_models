from PIL.Image import Image
import torch
import torch.nn.functional as F
import math
from random import random, randint, choices
from torchvision.transforms import Compose, AutoAugment, ToTensor, Resize


class CutMix(object):
    def __init__(self, prob=0.2, num_classes=-1):
        self.prob = prob
        self.num_classes = num_classes
        self.prev_sample = None
    
    def __call__(self, sample):
        images, labels = sample
        if self.prev_sample is None:
            self.prev_sample = sample
            return sample
        B, C, H, W = images.shape
        pop = [True, False]
        prev_im, prev_labels = self.prev_sample
        batches = choices(pop, weights=[self.prob, 1-self.prob], k=B)
        one_hot_labels = F.one_hot(labels, num_classes=self.num_classes)
        prev_one_hot_labels = F.one_hot(prev_labels, num_classes=self.num_classes)
        new_labels = torch.tensor(one_hot_labels, dtype=torch.float32)
        lambdas = torch.tensor([random()/4 + 0.15 for _ in range(B)], dtype=torch.float32)
        new_labels[batches] = prev_one_hot_labels[batches]*lambdas[batches, None] + one_hot_labels[batches]*(1-lambdas[batches, None])
        M = torch.zeros_like(images, dtype=torch.bool)
        for b in range(B):
            l_sqrt = math.sqrt(lambdas[b])
            r_w = math.floor(W*l_sqrt)
            r_h = math.floor(H*l_sqrt)
            r_x = randint(0, W-r_w)
            r_y = randint(0, H-r_h)
            M[b, :, r_y:r_y+r_h, r_x:r_x+r_w] = True
        images[batches] = (~M[batches])*images[batches] + M[batches]*prev_im[batches]
        return images, new_labels


def get_transform(prob=0.2):
    transform = Compose([AutoAugment(), CutMix(prob=prob)])


if __name__ == "__main__":
    import argparse
    import utils
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder
    from torchvision.transforms import Compose
    parser = argparse.ArgumentParser()
    parser.add_argument("ds_path")
    args = parser.parse_args()
    ds = ImageFolder(args.ds_path, transform=utils.get_transform(448))
    cut_mix = CutMix(prob=0.75, num_classes=len(ds.classes))
    data_loader = DataLoader(ds, batch_size=16, shuffle=True)
    n = 0
    for batch in data_loader:
        n += 1
        batch_2 = cut_mix(batch)
        if n > 1: break

    ims, labels = batch_2
    ims = utils.un_normalize(ims)
    B, H, W, C = ims.shape
    for b in range(B):
        plt.imshow(ims[b].permute([1, 2, 0]))
        args = torch.argsort(labels[b], descending=True)[:2]
        s1 = f"{ds.classes[args[0]]}: {labels[b, args[0]]:.2%}\n"
        s2 = f"{ds.classes[args[1]]}: {labels[b, args[1]]:.2%}"
        plt.annotate("".join([s1, s2]), xy=(50, 50))
        plt.show()
