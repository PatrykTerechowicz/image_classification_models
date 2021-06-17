import torch.nn as nn
import torch
class CrossEntropyLossOnBadExamples(nn.Module):
    def __init__(self):
        super(CrossEntropyLossOnBadExamples, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss(reduction="sum") # we use sum there, will count mean later
    
    def forward(self, output, target):
        B, N = output.shape
        preds = output.argmax(dim=1)
        mask = preds != target # only train on badly classified data
        random_mask = torch.rand(B) < 0.25 # also 25% chance of including in training
        mask = mask | random_mask
        loss = self.cross_entropy(output[mask], target[mask])
        return loss/B # divide per B because we count every sample as training step
