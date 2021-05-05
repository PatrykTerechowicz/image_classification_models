from efficientnet_pytorch.model import EfficientNet
import torch
import torch.nn as nn
from torchvision.models import MNASNet, SqueezeNet, ShuffleNetV2, ResNet, AlexNet

def perpare_squeezenet(squeezenet: SqueezeNet, num_classes: int):
    final_conv = nn.Conv2d(512, num_classes, kernel_size=1)
    squeezenet.classifier[1] = final_conv

def prepare_shufflenet(shufflenet: ShuffleNetV2, num_classes: int):
    shufflenet.fc = nn.Linear(shufflenet.fc.in_features, num_classes)

def prepare_resnet(resnet: ResNet, num_classes: int):
    resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)

def prepare_mnasnet(mnasnet: MNASNet, num_classes: int):
    new_classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, num_classes))
    mnasnet.classifier = new_classifier

def prepare_alexnet(alexnet: AlexNet, num_classes: int):
    alexnet.classifier[-1] = nn.Linear(4096, num_classes)