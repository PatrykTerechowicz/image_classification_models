import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import model_utils

from efficientnet_pytorch import EfficientNet, VALID_MODELS
from typing import Tuple

# not supported: shufflenet_v2_x1_5; shufflenet_v2_x2_0; mnasnet0_75
model_names = ["mobilenetV2", "mobilenetV3_small", "mobilenetV3_large", "densenet121", "densenet161", "densenet169", "densenet201", 
                "squeezenet1_0", "squeezenet1_1", "shufflenet_v2_x0_5", "shufflenet_v2_x1_0",
                "resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "wide_resnet50_2", "wide_resnet101_2", "mnasnet0_5",
                "mnasnet1_0", "alexnet"
                ]
model_names.extend(VALID_MODELS)


def parameters(model):
    return sum(p.numel() for p in model.parameters())


def create_model_by_name(model_name, out_classes) -> nn.Module:
    assert model_name in model_names, f"script doesn't support {model_name}, supported scripts: {model_names}"
    model = None
    if model_name == "mobilenetV2":
        model = torchvision.models.mobilenet_v2(pretrained=True)
        model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(model.last_channel, out_classes, bias=True))
    elif model_name == "mobilenetV3_small":
        model = torchvision.models.mobilenet_v3_small(pretrained=True)
        classifier = model.classifier
        new_lin = nn.Linear(classifier[-1].in_features, out_classes)
        model.classifier[-1] = new_lin
    elif model_name == "mobilenetV3_large":
        model = torchvision.models.mobilenet_v3_large(pretrained=True)
        classifier = model.classifier
        new_lin = nn.Linear(classifier[-1].in_features, out_classes)
        model.classifier[-1] = new_lin
    elif model_name == "densenet121":
        model = torchvision.models.densenet121(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, out_classes)
    elif model_name == "densenet161":
        model = torchvision.models.densenet161(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, out_classes)
    elif model_name == "densenet169":
        model = torchvision.models.densenet169(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, out_classes)
    elif model_name == "densenet201":
        model = torchvision.models.densenet201(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, out_classes)
    elif model_name == "squeezenet1_0":
        model = torchvision.models.squeezenet1_0(pretrained=True)
        model_utils.perpare_squeezenet(model, out_classes)
    elif model_name == "squeezenet1_1":
        model = torchvision.models.squeezenet1_1(pretrained=True)
        model_utils.perpare_squeezenet(model, out_classes)
    elif model_name == "shufflenet_v2_x0_5":
        model = torchvision.models.shufflenet_v2_x0_5(pretrained=True)
        model_utils.prepare_shufflenet(model, out_classes)
    elif model_name == "shufflenet_v2_x1_0":
        model = torchvision.models.shufflenet_v2_x1_0(pretrained=True)
        model_utils.prepare_shufflenet(model, out_classes)
    elif model_name == "resnet18":
        model = torchvision.models.resnet18(pretrained=True)
        model_utils.prepare_resnet(model, out_classes)
    elif model_name == "resnet34":
        model = torchvision.models.resnet34(pretrained=True)
        model_utils.prepare_resnet(model, out_classes)
    elif model_name == "resnet50":
        model = torchvision.models.resnet50(pretrained=True)
        model_utils.prepare_resnet(model, out_classes)
    elif model_name == "resnet101":
        model = torchvision.models.resnet101(pretrained=True)
        model_utils.prepare_resnet(model, out_classes)
    elif model_name == "resnet152":
        model = torchvision.models.resnet152(pretrained=True)
        model_utils.prepare_resnet(model, out_classes)
    elif model_name == "wide_resnet50_2":
        model = torchvision.models.wide_resnet50_2(pretrained=True)
        model_utils.prepare_resnet(model, out_classes)
    elif model_name == "wide_resnet101_2":
        model = torchvision.models.wide_resnet101_2(pretrained=True)
        model_utils.prepare_resnet(model, out_classes)
    elif model_name == "mnasnet0_5":
        model = torchvision.models.mnasnet0_5(pretrained=True)
        model_utils.prepare_mnasnet(model, out_classes)
    elif model_name == "mnasnet1_0":
        model = torchvision.models.mnasnet1_0(pretrained=True)
        model_utils.prepare_mnasnet(model, out_classes)
    elif model_name == "alexnet":
        model = torchvision.models.alexnet(pretrained=True)
        model_utils.prepare_alexnet(model, out_classes)
    elif model_name in VALID_MODELS:
        model = EfficientNet.from_pretrained(model_name, num_classes=out_classes, advprop=False)
    return model


def get_optimizer_by_model(model_name, model: torch.nn.Module, all_parameters=False) -> Tuple[optim.Optimizer, optim.lr_scheduler.LambdaLR]:
    assert model_name in model_names, f"script doesn't support {model_name}"
    optimizer = None
    optimizer_config = None
    optimizer_class = None
    trainable_parameters = None
    schedule_fn = lambda x: x
    if model_name == "mobilenetV2":
        optimizer_config = {
            "lr": 0.0045,
            "weight_decay": 4*1e-5,
            "momentum": 0.9
        }
        optimizer_class = torch.optim.RMSprop
        trainable_parameters = model.classifier.parameters()
        schedule_fn = lambda epoch: 0.98 ** epoch
    if model_name in ["mobilenetV3_small", "mobilenetV3_large"]:
        optimizer_config = {
            "lr": 0.1,
            "weight_decay": 1e-5,
            "momentum": 0.9
        }
        optimizer_class = torch.optim.RMSprop
        trainable_parameters = model.classifier[-1].parameters()
        schedule_fn = lambda epoch: max(1-epoch/10+0.1, 1e-5)
    if model_name in ["densenet121", "densenet161", "densenet169", "densenet201"]:
        optimizer_config = {
            "lr": 0.01,
            "nesterov": True,
            "weight_decay": 1e-4,
            "momentum": 0.9
        }
        optimizer_class = torch.optim.SGD
        trainable_parameters = model.classifier.parameters()
        schedule_fn = lambda epoch: 0.98 ** epoch 
    if model_name in ["squeezenet1_0", "squeezenet1_1"]:
        optimizer_config = {
            "lr": 0.04,
            "weight_decay": 0.0002,
            "momentum": 0.9,
            "nesterov": True
        }
        optimizer_class = torch.optim.SGD
        trainable_parameters = model.classifier[1].parameters()
        schedule_fn = lambda epoch: max(1-epoch/10+0.1, 1e-5)
    if model_name in ["shufflenet_v2_x0_5", "shufflenet_v2_x1_0"]:
        optimizer_config = {
            "lr": 0.0045,
            "weight_decay": 4*1e-5,
            "momentum": 0.9
        }
        optimizer_class = torch.optim.RMSprop
        trainable_parameters = model.fc.parameters()
        schedule_fn = lambda epoch: 0.98 ** epoch
    if model_name in ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "wide_resnet50_2", "wide_resnet101_2"]:
        optimizer_config = {
            "lr": 0.1,
            "weight_decay": 0.0001,
            "momentum": 0.9,
            "nesterov": True
        }
        optimizer_class = torch.optim.SGD
        trainable_parameters = model.fc.parameters()
        schedule_fn = lambda epoch: 0.75 ** epoch
    if model_name in ["mnasnet0_5", "mnasnet1_0"]:
        optimizer_config = {
            "lr": 0.256,
            "momentum": 0.9,
            "weight_decay": 1e-5,
        }
        optimizer_class = torch.optim.RMSprop
        trainable_parameters = model.classifier.parameters()
        schedule_Fn = lambda epoch: 0.75 ** epoch
    if model_name == "alexnet":
        optimizer_config = {
            "lr": 0.0045,
            "weight_decay": 4*1e-5,
            "momentum": 0.9
        }
        optimizer_class = torch.optim.RMSprop
        trainable_parameters = model.classifier[-1].parameters()
        schedule_fn = lambda epoch: 0.98 ** epoch
    if model_name in VALID_MODELS:
        pass

    if all_parameters: trainable_parameters = model.parameters()

    optimizer = optimizer_class(trainable_parameters, **optimizer_config)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=schedule_fn)
    return optimizer, scheduler


if __name__ == "__main__":
    for model_name in model_names:
        model = create_model_by_name(model_name, 50)
        a = torch.zeros((1, 3, 224, 224))
        optimizer, scheduler = get_optimizer_by_model(model_name, model)
        print(f"{model_name}: params={parameters(model)}; out_size={model(a).squeeze(0).shape}")