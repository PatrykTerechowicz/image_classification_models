from augment import CutMix
import torch
import datetime
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from torchvision.transforms.autoaugment import AutoAugment
from torchvision.transforms.transforms import Compose
import models
import utils
import metrics
import math
import os
import numpy as np
import custom
from dataset.dataset_preloader import DatasetPreloaded
from argparse import ArgumentParser
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from typing import Callable
from clearml import Task


def validate(model: nn.Module, valid_loader: data.DataLoader, loss_fn: Callable=nn.CrossEntropyLoss(reduction="mean"), cuda=False):
    valid_entropy_history = []
    correct_preds = 0
    correct_topk = 0
    total_samples = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(valid_loader), total=len(valid_loader), desc="Validating"):
            sample, target = batch
            if cuda: sample, target = utils.copy_batch_to_cuda(batch)
            total_samples += len(sample)
            net_out = model(sample)
            loss = loss_fn(net_out, target)
            valid_entropy_history.append(loss.item())
            correct_preds += metrics.accuracy(target, net_out)
            correct_topk += metrics.topk_accuracy(target, net_out)
    
    return np.array(valid_entropy_history).mean(), correct_preds/total_samples, correct_topk/total_samples


def train_one_epoch(model: nn.Module, optimizer: optim.Optimizer, train_loader: data.DataLoader, loss_fn: Callable=nn.CrossEntropyLoss(reduction="none"), cuda=False):
    entropy = []
    correct = 0
    train_samples = 0
    model.train()
    for batch_idx, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc="Training"):
        optimizer.zero_grad()
        sample, target = batch
        train_samples += len(sample)
        if cuda: sample, target = utils.copy_batch_to_cuda(batch)
        net_out = model(sample)
        loss = loss_fn(net_out, target)
        entropy.append(loss.item())
        loss.backward()
        optimizer.step()
        correct += metrics.accuracy(target, net_out)
    return np.array(entropy).mean(), correct/train_samples


if __name__ == "__main__":
    task = Task.init("Klasyfikacja", "train")
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("-train_path", type=str, help="path to training dataset directory")
    parser.add_argument("-valid_path", type=str, help="path to validation dataset directory")
    parser.add_argument("-topk", default=5, type=int, help="When calculating topk metric what k should be?")
    parser.add_argument("-epochs", default=10, type=int, help="how many epochs")
    parser.add_argument("-net", default="alexnet", type=str, help="name of model to train")
    parser.add_argument("-num_classes", default=1000, type=int, help="how many classes your model has to train")
    parser.add_argument("-target_size", default=224, type=int, help="how big images should be for training")
    parser.add_argument("-model_path", default=None, type=str, help="if given then will load model from this path before start")
    parser.add_argument("-batch_size", default=64, type=int, help="batch_size")
    parser.add_argument("-num_workers", default=2, type=int, help="0 for no multiprocessing for data loading")
    parser.add_argument("-cuda", action="store_true")
    parser.add_argument("-logistic_regression", action="store_true", help="if given then only train last classifier layer")
    parser.add_argument("-all_to_ram", action="store_true", help="if given then whole datasets will be preloaded on ram")
    parser.add_argument("-auto_augment", action="store_true")
    parser.add_argument("-cut_mix", type=int, default=0, help="probability(in percent ie. 20 for 0.2 probability) of using cut_mix, if 0 or not given then never use cutmix")
    parser.add_argument("-train_on_wrongly", action="store_true", help="If given then will only train on samples that were wrongly classified")
    args = parser.parse_args()
    task.update_task({"name": f"Trenowanie {args.net}"})
    print(f"Loading structure of {args.net}")
    model = models.create_model_by_name(args.net, args.num_classes)
    optimizer, scheduler = models.get_optimizer_by_model(args.net, model, all_parameters=not args.logistic_regression)
    if args.model_path:
        print(f"Loading weights from {args.model_path}")
        save = torch.load(args.model_path)
        model.load_state_dict(save['model'])
    print(f"Cuda is set to {args.cuda}")
    if args.cuda: model.cuda()

    transform = utils.get_transform(args.target_size)
    train_transform = transform
    if args.auto_augment:
        train_transform = Compose([train_transform, AutoAugment()])
    if args.cut_mix:
        train_transform = Compose([train_transform, CutMix(prob=float(args.cut_mix/100))])
    ds_train = ImageFolder(args.train_path, transform=train_transform)
    ds_valid = ImageFolder(args.valid_path, transform=transform)

    if args.all_to_ram:
        print("Loading train dataset to RAM")
        ds_train = DatasetPreloaded(ds_train)
        print("Loading valid dataset to RAM")
        ds_valid = DatasetPreloaded(ds_valid)
    
    loader_settings = {"batch_size": args.batch_size, "num_workers": args.num_workers}
    train_loader = data.DataLoader(ds_train, drop_last=True, shuffle=True, **loader_settings)
    valid_loader = data.DataLoader(ds_valid, **loader_settings)

    train_samples = len(ds_train)
    valid_samples = len(ds_valid)
    train_batches = math.floor(train_samples/args.batch_size)
    valid_batches = math.ceil(valid_samples/args.batch_size)

    loss_fn = nn.CrossEntropyLoss(reduction="mean")
    if args.train_on_wrongly: loss_fn = custom.CrossEntropyLossOnBadExamples()
    for epoch in range(args.epochs):
        entropy, accuracy = train_one_epoch(model, optimizer, train_loader, loss_fn, cuda=args.cuda)
        task.logger.report_scalar("entropy", "train", entropy, iteration=epoch)
        task.logger.report_scalar("accuracy", "train", accuracy, iteration=epoch)
        entropy, accuracy, _ = validate(model, valid_loader, loss_fn=loss_fn, cuda=args.cuda)
        task.logger.report_scalar("entropy", "valid", entropy, iteration=epoch)
        task.logger.report_scalar("accuracy", "valid", accuracy, iteration=epoch)
    state = model.state_dict()
    date_s = datetime.datetime.today().strftime("%d-%m-%y_%H-%M")
    out_dir = f"D:\\Modele\\klasyfikacja\\{args.net}\\date_s"
    try:
        os.makedirs(out_dir)
    except Exception:
        pass
    torch.save({"model": state}, f"{out_dir}\\final.pth")
