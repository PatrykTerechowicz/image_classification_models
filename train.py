import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import models
import utils
import metrics
import math
from dataset.dataset_preloader import DatasetPreloaded
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from typing import Callable


def validate(model: nn.Module, valid_loader: data.DataLoader, total_batches: int, total_samples: int, loss_fn: Callable=nn.CrossEntropyLoss(reduction="mean"), cuda=False):
    valid_entropy_history = []
    correct_preds = 0
    correct_topk = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(valid_loader), total=total_batches, desc="Validating"):
            sample, target = batch
            if cuda: sample, target = utils.copy_batch_to_cuda(batch)
            net_out = model(sample)
            loss = loss_fn(net_out, target)
            valid_entropy_history.append(loss.item())
            correct_preds += metrics.accuracy(target, net_out)
            correct_topk += metrics.topk_accuracy(target, net_out)
    
    return valid_entropy_history, correct_preds/total_samples, correct_topk/total_samples


def train_one_epoch(model: nn.Module, optimizer: optim.Optimizer, scheduler: optim.lr_scheduler.LambdaLR, train_loader: data.DataLoader, total_batches: int, train_samples: int, loss_fn: Callable=nn.CrossEntropyLoss(reduction="mean"),cuda=False):
    entropy_history = []
    correct = 0
    model.train()
    for batch_idx, batch in tqdm(enumerate(train_loader), total=total_batches, desc="Training"):
        optimizer.zero_grad()
        sample, target = batch
        if cuda: sample, target = utils.copy_batch_to_cuda(batch)
        net_out = model(sample)
        loss = loss_fn(net_out, target)
        entropy_history.append(loss.item())
        loss.backward()
        optimizer.step()
        correct += metrics.accuracy(target, net_out)
    scheduler.step()
    return entropy_history, correct/train_samples

        
def save_entropy(summary_writer, entropy_history, name, epoch, total_batches):
    for n, e in enumerate(entropy_history):
        summary_writer.add_scalar(name, e, global_step=(epoch*total_batches + n))


def train(model: nn.Module, optimizer: optim.Optimizer, scheduler: optim.lr_scheduler.LambdaLR,
    train_loader: data.DataLoader, valid_loader: data.DataLoader, summary_writer: SummaryWriter,
    train_batches: int, train_samples: int, valid_batches: int, valid_samples: int, epochs: int, cuda=False):
    for epoch in range(epochs):
        print(f"Starting epoch {epoch+1}")
        entropy_history, train_accuracy = train_one_epoch(model, optimizer, scheduler, train_loader, train_batches, train_samples, cuda=cuda)
        save_entropy(summary_writer, entropy_history, "train/entropy", epoch=epoch, total_batches=train_batches)
        summary_writer.add_scalar("train/accuracy", train_accuracy*100, global_step=epoch)
        valid_entropy_history, accuracy, topk = validate(model, valid_loader, valid_batches, valid_samples, cuda=cuda)
        summary_writer.add_scalar("valid/accuracy", accuracy*100, global_step=epoch)
        summary_writer.add_scalar("valid/topk", topk*100, global_step=epoch)
        save_entropy(summary_writer, valid_entropy_history, "valid/entropy", epoch=epoch, total_batches=valid_batches)


if __name__ == "__main__":
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("--train_path", type=str, help="path to training dataset directory")
    parser.add_argument("--valid_path", type=str, help="path to validation dataset directory")
    parser.add_argument("--topk", default=5, type=int, help="When calculating topk metric what k should be?")
    parser.add_argument("--epochs", default=10, type=int, help="how many epochs")
    parser.add_argument("--net", default="alexnet", type=str, help="name of model to train")
    parser.add_argument("--num_classes", default=1000, type=int, help="how many classes your model has to train")
    parser.add_argument("--target_size", default=224, type=int, help="how big images should be for training")
    parser.add_argument("--continue_training", action="store_true", help="use this if you want continue from checkpoint")
    parser.add_argument("--model_path", default="./best.pth", type=str, help="if also continue_training is set to true then will use this model to start")
    parser.add_argument("--log_dir", default="./logs", type=str, help="where to put tensorboard logs")
    parser.add_argument("--log_name", default="train", type=str, help="name of experiment, can be anything")
    parser.add_argument("--batch_size", default=64, type=int, help="batch_size")
    parser.add_argument("--num_workers", default=2, type=int, help="0 for no multiprocessing for data loading")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--all_parameters", action="store_true", help="if given then whole models weight will be updated")
    parser.add_argument("--all_to_ram", action="store_true", help="if given then whole datasets will be preloaded on ram")
    args = parser.parse_args()

    print(f"Loading structure of {args.net}")
    model = models.create_model_by_name(args.net, args.num_classes)
    optimizer, scheduler = models.get_optimizer_by_model(args.net, model, all_parameters=args.all_parameters)
    if args.continue_training:
        print(f"Loading weights from {args.model_path}")
        save = torch.load(args.model_path)
        model.load_state_dict(save['model'])
    print(f"Cuda is set to {args.cuda}")
    if args.cuda: model.cuda()
    print(f"Writing logs to {args.log_dir}")
    summary_writer = SummaryWriter(f"{args.log_dir}/{args.net}/{args.log_name}")

    transform = utils.get_transform(args.target_size)
    ds_train = ImageFolder(args.train_path, transform=transform)
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

    train(model, optimizer, scheduler, train_loader, valid_loader, summary_writer, train_batches, train_samples, valid_batches, valid_samples, args.epochs, cuda=args.cuda)
    state = model.state_dict()
    torch.save({"model": state}, f"{args.log_dir}/{args.net}/{args.log_name}/final.pth")
