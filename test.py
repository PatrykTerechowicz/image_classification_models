import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import argparse
import metrics
import utils
import models
import math
import plot_utils 
import torchvision
from torch.utils.tensorboard import SummaryWriter
from typing import Callable

parser = argparse.ArgumentParser()

parser.add_argument("--model_name", default="alexnet", help="Name of model.")
parser.add_argument("--num_classes", default=1000, type=int, help="On how many classes model has been trained?")
parser.add_argument("--class_names", default="./classes.txt", help="txt file containing name classes, used for ensuring that labels aren't mismatched.")
parser.add_argument("--path", default="./best.pth", help="Relative or absolute path to model.")
parser.add_argument("--ds_path", default="./DS/test", help="Relative or absolute path to dataset.")
parser.add_argument("--target_size", default=224, type=int, help="How big should images be for testing? Should be at least 224.")
parser.add_argument("--save_fig", default=False, type=bool, help="Should figures be saved?")
parser.add_argument("--log_dir", default="./logs", type=str)
parser.add_argument("--test_name", default="test", type=str, help="Name the test so its distinguishable from rest.")
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--num_workers", default=2, type=int)
parser.add_argument("--cuda", action="store_true", help="Option for enabling testing using cuda.")
parser.add_argument("--notebook", action="store_true", help="Use this if you want to run this script on jupyter notebooks.")

args = parser.parse_args()

assert not args.save_fig, "save_fig currently unsuported" # TODO: add support

if args.notebook: from tqdm.notebook import tqdm
else: from tqdm import tqdm


def test(model: nn.Module, test_loader: data.DataLoader, summary_writer: SummaryWriter, total_batches, data_len, loss_fn: Callable=F.cross_entropy, save_fig=False, cuda=True):
    """Tests models and returns accuracy, top-k accuracy and crosscategorical_loss.

    Args:
        model (nn.Module): Model to be trained
        test_loader (data.DataLoader): dataloader used for loading data

    Returns:
        [type]: [description]
    """
    correct_predictions = 0
    correct_topk_predictions = 0
    total_loss = 0
    for batch_idx, batch in tqdm(enumerate(test_loader), total=total_batches):
        sample, target = batch
        if cuda:
            sample = sample.cuda()
            target = target.cuda()
        net_out = model(sample)
        correct_predictions += metrics.accuracy(target, net_out)
        correct_topk_predictions += metrics.topk_accuracy(target, net_out)
        loss = loss_fn(net_out, target)
        summary_writer.add_scalars("test", {"loss": loss})
        total_loss += torch.sum(loss)
        if args.save_fig:
            figs = None
            summary_writer.add_figure("predictions", figs)
    
    accuracy = correct_predictions/data_len
    topk_accuracy = correct_topk_predictions/data_len
    return accuracy, topk_accuracy, loss/data_len

if __name__ == "__main__":
    print(f"Defining structure of {args.model_name}")
    model = models.create_model_by_name(args.model_name, args.num_classes)
    if args.cuda: model.cuda()
    optimizer, scheduler = models.get_optimizer_by_model(args.model_name, model)
    print(f"{args.model_name} has {models.parameters(model)} parameters")
    print(f"Loading weights of {args.model_name} from {args.path}")
    saved_state_dict = torch.load(args.path)["model"]
    model.load_state_dict(saved_state_dict)

    dataset = torchvision.datasets.ImageFolder(args.ds_path, transform=utils.get_transform(args.target_size))
    class_names = []
    with open(args.class_names) as file:
        class_names = [l.strip("\n\r") for l in file.readlines()]
    assert class_names == dataset.classes, f"Class names dont match!" # TODO: print first not equal elements
    
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.cuda)
    # loading ended
    print(f"Creating tensorboard logdir in {args.log_dir}/{args.test_name}")
    logdir = f"{args.log_dir}/{args.test_name}"
    summary_writer = SummaryWriter(logdir)
    
    sample1 = dataset[0][0].unsqueeze(0)
    if args.cuda: sample1 = sample1.cuda()
    summary_writer.add_graph(model, sample1)
    
    accuracy, topk_accuracy, loss = test(model, data_loader, summary_writer, math.ceil(len(dataset)/args.batch_size), len(dataset), save_fig=args.save_fig, cuda=args.cuda)
    summary_writer.add_text("test", f"{args.model_name} has achieved:\n->accuracy: {accuracy:.2%}\n->TopK: {topk_accuracy:.2%}\n->Mean Loss: {loss:.6f}")
    print(f"End of testing. Saved logs in {logdir}.")
    summary_writer.close()