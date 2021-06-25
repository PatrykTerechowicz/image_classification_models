import argparse
import models
import utils
import torch
import torch.nn as nn
import custom
import metrics
from clearml import Task
from torch.utils.data import DataLoader
from tqdm import tqdm
if __name__ == "__main__":
    task = Task.init("Klasyfikacja", "siamese-train")
    parser = argparse.ArgumentParser()
    parser.add_argument("-train_path", type=str)
    parser.add_argument("-valid_path", type=str)
    parser.add_argument("-net", type=str, default="mobilenetV2", help="use only mobilenetV2")
    parser.add_argument("-epochs", type=int, default=10)
    parser.add_argument("-batches", type=int, default=64)
    parser.add_argument("-num_classes", type=int, default=1000)
    parser.add_argument("-num_workers", type=int, default=16)
    parser.add_argument("-cuda", action="store_true")
    parser.add_argument("-model_path", default=None, type=str)
    parser.add_argument("-target_size", type=str, default=224)
    parser.add_argument("-margin", default=1.0, type=float)
    parser.add_argument("-coef", default=1e-3, type=float)
    args = parser.parse_args()
    task.update_task({"name": f"Train {args.net}", "tags": ["Siamese"]})

    # MODEL
    model = models.create_model_by_name(args.net, args.num_classes)
    classifier = model.classifier
    model.classifier = nn.Identity()
    optimizer, scheduler = models.get_optimizer_by_model(args.net, model, all_parameters=True)
    classifier = model.classifier
    model.classifier = nn.Identity()
    if args.model_path:
        save = torch.load(args.model_path)
        model.load_state_dict(save["model"])
    if args.cuda: model.cuda()
    transform = utils.get_transform(args.target_size)
    loss = nn.CrossEntropyLoss(reduction="mean")
    triplet = nn.TripletMarginLoss(margin=args.margin)
    # DATASET
    transform = utils.get_transform(args.target_size)
    ds_train = custom.SiameseImageFolder(args.train_path, transform=transform)
    ds_valid = custom.ImageFolder(args.valid_path, transform=transform)
    train_loader = DataLoader(ds_train, batch_size=args.batches, num_workers=args.num_workers, pin_memory=True, drop_last=True, shuffle=True)
    valid_loader = DataLoader(ds_valid, batch_size=args.batches, num_workers=args.num_workers, pin_memory=True)
    # LOOP
    for e in range(args.epochs):
        mean_loss = 0
        correct = 0
        total = 0
        for iter, data in tqdm(enumerate(train_loader), desc="Train", total=len(train_loader)):
            optimizer.zero_grad()
            A, P, N, A_targets, N_targets = data # A, P, N - anchors, positives, negatives
            total += A.shape[0]
            A = A.cuda()
            P = P.cuda()
            N = N.cuda()
            A_targets = A_targets.cuda()
            N_targets = N_targets.cuda()
            A_vecs = model(A) # returns vectors
            P_vecs = model(P) # returns vectors
            N_vecs = model(N) # returns vectors
            A_out = classifier(A_vecs) # returns predictions
            correct += metrics.accuracy(A_targets, A_out)
            P_out = classifier(P_vecs) # returns predictions
            N_out = classifier(N_vecs) # returns predictions
            triplet_loss = triplet(A_vecs, P_vecs, N_vecs)  # losses
            A_loss = loss(A_out, A_targets)                 # losses
            P_loss = loss(P_out, A_targets)                 # losses
            N_loss = loss(N_out, N_targets)                 # losses
            loss_sum = triplet_loss + args.coef*(A_loss + P_loss + N_loss)
            T = iter+1
            mean_loss = (mean_loss*iter + loss_sum)/T
            loss_sum.backward()
            optimizer.step()
        scheduler.step()
        task.logger.report_scalar("train_loss", "train", mean_loss, iteration=e)
        task.logger.report_scalar("accuracy", "train", correct/total, iteration=e)
        mean_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for iter, data in tqdm(enumerate(valid_loader), desc="Validating", total=len(valid_loader)):
                images, targets = data
                images = images.cuda()
                targets = targets.cuda()
                vecs = model(images)
                outs = classifier(vecs)
                correct += metrics.accuracy(targets, outs)
                l = loss(outs, targets)
                T = iter+1
                total += len(targets)
                mean_loss = (mean_loss*iter + l)/T
        task.logger.report_scalar("valid_loss", "valid", mean_loss, iteration=e)
        task.logger.report_scalar("accuracy", "valid", correct/total, iteration=e)
                