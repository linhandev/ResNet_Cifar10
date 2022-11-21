import json
import random
import time
import argparse
import pathlib
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.utils.data as data
import torchvision.transforms as T
from torchvision.ops import focal_loss
import numpy as np

from data.load_data import load_data
from models.model_choice import model_choice

# fix random seeds to get reproducible results
SEED = 1111

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def train(
    model: torch.nn.Module,
    iterator: torch.utils.data.dataloader.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    writer: torch.utils.tensorboard.writer.SummaryWriter,
    n_sample: int,
):

    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for (x, y) in iterator:

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        y_pred = model(x)
        loss = criterion(y_pred, y)

        acc = calculate_accuracy(y_pred, y)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()
        n_sample += x.shape[0]

        writer.add_scalars(
            main_tag="Train",
            tag_scalar_dict={"acc": acc.item(), "loss": loss.item()},
            global_step=n_sample,
        )

    return epoch_loss / len(iterator), epoch_acc / len(iterator), n_sample


def evaluate(
    model: torch.nn.Module,
    iterator: torch.utils.data.dataloader.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
):

    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():

        for (x, y) in iterator:

            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)

            loss = criterion(y_pred, y)

            acc = calculate_accuracy(y_pred, y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc


def trainer(
    model_name: str,
    num_epoch: int,
    batch_size: int,
    learning_rate: float,
    model_save_path: pathlib.Path,
    do_aug: bool,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    bs_increase_at: [int],
    bs_increase_by: [int],
    opt: dict,
    loss: str,
    writer: torch.utils.tensorboard.writer.SummaryWriter,
):
    # 0. record start time
    tic = time.time()

    # 1. load data, setup model and other training related components
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # train_iterator, valid_iterator, test_iterator = load_data(batch_size, do_aug)

    train_data, valid_data, test_data = load_data(batch_size, do_aug)
    train_iterator = data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)
    valid_iterator = data.DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=8)

    model = model_choice(model_name).to(device)
    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if loss == "bce":
        criterion = nn.CrossEntropyLoss().to(device)
    elif loss == "focal":
        criterion = focal_loss
    else:
        print(f"{loss} loss not supported")

    if optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    elif optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    else:
        print(f"Invalid optimizer {optimizer}")
        exit()

    if scheduler is not None:
        if scheduler == "ReduceLROnPlateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)
        elif scheduler == "PolynomialLR":
            scheduler = optim.lr_scheduler.PolynomialLR(optimizer, total_iters=num_epoch, verbose=True)
        else:
            print(f"Invalid lr scheduler {scheduler}")
            exit()

    min_valid_loss = torch.inf
    min_eval_loss_at = 0
    n_sample = 0  # record based on number of samples trained, to keep the plots comparable across varying batch size

    # 2. run training and validataion
    for epoch in range(1, num_epoch + 1):

        for bs_increase_idx in range(len(bs_increase_at)):
            if epoch == bs_increase_at[bs_increase_idx]:
                batch_size = int(batch_size * bs_increase_by[bs_increase_idx])
                print(f"increasing bs to {batch_size}")
                train_iterator = data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
                valid_iterator = data.DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=4)

        # 2.1 train
        train_loss, train_acc, n_step = train(model, train_iterator, optimizer, criterion, device, writer, n_sample)
        # 2.2 validation
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion, device)
        writer.add_scalars("Val", {"acc": valid_acc, "loss": valid_loss}, epoch)

        # 2.3 adjust learning rate
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(valid_loss)
            else:
                scheduler.step()
        writer.add_scalar("Train/lr", optimizer.param_groups[0]["lr"], epoch)
        writer.add_scalar("Train/bs", batch_size, epoch)

        # 2.4 print training progress
        print(
            "Epoch:%d -> train_loss:%.5f, train_acc:%.5f || valid_loss:%.5f, valid_acc:%.5f || ETR:%.2f min"
            % (
                epoch,
                train_loss,
                train_acc,
                valid_loss,
                valid_acc,
                (time.time() - tic) / epoch * (num_epoch - epoch) / 60,
            )
        )

        # 2.5 save best model
        if min_valid_loss > valid_loss:
            torch.save(model.state_dict(), model_save_path / f"{model_name}_best.pt")
            min_valid_loss = valid_loss
            min_eval_loss_at = epoch
            print("The current min validation_loss: %.5f" % (min_valid_loss))

        print("---------------------------------------------")

    training_time = (time.time() - tic) / 60
    print(f"Training finished, took {training_time:.2f} mins")

    # 3. save training configuration and some stats for easier experiment management
    opt["num_param"] = num_param
    opt["min_valid_loss"] = min_valid_loss
    opt["training_time"] = training_time
    save_path = opt["model_save_path"]
    opt["run_id"] = save_path.name.split("-")[-1]
    opt["model_save_path"] = save_path._str
    opt["model"] = str(model)
    opt["min_eval_loss_at"] = min_eval_loss_at
    (save_path / "configs.json").write_text(json.dumps(opt))


def evaluater(
    model_name: str,
    model_save_path: pathlib.Path,
    writer: torch.utils.tensorboard.writer.SummaryWriter,
):
    # _, _, test_loader = load_data(512)

    _, _, test_data = load_data(512)
    test_iterator = data.DataLoader(test_data, batch_size=512, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_model = model_choice(model_name).to(device)
    best_model.load_state_dict(torch.load(Path(model_save_path) / f"{model_name}_best.pt"))

    criterion = nn.CrossEntropyLoss().to(device)
    test_loss, test_acc = evaluate(best_model, test_iterator, criterion, device)

    writer.add_scalars(
        main_tag="Test",
        tag_scalar_dict={"acc": test_acc, "loss": test_loss},
        global_step=1,
    )

    print("test_loss:", test_loss, "test_accuracy", test_acc)


if __name__ == "__main__":
    # 0. parse training parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="resnet")
    parser.add_argument("--num-epoch", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--do-aug", type=bool, default=True)
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--scheduler", type=str, default=None)
    parser.add_argument("--bs-increase-at", nargs="*", type=int, default=[])
    parser.add_argument("--bs-increase-by", nargs="*", type=int, default=[])
    parser.add_argument("--loss", type=str, default="bce")
    parser.add_argument("--model-save-path", type=str, default="./output")
    opt = vars(parser.parse_args())

    # attatch a run id to model save path
    opt["model_save_path"] = Path(opt["model_save_path"]) / f"{opt['model_name']}-{str(int(time.time()))}"
    opt["model_save_path"].mkdir(parents=True)
    print(f"This run saves to {opt['model_save_path']._str}")

    # tensorboard writer
    writer = SummaryWriter(log_dir=opt["model_save_path"] / "log")

    # 1. run training
    trainer(**opt, writer=writer, opt=opt)

    # 2. run evaluation
    evaluater(model_name=opt["model_name"], model_save_path=opt["model_save_path"], writer=writer)


"""
python train.py \
    --model-name 'resnet_de_resblock' \
    --num-epoch 200 \
    --batch-size 128 \
    --learning-rate 1e-3 \
    --optimizer AdamW
    # --scheduler ReduceLROnPlateau
    # --bs-increase 20
"""
