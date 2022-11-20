from pathlib import Path
import random
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import numpy as np

from data.load_data import load_data
from models.model_choice import model_choice

SEED = 1111

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def evaluate(model, iterator, criterion, device):

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


def evaluater(model_name, model_save_path):

    # _, _, test_iterator = load_data(512)

    _, _, test_data = load_data(512)
    test_iterator = data.DataLoader(
        test_data, batch_size=512, shuffle=False, num_workers=2
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_model = model_choice(model_name).to(device)
    best_model.load_state_dict(
        torch.load(Path(model_save_path) / f"{model_name}_best.pt", map_location=device)
    )

    print(
        "number of parameters:",
        sum(p.numel() for p in best_model.parameters() if p.requires_grad),
    )

    criterion = nn.CrossEntropyLoss().to(device)
    test_loss, test_acc = evaluate(best_model, test_iterator, criterion, device)

    print("test_loss:", test_loss, "test_accuracy", test_acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="resnet")
    parser.add_argument(
        "--model-save-path",
        type=str,
        default="/scratch/hp2173/DL_MiniProject/model_checkpoint/",
    )
    opt = parser.parse_args()
    evaluater(**vars(opt))
