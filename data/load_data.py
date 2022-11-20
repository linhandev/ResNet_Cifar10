import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import torchvision.transforms as transforms
import torchvision.datasets as datasets

import random

SEED = 1111

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def load_data(batch_size, do_aug=True):

    ROOT = ".data"
    train_data = datasets.CIFAR10(root=ROOT, train=True, download=True)

    means = train_data.data.mean(axis=(0, 1, 2)) / 255
    stds = train_data.data.std(axis=(0, 1, 2)) / 255

    test_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=means, std=stds)])

    train_transforms = (
        transforms.Compose(
            [
                transforms.RandomRotation(5),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomCrop(32, padding=2),
                transforms.ToTensor(),
                transforms.Normalize(mean=means, std=stds),
            ]
        )
        if do_aug
        else test_transforms
    )

    train_data = datasets.CIFAR10(ROOT, train=True, download=True, transform=train_transforms)

    test_data = datasets.CIFAR10(ROOT, train=False, download=True, transform=test_transforms)

    VALID_RATIO = 0.9

    n_train_examples = int(len(train_data) * VALID_RATIO)
    n_valid_examples = len(train_data) - n_train_examples

    train_data, valid_data = data.random_split(train_data, [n_train_examples, n_valid_examples])

    # train_iterator = data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)

    # valid_iterator = data.DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=4)

    # test_iterator = data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=8)

    # return train_iterator, valid_iterator, test_iterator

    return train_data, valid_data, test_data
