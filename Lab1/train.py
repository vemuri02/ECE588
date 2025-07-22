import os
import sys
import time
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

# Import model and helper functions
sys.path.insert(0, ".")

from model.model import AlexNet
from helper_functions.helper_evaluate import compute_accuracy
from helper_functions.helper_data import get_dataloaders_cifar10
from helper_functions.helper_train import train_classifier_simple_v1

def set_all_seeds(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def run_training(args):
    ##########################
    ### SETTINGS
    ##########################
    set_all_seeds(args.random_seed)

    if args.device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")

    ##########################
    ### Dataset
    ##########################
    train_transforms = transforms.Compose([
        transforms.Resize((70, 70)),
        transforms.RandomCrop((64, 64)),
        transforms.ToTensor()
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((70, 70)),
        transforms.CenterCrop((64, 64)),
        transforms.ToTensor()
    ])

    train_loader, valid_loader, test_loader = get_dataloaders_cifar10(
        batch_size=args.batch_size,
        num_workers=2,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        validation_fraction=0.1
    )

    ##########################
    ### Model
    ##########################
    model = AlexNet(num_classes=10)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    ##########################
    ### Training
    ##########################
    log_dict = train_classifier_simple_v1(
        num_epochs=args.num_epochs,
        model=model,
        optimizer=optimizer,
        device=device,
        train_loader=train_loader,
        valid_loader=valid_loader,
        logging_interval=50
    )

    # Return everything consistently
    return log_dict, train_loader, valid_loader, test_loader, model


def run_training_vars(random_seed, learning_rate, batch_size, num_epochs, device):
    args = argparse.Namespace(
        random_seed=random_seed,
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_epochs=num_epochs,
        device=device
    )
    return run_training(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AlexNet on CIFAR-10")

    parser.add_argument("--random_seed", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_epochs", type=int, default=40)
    parser.add_argument("--device", type=str, default="auto", choices=["cpu", "cuda", "auto"])

    args = parser.parse_args()

    log_dict, train_loader, valid_loader, test_loader, model = run_training(args)

    print("Training complete. Final log_dict keys:", log_dict.keys())
