import torch
import torch.nn as nn
import torch.optim as optim

from .cli import prompt_device, prompt_hyperparameters
from .config import DEFAULT_HYPERPARAMS
from .data import create_datasets, create_loaders
from .model import SimpleCNN
from .train import train_model


def main():
    device = prompt_device()
    print("Using device:", device)

    params = prompt_hyperparameters(DEFAULT_HYPERPARAMS)

    train_dataset, test_dataset = create_datasets("./data")
    train_loader, test_loader = create_loaders(
        train_dataset,
        test_dataset,
        params["batch_size"],
    )

    model = SimpleCNN().to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])

    print("model is training, please wait")

    best_test_acc = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=params["num_epochs"],
        save_path="best_mnist_cnn.pth",
    )

    print("training completed")
    print(f"Accuracy: {best_test_acc:.4f}")


if __name__ == "__main__":
    main()
