from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def create_datasets(data_root):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_dataset = datasets.MNIST(
        root=data_root,
        train=True,
        transform=transform,
        download=True,
    )

    test_dataset = datasets.MNIST(
        root=data_root,
        train=False,
        transform=transform,
        download=True,
    )

    return train_dataset, test_dataset


def create_loaders(train_dataset, test_dataset, batch_size):
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, test_loader
