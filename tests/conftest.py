import pytest
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms.transforms import ToTensor


@pytest.fixture(scope="module")
def training_loader_fMNIST():
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )
    return DataLoader(training_data, batch_size=64)


@pytest.fixture(scope="module")
def testing_loader_fMNIST():
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )
    return DataLoader(test_data, batch_size=64)
