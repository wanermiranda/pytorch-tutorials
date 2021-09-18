import pytest
from torchvision.transforms.transforms import ToTensor
from torchvision import datasets

@pytest.fixture(scope="module")
def training_loader_fMNIST():
    return datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

@pytest.fixture(scope="module")
def testing_loader_fMNIST():
    return datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )