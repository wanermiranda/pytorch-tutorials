import pytest
from torch._C import layout
from wompth.models.base import NeuralNetwork

def test_model_build(training_loader_fMNIST):
    nn = NeuralNetwork(layout=[28*28, 128, 128, 5])
    assert nn.__str__() == 'NeuralNetwork(\n  (_flatten): Flatten(start_dim=1, end_dim=-1)\n  (_loss_fn): CrossEntropyLoss()\n  (_linear_relu_stack): Sequential(\n    (0): Linear(in_features=784, out_features=128, bias=True)\n    (1): ReLU()\n    (2): Linear(in_features=128, out_features=128, bias=True)\n    (3): ReLU()\n    (4): Linear(in_features=128, out_features=5, bias=True)\n  )\n)'