from functools import partial
from typing import List

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import Compose, Lambda, ToTensor


class NeuralNetwork(nn.Module):
    def __init__(
        self,
        layout: List[float] = [28 * 28, 512, 512, 10],
        device="cuda",
        loss_fn=nn.CrossEntropyLoss(),
        optimizer_partial=partial(torch.optim.SGD, lr=1e-3),
    ):
        """Starts a simple mlp using pytorch, following the layout provided and ReLu as a default activation function.

        Args:
            layout (List[float], optional): Network layout layer sizes and outputs in order. Defaults to [28*28, 512, 512, 10].
            device (str, optional): Device in wich the NN will run (cpu, cuda). Defaults to "cuda".
        """
        super(NeuralNetwork, self).__init__()
        self._flatten = nn.Flatten()
        self._loss_fn = loss_fn
        self._optimizer_partial = optimizer_partial
        self._build_stack(layout=layout)
        self._send_to_device(device=device)

    def _build_stack(self, layout):
        self._layout = layout
        stack = []
        for i in range(len(self._layout) - 1):
            stack.append(nn.Linear(self._layout[i], self._layout[i + 1]))
            if i + 1 < len(self._layout) - 1:
                stack.append(nn.ReLU())

        self._linear_relu_stack = nn.Sequential(*stack)

    def _send_to_device(self, device="cuda"):
        self._device = (
            "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"
        )
        self.to(device)
        self._optimizer = self._optimizer_partial(self.parameters())
        print("Using {} device".format(self._device))

    def forward(self, x):
        x = self._flatten(x)
        logits = self._linear_relu_stack(x)
        return logits

    def fit(self, train_loader: DataLoader, test_loader: DataLoader = None, epochs=5):
        """Fit the model into the dataset provided

        Args:
            train_loader (DataLoader): Training dataset composed by features X and labels y
            test_loader (DataLoader): Testing dataset composed by features X and labels y
        """
        for e in range(epochs):
            print(f"Epoch {e+1}\n-------------------------------")
            self._fit_epoch(train_loader)
            if test_loader:
                self._test(test_loader)
        print("Done!")

    def _fit_epoch(self, train_loader):
        size = len(train_loader.dataset)  # type: ignore
        self.train()
        for batch, (X, y) in enumerate(train_loader):
            X, y = X.to(self._device), y.to(self._device)

            # Compute prediction error
            pred = self(X)
            loss = self._loss_fn(pred, y)

            # Backpropagation
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def _test(self, test_loader: DataLoader):
        size = len(test_loader.dataset)  # type: ignore
        num_batches = len(test_loader)
        self.eval()
        test_loss, correct = 0.0, 0.0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(self._device), y.to(self._device)
                pred = self(X)
                test_loss += self._loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= float(num_batches)
        correct /= float(size)
        print(
            f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
        )
