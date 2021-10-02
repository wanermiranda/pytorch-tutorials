from abc import abstractmethod
from functools import partial
from typing import Any, List, Tuple

import torch
from torch import nn
from torch.functional import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import Compose, Lambda, ToTensor


class BaseNetwork(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self._device = device

    def _send_to_device(self, device="cuda"):
        self._device = (
            "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"
        )
        self.to(device)
        if hasattr(self, "_optimizer_partial"):
            self._optimizer = self._optimizer_partial(self.parameters())
        print("Using {} device".format(self._device))

    @abstractmethod
    def _build_stack(self):
        """"""

    def fill_weights(self, uniform=False, weight=0.42):
        fill_func = partial(BaseNetwork._fill_weights, uniform=uniform, weight=weight)
        self.apply(fill_func)

    @staticmethod
    def _fill_weights(m, uniform=False, weight=0.42):
        classname = m.__class__.__name__
        # for every Linear layer in a model..
        if classname.find("Linear") != -1:
            # apply a uniform distribution to the weights and a bias=0
            if uniform:
                m.weight.data.uniform_(0.0, 1.0)
            else:
                m.weight.data.fill_(weight)

            m.bias.data.fill_(0)

    def _compile(self, device):
        if len(self._layout):
            self._build_stack()
            self._send_to_device(device=device)

    def load_states_from(self, source_net):
        self.load_state_dict(source_net.state_dict())
        
class NeuralNetwork(BaseNetwork):
    def __init__(
        self,
        layout: List[float] = [],
        device="cuda",
        loss_fn=nn.CrossEntropyLoss(),
        optimizer_partial=partial(torch.optim.SGD, lr=1e-3),
        class_labels: List[str] = [],
    ):
        """Starts a simple mlp using pytorch, following the layout provided and ReLu as a default activation function.

        Args:
            layout (List[float], optional): Network layout layer sizes and outputs in order. Defaults to [28*28, 512, 512, 10].
            device (str, optional): Device in wich the NN will run (cpu, cuda). Defaults to "cuda".
        """
        super().__init__(device)
        self._flatten = nn.Flatten()
        self._loss_fn = loss_fn
        self._optimizer_partial = optimizer_partial
        self._class_labels = class_labels
        self._layout = layout
        self._compile(device)

    def _build_stack(self):
        stack = []
        for i in range(len(self._layout) - 1):
            stack.append(nn.Linear(self._layout[i], self._layout[i + 1]))
            if i + 1 < len(self._layout) - 1:
                stack.append(nn.ReLU())

        self._network_stack = nn.Sequential(*stack)

    def forward(self, x):
        x = self._flatten(x)
        logits = self._network_stack(x)
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
        """Test the network using the dataset passed using a dataloder

        Args:
            test_loader (DataLoader): Test set with features X and labels y
        """

        _, avg_loss, accuracy = self._predict(test_loader)

        print(
            f"Test Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {avg_loss:>8f} \n"
        )

    def predict(self, data_loader: DataLoader):
        """Test the network using the dataset passed using a dataloder

        Args:
            data_loader (DataLoader): Data set with features X and labels y
        """

        result_set, _, _ = self._predict(data_loader)

        labels = result_set.tolist()

        if len(self._class_labels):
            labels = list(map(lambda i, labels=self._class_labels: labels[i], labels))  # type: ignore

        return labels

    def _predict(self, data_loader: DataLoader) -> Tuple[torch.Tensor, float, float]:
        """Given a set eval all the predictions and returns the classes, accuracy and loss

        Args:
            data_loader (DataLoader): Data set with features X and labels y

        Returns:
            Tuple[torch.Tensor, float, float]: classes, accuracy and loss
        """
        size = len(data_loader.dataset)  # type: ignore
        num_batches = len(data_loader)
        total_loss, correct_samples = 0.0, 0.0
        avg_loss, accuracy = 0.0, 0.0
        self.eval()

        with torch.no_grad():
            result_set = torch.tensor([], dtype=torch.int32, device=self._device)
            for X, y in data_loader:
                X, y = X.to(self._device), y.to(self._device)
                pred = self(X)

                y_ = pred.argmax(1)
                result_set = torch.cat([result_set, y_.type(torch.int32)], dim=0)

                total_loss += self._loss_fn(pred, y).item()
                correct_samples += (y_ == y).type(torch.float).sum().item()

        avg_loss = total_loss / float(num_batches)
        accuracy = correct_samples / float(size)

        return result_set, avg_loss, accuracy

    def save(self, path: str):
        file_dict = {
            "state": self.state_dict(),
            "_class_labels": self._class_labels,
            "_layout": self._layout,
            "_device": self._device,
            "_optimizer_partial": self._optimizer_partial,
            "_loss_fn": self._loss_fn,
        }
        torch.save(file_dict, path)

    def load(self, path: str):
        file_dict = torch.load(path)
        for attr in file_dict.keys():
            if hasattr(self, attr) and attr.startswith("_"):
                setattr(self, attr, file_dict[attr])

        self._build_stack()
        self.load_state_dict(file_dict["state"])
        self._send_to_device(device=self._device)

