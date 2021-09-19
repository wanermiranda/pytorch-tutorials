import random
from collections import deque, namedtuple
from functools import partial
from typing import List, Tuple

import torch.nn.functional as F
import torch.optim as optim
from torch import nn

from wompth.models.base import BaseNetwork

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

LayerConf = namedtuple("LayerConf", ("input", "kernel_size", "stride", "batch_norm"))

ScreenDims = namedtuple("ScreenDims", ("height", "width"))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(BaseNetwork):
    def __init__(
        self,
        device="cuda",
        layout: List[LayerConf] = [],
        screen_dims: ScreenDims = ScreenDims(height=0, width=0),
        outputs: float = 0,
        activation_func=partial(F.relu),
        optimizer_partial=partial(optim.RMSprop),
    ):
        super().__init__(device=device)
        self._layout: List[LayerConf] = layout
        self._screen_dims = screen_dims
        self._outputs = outputs
        self._activation_func = activation_func
        self._compile(self._device)

    def _build_stack(self):
        stack = []
        conv_w = None
        conv_h = None
        for i in range(len(self._layout)):
            layer = self._layout[i]

            if i < len(self._layout) - 1:
                next_layer = self._layout[i + 1]
                stack.append(
                    nn.Conv2d(
                        layer.input,
                        next_layer.input,
                        kernel_size=layer.kernel_size,
                        stride=layer.stride,
                    )
                )
            else:  # Last layer gets the output size from the batch norm size
                stack.append(
                    nn.Conv2d(
                        layer.input,
                        layer.batch_norm,
                        kernel_size=layer.kernel_size,
                        stride=layer.stride,
                    )
                )

            stack.append(nn.BatchNorm2d(layer.batch_norm))

            if not conv_h:  # Define the conv size based on the first layer
                conv_w = DQN.conv2d_size_out(
                    self._screen_dims.width,
                    kernel_size=layer.kernel_size,
                    stride=layer.stride,
                )
                conv_h = DQN.conv2d_size_out(
                    self._screen_dims.height,
                    kernel_size=layer.kernel_size,
                    stride=layer.stride,
                )
            else:  # recursivelly calculate the conv size
                conv_w = DQN.conv2d_size_out(
                    conv_w, kernel_size=layer.kernel_size, stride=layer.stride
                )
                conv_h = DQN.conv2d_size_out(
                    conv_h, kernel_size=layer.kernel_size, stride=layer.stride
                )

        # Setting the network layers
        for i, layer in enumerate(stack):
            setattr(self, f"_{type(layer).__name__}_{i//2}", layer)

        self._linear_input_size = conv_w * conv_h * self._layout[-1].batch_norm
        self._head = nn.Linear(self._linear_input_size, self._outputs)
        self._network_stack = stack

    @staticmethod
    def conv2d_size_out(size, kernel_size=5, stride=2):
        return (size - (kernel_size - 1) - 1) // stride + 1

    def forward(self, x):
        x = x.to(self._device)
        for i in range(len(self._network_stack) / 2):
            batch_norm = self._network_stack[i + 1]
            conv = self._network_stack[i]
            x = self._activation_func(batch_norm(conv(x)))

        return self._head(x.view(x.size(0), -1))
