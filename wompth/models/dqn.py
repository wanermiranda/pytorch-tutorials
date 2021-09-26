import random
from collections import deque, namedtuple
from functools import partial
from typing import List, NamedTuple, Tuple
from dataclasses import dataclass
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
import torch
import math
from gym.core import Env
from wompth.models.base import BaseNetwork
from itertools import count

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

LayerConf = namedtuple("LayerConf", ("input", "kernel_size", "stride", "batch_norm"))

ScreenDims = namedtuple("ScreenDims", ("height", "width"))
@dataclass
class DQNConf:
    BATCH_SIZE:float = 128
    GAMMA:float = 0.999
    EPS_START:float = 0.9
    EPS_END:float = 0.05
    EPS_DECAY:int = 200
    TARGET_UPDATE:int = 10
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
        conf:DQNConf = DQNConf(),
        layout: List[LayerConf] = [],
        screen_dims: ScreenDims = ScreenDims(height=0, width=0),
        outputs: float = 0,
        activation_func=partial(F.relu),
        optimizer_partial=partial(optim.RMSprop),
        memory = ReplayMemory(10000)
    ):
        super().__init__(device=device)
        self._layout: List[LayerConf] = layout
        self._screen_dims = screen_dims
        self._outputs = outputs
        self._activation_func = activation_func
        self._conf = conf
        self._steps_done = 0
        self._optimizer_partial = optimizer_partial
        self._memory = memory
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
        self._network_stack = []
        # Setting the network layers
        for i, layer in enumerate(stack):
            layer_name = f"_{type(layer).__name__}_{i//2}"
            setattr(self, layer_name, layer)
            self._network_stack.append(layer_name)

        self._linear_input_size = conv_w * conv_h * self._layout[-1].batch_norm
        self._head = nn.Linear(self._linear_input_size, self._outputs)

    @staticmethod
    def conv2d_size_out(size, kernel_size=5, stride=2):
        return (size - (kernel_size - 1) - 1) // stride + 1

    def forward(self, x):
        x = x.to(self._device)
        for i in range(0, len(self._network_stack), 2):
            batch_norm = getattr(self, self._network_stack[i + 1])
            conv = getattr(self, self._network_stack[i])
            x = self._activation_func(batch_norm(conv(x)))

        return self._head(x.view(x.size(0), -1))

    # to be used in the policy network
    def select_action(self, state):
        sample = random.random()
        conf = self._conf
        eps_threshold = conf.EPS_END + (conf.EPS_START - conf.EPS_END) * \
            math.exp(-1. * self._steps_done / conf.EPS_DECAY)
        self._steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # pick action with the larger expected reward.
                return self(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self._outputs)]], device=self._device, dtype=torch.long)

    def optimize_model(self, target_net, criterion = nn.SmoothL1Loss()):
        conf = self._conf
        if len(self._memory) < conf.BATCH_SIZE:
            return
        transitions = self._memory.sample(conf.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self._device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(conf.BATCH_SIZE, device=self._device)
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * conf.GAMMA) + reward_batch

        # Compute Huber loss
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self._optimizer.zero_grad()
        loss.backward()
        for param in self.parameters():
            param.grad.data.clamp_(-1, 1)
        self._optimizer.step()


    @staticmethod
    def fit_networks(policy_net, target_net, env:Env, get_screen:callable, num_episodes=600) -> List: 
        episode_durations = []
        device = policy_net._device
        for i_episode in range(num_episodes):
            # Initialize the environment and state
            env.reset()
            last_screen = get_screen()
            current_screen = get_screen()
            state = current_screen - last_screen
            for t in count():
                # Select and perform an action
                action = policy_net.select_action(state)
                _, reward, done, _ = env.step(action.item())
                reward = torch.tensor([reward], device=device)

                # Observe new state
                last_screen = current_screen
                current_screen = get_screen()
                if not done:
                    next_state = current_screen - last_screen
                else:
                    next_state = None

                # Store the transition in memory
                policy_net._memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                policy_net.optimize_model(target_net)
                if done:
                    episode_durations.append(t + 1)
                    break
            # Update the target network, copying all weights and biases in DQN
            if i_episode % policy_net._conf.TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())
        return episode_durations