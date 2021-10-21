import math
import random
from collections import deque, namedtuple
from dataclasses import dataclass
from functools import partial
from itertools import count
from typing import Callable, List, NamedTuple, Tuple

import torch
import torch.nn.functional as F
import torch.optim as optim
from gym.core import Env
from torch import nn
import copy

from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms


from wompth.models.base import BaseNetwork, NeuralNetwork

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

ScreenDims = namedtuple("ScreenDims", ("height", "width"))


@dataclass
class DQNConf:
    BATCH_SIZE: float = 128
    GAMMA: float = 0.999
    EPS_START: float = .9
    MAX_EPISODES: int = 1000
    TARGET_UPDATE: int = 10


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
        conf: DQNConf = DQNConf(),
        optimizer_partial=partial(optim.Adam),
        memory=ReplayMemory(10000),
        outputs=2,
    ):
        super().__init__(device=device)
        self._conf = conf
        self._episodes_done = 0
        self._outputs = outputs
        self._epsilon  = conf.EPS_START
        self._optimizer_partial = optimizer_partial
        self._memory = memory


    def epsilon(self, A=0.3, B=0.1, C=0.1):
        # reference: https://medium.com/analytics-vidhya/stretched-exponential-decay-function-for-epsilon-greedy-algorithm-98da6224c22f
        conf = self._conf 

        standardized_time=(self._episodes_done-A*conf.MAX_EPISODES)/(B*conf.MAX_EPISODES)
        standardized_time = torch.tensor(math.exp(-standardized_time), dtype=torch.double, device=self._device)
        cosh=torch.cosh(standardized_time)
        epsilon=1.1-(1/cosh+(self._episodes_done*C/conf.MAX_EPISODES))
        return epsilon * conf.EPS_START


    def select_action(self, state):
    
        sample = random.random()
        self._epsilon = self.epsilon()

        if sample > self._epsilon:
            with torch.no_grad():
                # pick action with the larger expected reward.
                return self(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor(
                [[random.randrange(self._outputs)]],
                device=self._device,
                dtype=torch.long,
            )

        
    def optimize_model(self, target_net, criterion=nn.MSELoss()):
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
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self._device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )
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
        next_state_values[non_final_mask] = (
            target_net(non_final_next_states).max(1)[0].detach()
        )
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * conf.GAMMA) + reward_batch

        # Compute Huber loss
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self._optimizer.zero_grad()
        loss.backward()

        self._optimizer.step()

    def load_states_from(self, source_net):
        super().load_states_from(source_net)
        self._memory = copy.deepcopy(source_net._memory)
        # self._steps_done = source_net._steps_done

def fit_networks(
    policy_net:DQN, target_net:DQN, env: Env, get_screen: Callable, num_episodes=600, episode_durations = [], reward_function=None
) -> List:
    def moving_average_pth(x, w=10):
        kernel = [1/w] * w
        ts_tensor = torch.Tensor(x).reshape(1, 1, -1)
        kernel_tensor = torch.Tensor(kernel).reshape(1, 1, -1)
        return F.conv1d(ts_tensor, kernel_tensor).reshape(-1)


    policy_net._max_episodes = num_episodes
    if not target_net:
        target_net = policy_net

    conf = policy_net._conf

    writer = SummaryWriter(comment=f'{"single" if not target_net else "double"}-{str(conf)}' )

    device = policy_net._device
    initial_gamma = conf.GAMMA

    for i_episode in range(num_episodes):
        # Initialize the environment and state
        env.reset()
        last_screen = get_screen()
        current_screen = get_screen()
        state = current_screen - last_screen
        policy_net._episodes_done = i_episode
        reward_sum = 0
        for t in count():
            # Select and perform an action
            action = policy_net.select_action(state)
            _, reward, done, _ = env.step(action.item())

            if reward_function:   
                reward = reward_function(done, t)
            
            reward_sum += reward

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
        
        writer.add_scalar('Epsilon/train', policy_net._epsilon , i_episode)
        writer.add_scalar('Duration/train', t , i_episode)
        writer.add_scalar('Reward/train', reward_sum , i_episode)
        writer.add_scalar('Gamma/train', conf.GAMMA , i_episode)

        # conf.GAMMA = initial_gamma * (1
        # 
        # 
        # .0 - ((float(i_episode // conf.TARGET_UPDATE) * conf.TARGET_UPDATE) / float(conf.MAX_EPISODES)))
        
        # Update the target network, copying all weights and biases in DQN
        if i_episode >= conf.TARGET_UPDATE:
            mv_avg = moving_average_pth(episode_durations, conf.TARGET_UPDATE)
            last_mv_avg = float(mv_avg[-1])
            max_mv_avg = float(mv_avg.max())
            update_target =  (i_episode % conf.TARGET_UPDATE) == 0 #last_mv_avg >= max_mv_avg
            writer.add_scalar('MovingAvg/train', last_mv_avg, i_episode)
            writer.add_scalar('MaxMovingAvg/train', max_mv_avg, i_episode)


            if last_mv_avg == max_mv_avg: 
                policy_net.save(f'model_{i_episode}_avg_{last_mv_avg}.pth')


            if update_target:
                target_net.load_states_from(policy_net)

            
            if last_mv_avg > 200: 
                writer.close()
                return episode_durations

    writer.close()

    return episode_durations
