# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import sys 
sys.path.append('./')
env = gym.make('CartPole-v0').unwrapped

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()
# get_ipython().run_line_magic('matplotlib', 'inline')


# %%
from wompth.models.dqn import Transition, ReplayMemory, DQN, ScreenDims,LayerConf


# %%
resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])


def get_cart_location(screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

def get_screen():
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    # Cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(screen_width)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0)


env.reset()
plt.figure()
plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),
           interpolation='none')
plt.title('Example extracted screen')
plt.show()


# %%

# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym. Typical dimensions at this point are close to 3x40x90
# which is the result of a clamped and down-scaled render buffer in get_screen()
init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape

# Get number of actions from gym action space
n_actions = env.action_space.n
screen_dims = ScreenDims(screen_height, screen_width)
network_layout = [
    LayerConf(input=3, kernel_size=5, stride=2, batch_norm=16),
    LayerConf(input=16, kernel_size=5, stride=2, batch_norm=16),
    LayerConf(input=32, kernel_size=5, stride=2, batch_norm=32),
]


# %%
policy_net = DQN(layout=network_layout, screen_dims=screen_dims, outputs=n_actions)
target_net = DQN(layout=network_layout, screen_dims=screen_dims, outputs=n_actions)


# %%
policy_net._linear_input_size


# %%
policy_net


# %%

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()


def moving_average_pth(x, w=10):
    kernel = [1/w] * w
    ts_tensor = torch.Tensor(x).reshape(1, 1, -1)
    kernel_tensor = torch.Tensor(kernel).reshape(1, 1, -1)
    return F.conv1d(ts_tensor, kernel_tensor).reshape(-1)

def plot_durations(i_episode, episode_durations):
    display.clear_output(wait=True)

    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel(f'Episode {i_episode}')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    means = moving_average_pth(durations_t)
    plt.plot(means.numpy())
    display.display(plt.gcf())


# %%
num_episodes = 2000
episodes_print = 10
episode_durations = []
for i in range(0, num_episodes, episodes_print):
    print (len(episode_durations))
    durations = DQN.fit_networks(policy_net,target_net, env, get_screen, num_episodes=episodes_print)
    episode_durations += durations
    plot_durations(i, episode_durations)
print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()


# %%



