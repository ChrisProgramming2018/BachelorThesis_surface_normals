import os
import cv2
import sys
import torch
import numpy as np
from gym.spaces import Box
from gym import Wrapper
from collections import deque
import visualpriors
from PIL import Image
import torchvision.transforms.functional as TF
from taskonomy_network import TaskonomyNetwork


class FrameStack(Wrapper):
    r"""Observation wrapper that stacks the observations in a rolling manner.
    For example, if the number of stacks is 4, then the returned observation contains
    the most recent 4 observations. For environment 'Pendulum-v0', the original observation
    is an array with shape [3], so if we stack 4 observations, the processed observation
    has shape [4, 3].
    .. note::
        To be memory efficient, the stacked observations are wrapped by :class:`LazyFrame`.
    .. note::
        The observation space must be `Box` type. If one uses `Dict`
        as observation space, it should apply `FlattenDictWrapper` at first.
    Example::
        >>> import gym
        >>> env = gym.make('PongNoFrameskip-v0')
        >>> env = FrameStack(env, 4)
        >>> env.observation_space
        Box(4, 210, 160, 3)
    Args:
        env (Env): environment object
        num_stack (int): number of stacks
        lz4_compress (bool): use lz4 to compress the frames internally
    """
    def __init__(self, env, config):
        super(FrameStack, self).__init__(env)
        self.state_buffer = deque([], maxlen=config["history_length"])
        self.env = env
        self.size = config["size"]
        self.device = config["device"]
        self.history_length = config["history_length"]
        self.model = TaskonomyNetwork()
        self.model.load_model("trained_models/model-21170.39453125")

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation = cv2.resize(observation,(256, 256))
        img = observation
        rgb_array= TF.to_tensor(observation)
        observation = self.model(rgb_array.unsqueeze(0))
        observation = np.array(observation.squeeze(0))
        observation = observation.transpose(1,2,0)
        observation = cv2.resize(observation,(84, 84))
        state = self._create_next_obs(observation)
        return state, reward, done, img

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        observation = cv2.resize(observation,(256, 256))
        rgb_array= TF.to_tensor(observation)
        observation = self.model(rgb_array.unsqueeze(0))
        observation = np.array(observation.squeeze(0))
        observation = observation.transpose(1,2,0)
        observation = cv2.resize(observation,(84, 84))
        state = self._stacked_frames(observation)
        return state
        
    def _create_next_obs(self, state):
        state =  cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        state = cv2.resize(state,(self.size, self.size))
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        self.state_buffer.append(state)
        state = torch.stack(list(self.state_buffer), 0)
        state = state.cpu()
        obs = np.array(state)
        return obs


    def _stacked_frames(self, state):
        # Transform to normals feature
        # representation = visualpriors.representation_transform(state, 'normal', device=self.device)
        # Transform to normals feature and then visualize the readout
        # pred = visualpriors.feature_readout(state, 'normal', device=self.device)
        # TF.to_pil_image(pred[0].cpu()/ 2. + 0.5).save('test_normals_readout.png')
        state =  cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        state = cv2.resize(state,(self.size, self.size))
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        zeros = torch.zeros_like(state)
        for idx in range(self.history_length - 1):
            self.state_buffer.append(zeros)
        self.state_buffer.append(state)

        state = torch.stack(list(self.state_buffer), 0)
        state = state.cpu()
        obs = np.array(state)
        return obs


def mkdir(base, name):
    """
    Creates a direction if its not exist
    Args:
       param1(string): base first part of pathname
       param2(string): name second part of pathname
    Return: pathname 
    """
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path





def write_into_file(pathname, text):
    """
    """
    with open(pathname+".txt", "a") as myfile:
        myfile.write(text)
        myfile.write('\n')

