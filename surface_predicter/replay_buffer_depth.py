import os
import cv2
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import numpy as np
from create_surface_normals import create_normal
import torchvision.transforms.functional as TF


class ReplayBufferDepth(object):
    """Buffer to store environment transitions."""
    def __init__(self, depth_shape, obs_shape, normal_shape, capacity, device):
        self.capacity = capacity
        self.device = device
        self.depth = np.empty((capacity, *depth_shape), dtype=np.float32)
        self.normals = np.empty((capacity, *normal_shape), dtype=np.float32)
        self.obses = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.idx = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, depth, obs):
        np.copyto(self.depth[self.idx], depth)
        np.copyto(self.obses[self.idx], obs.astype(np.uint8))
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.capacity if self.full else self.idx, size=batch_size)    
        obses = self.obses[idxs]
        depth = self.obses[idxs]
        normals = self.normals[idxs]
        #obses = torch.as_tensor(obs, device=self.device).float()
        depth = torch.as_tensor(depth, device=self.device).float()
        # normals = torch.as_tensor(normals, device=self.device).float()
        
        normal_t_list = []
        for n in normals:
            normal_t_list.append(TF.to_tensor(n))
        normals = torch.stack(normal_t_list, dim=0)
        
        obs_t_list = []
        for obs in obses:
            obs_t_list.append(TF.to_tensor(obs))
        obses = torch.stack(obs_t_list, dim=0)
        return obses, depth, normals

    def create_surface_normals(self, filename):
        for index in range(self.idx):
            print("create normals {} of {}".format(index, self.idx))
            normal_array = create_normal(self.depth[index]) * 255
            #cv2.imshow("surface_image_255", normal_array)
            #cv2.waitKey(0)
            # print(normal_array.shape)
            np.copyto(self.normals[index], normal_array)
            #cv2.imshow("surface_image_255", self.normals[index])
            #cv2.waitKey(0)
        print("save normals ...")
        self.save_memory_normals(filename)
    
    def test_surface_normals(self, index):
        array_RGB = self.obses[index]
        print("RGB buffer ", array_RGB.shape)
        # array_RGB = array_RGB.transpose(1,2,0) 
        frame = cv2.imwrite("array_RGB{}.png".format(index), np.array(array_RGB))
        print("shape of RGB ", array_RGB.shape)
        # array_RGB = new_obs.transpose(1,2,0) 
        array_depth = self.depth[index]
        array_normals = self.normals[index]
        array_surface_normals = create_normal(array_depth)
        frame = cv2.imwrite("surface_image_compute{}.png".format(index), np.array(array_surface_normals * 255))
        # frame = cv2.imwrite("surface_image_buffer{}.png".format(index), np.array(array_normals * 255))
        cv2.imshow("depth_image", array_depth)
        cv2.waitKey(0)
        print("shape of RGB ", array_RGB.shape)
        cv2.imshow("RGB_image", array_RGB)
        cv2.waitKey(0)
        cv2.imshow("RGB_image", array_RGB[:,:,::-1])
        cv2.waitKey(0)
        # array_normals = np.swapaxes(array_normals, 2,-3)
        print("shape of normals buffer ", array_normals.shape)
        print("shape of normals buffer ", array_normals.max())
        cv2.imshow("normals buffer", array_normals)
        cv2.waitKey(0)
        cv2.imshow("surface_image_255", array_surface_normals * 255)
        cv2.waitKey(0)


    def save_memory_normals(self, filename):
        """
        Use numpy save function to store the data in a given file
        """
        if not os.path.exists(filename):
            os.makedirs(filename)
        with open(filename + '/depth_obses.npy', 'wb') as f:
            np.save(f, self.depth)
        
        with open(filename + '/obses.npy', 'wb') as f:
            np.save(f, self.obses)
        
        with open(filename + '/normals.npy', 'wb') as f:
            np.save(f, self.normals)
        
        with open(filename + '/index.txt', 'w') as f:
            f.write("{}".format(self.idx))
        
        print("save buffer to {}".format(filename))
    
    def load_memory_normals(self, filename):
        """
        Use numpy load function to store the data in a given file
        """
        
        with open(filename + '/obses.npy', 'rb') as f:
            self.obses = np.load(f)

        with open(filename + '/depth_obses.npy', 'rb') as f:
            self.depth = np.load(f)
        
        with open(filename + '/normals.npy', 'rb') as f:
            self.normals = np.load(f)

        with open(filename + '/index.txt', 'r') as f:
            self.idx = int(f.read())
    
    def save_memory(self, filename):
        """
        Use numpy save function to store the data in a given file
        """
        if not os.path.exists(filename):
            os.makedirs(filename)
        with open(filename + '/depth_obses.npy', 'wb') as f:
            np.save(f, self.depth)
        
        with open(filename + '/obses.npy', 'wb') as f:
            np.save(f, self.obses)
        
        
        with open(filename + '/index.txt', 'w') as f:
            f.write("{}".format(self.idx))
        
        print("save buffer to {}".format(filename))
    
    def load_memory(self, filename):
        """
        Use numpy load function to store the data in a given file
        """
        
        with open(filename + '/obses.npy', 'rb') as f:
            self.obses = np.load(f)

        with open(filename + '/depth_obses.npy', 'rb') as f:
            self.depth = np.load(f)
        
        with open(filename + '/index.txt', 'r') as f:
            self.idx = int(f.read())
