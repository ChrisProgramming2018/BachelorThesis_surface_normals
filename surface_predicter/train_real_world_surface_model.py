from taskonomy_network import TaskonomyEncoder, TaskonomyDecoder, TaskonomyNetwork,  TASKS_TO_CHANNELS
from models import VisualPrior, VisualPriorRepresentation
from PIL import Image
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import cv2
import numpy as np
import sys
import torch
import time
from collections import deque
import torch.optim as optim
import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

def time_format(sec):
    """
    Args:
    param1():
    """
    hours = sec // 3600
    rem = sec - hours * 3600
    mins = rem // 60
    secs = rem - mins * 60
    return hours, mins, round(secs,2)

t0 = time.time()

#vp = VisualPrior()
#feature_tasks= ["normal"]
#vr = VisualPriorRepresentation()
#vr._load_unloaded_nets(feature_tasks)
TASKONOMY_PRETRAINED_WEIGHT_FILES= ["normal_decoder-8f18bfb30ee733039f05ed4a65b4db6f7cc1f8a4b9adb4806838e2bf88e020ec.pth", "normal_encoder-f5e2c7737e4948e3b2a822f584892c342eaabbe66661576ba50db7cdd40561c5.pth"]
#path = "pretrained_model_weights"
path_de = 'https://github.com/alexsax/visual-prior/raw/networks/assets/pytorch/' + str(TASKONOMY_PRETRAINED_WEIGHT_FILES[0])
path_en = 'https://github.com/alexsax/visual-prior/raw/networks/assets/pytorch/' + str(TASKONOMY_PRETRAINED_WEIGHT_FILES[1])
model = TaskonomyNetwork(load_encoder_path=path_en, load_decoder_path=path_de)

model_path = "trained_realsim_normal_model"

if not os.path.exists(model_path):
    os.makedirs(model_path)
model.save_model(model_path + "/real_sim_normal_model-{}".format(0))
model.encoder.eval_only = False
model.decoder.eval_only = False

for param in model.parameters():
        param.requires_grad = True

model.cuda()
model.train()

lr = 3e-4
optimizer = optim.Adam(model.parameters(), lr=lr)

from replay_buffer_depth import ReplayBufferDepth

now = datetime.now()    
dt_string = now.strftime("%d_%m_%Y_%H:%M:%S")
pathname = "realsim_model_surface_normals"
pathname += dt_string
tensorboard_name = 'runs/' + pathname
writer = SummaryWriter(tensorboard_name)


size = 256
memory = ReplayBufferDepth((size, size), (size, size, 3), (size, size, 3), 15001, "cuda")
memory_valid = ReplayBufferDepth((size, size), (size, size, 3), (size, size, 3), 2001, "cuda")

path = "../data/sim_real_buffer-train"
print("Load buffer ...")
memory.load_memory_normals(path)
print("... buffer size {} loaded".format(memory.idx))
#memory.test_surface_normals(42)
#memory_valid.idx = 100
#memory.idx = 100
path = "../data/sim_real_buffer-valid"
print("Load valid buffer ...")
memory_valid.load_memory_normals(path)
print("... valid buffer {} loaded".format(memory_valid.idx))
#memory_valid.test_surface_normals(420)

print("buffer size train ", memory.idx)
print("buffer size valid ", memory_valid.idx)
torch.cuda.empty_cache()
batch_size = 32
scores_window = deque(maxlen=100) 
epochs = int(100e4)
for epoch in range(epochs):
    print('\rEpisode {}'.format(epoch), end="")
    rgb_batch, depth_batch, normal_batch = memory.sample(batch_size)
    x_recon = model(rgb_batch.cuda())

    # loss = -torch.mean(torch.sum(normal_batch * torch.log(1e-5 + x_recon) + (1 - normal_batch) * torch.log(1e-5 + 1 - x_recon), dim=1))
    optimizer.zero_grad()
    loss = F.mse_loss(x_recon, normal_batch.cuda())
    loss.backward()
    optimizer.step()
    scores_window.append(loss.item())
    mean_loss = np.mean(scores_window)
    writer.add_scalar('loss', loss.item(), epoch)  
    writer.add_scalar('mean_loss', mean_loss, epoch)  
    
    if epoch % 75 == 0:
        model.eval()
        eval_loss = 0
        evaL_size = 25
        for i in range(evaL_size):
            rgb_batch, depth_batch, normal_batch = memory_valid.sample(batch_size)
            x_recon = model(rgb_batch.cuda()).detach()
            loss = F.mse_loss(x_recon, normal_batch.cuda())
            eval_loss +=loss
        eval_loss = eval_loss / evaL_size
        model.save_model(model_path + "/model_step_{}_eval_loss_{:.10f}".format(epoch, eval_loss))
        model.train()
        text = "Eval model {} eval loss {:10f} time {}  \r".format(epoch, eval_loss, time_format(time.time() - t0))
        writer.add_scalar('eval_loss', eval_loss, epoch)  
        print("  ")
        print(text)

    if epoch % 5 == 0:
        text = "Epochs {}  loss {:.5f}  ave loss {:.5f}  time {}  \r".format(epoch, loss, mean_loss, time_format(time.time() - t0))
        print("  ")
        print(text)
