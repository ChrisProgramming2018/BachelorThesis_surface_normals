from taskonomy_network import TaskonomyEncoder, TaskonomyDecoder, TaskonomyNetwork,  TASKS_TO_CHANNELS
from models import VisualPrior, VisualPriorRepresentation
from PIL import Image
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import cv2
import numpy as np
import sys
import torch

#vp = VisualPrior()
#feature_tasks= ["normal"]
#vr = VisualPriorRepresentation()
#vr._load_unloaded_nets(feature_tasks)
TASKONOMY_PRETRAINED_WEIGHT_FILES= ["normal_decoder-8f18bfb30ee733039f05ed4a65b4db6f7cc1f8a4b9adb4806838e2bf88e020ec.pth", "normal_encoder-f5e2c7737e4948e3b2a822f584892c342eaabbe66661576ba50db7cdd40561c5.pth"]
#path = "pretrained_model_weights"
path_de = 'https://github.com/alexsax/visual-prior/raw/networks/assets/pytorch/' + str(TASKONOMY_PRETRAINED_WEIGHT_FILES[0])
path_en = 'https://github.com/alexsax/visual-prior/raw/networks/assets/pytorch/' + str(TASKONOMY_PRETRAINED_WEIGHT_FILES[1])
model = TaskonomyNetwork(load_encoder_path=path_en, load_decoder_path=path_de)

from replay_buffer_depth import ReplayBufferDepth
   
size = 256
memory = ReplayBufferDepth((size, size), (size, size, 3), (size, size, 3), 15000, "cuda")
path = "normals_memory5k"
memory.load_memory(path)


batch_size = 32


rgb_batch, depth_batch, normal_batch = memory.sample(batch_size)

#rgb_batch= TF.to_tensor(rgb_batch)
print("r", rgb_batch.shape)
y = model(rgb_batch)
loss = F.mse_loss(rgb_batch, y)
print(loss)
print(y.shape)
sys.exit()
"""
resized = []
for i in y:
    print(i.shape)
    obs = cv2.resize(np.array(i.transpose(2,0,1)), dsize=(84, 84), interpolation=cv2.INTER_CUBIC)
    print(obs.shape)
    sys.exit()
    resized.append(TF.to_tensor(obs))

y = torch.stack(resized, dim=0)

print(y.shape)
loss = F.mse_loss(rgb_batch, y)
print(loss)
print(y.shape)
sys.exit()
#rgb_batch= TF.to_tensor(rgb_batch)
# y = model(rgb_batch)
print(normal_batch.shape)
# x = x.transpose(1,2,0)
#red, green, blue = x.T 
#x = np.array([blue, green, red])
#print(x.shape)
#x = x.transpose()
"""

x = memory.obses[0]
print("from memory", x.shape)
#cv2.imshow("depth_image", x.transpose(1,2,0))
#cv2.imshow("depth_image", x[...,::-1])
#cv2.imshow("depth_image", n)
#cv2.imshow("depth_image", x)
#cv2.waitKey(0)

x = TF.to_tensor(x)
#image = Image.open('test.png')
# x = TF.to_tensor(TF.resize(image, 256)) * 2 - 1
#x = TF.to_tensor(TF.resize(image, 84)) #* 2 - 1
#x1 = np.array(image) #* 2 - 1
#cv2.imshow("depth_image", x1.transpose(1,2,0))
#cv2.imshow("depth_image", x1)
#cv2.waitKey(0)
print("before stack ", x.shape)
x = torch.stack([x,x], dim=0)
# x = x.unsqueeze_(0)
print("input ", x.shape)
#print(model)
y = model(x)
print(y.shape)
print(y)
print(y.min(), y.max())
TF.to_pil_image(y[0] / 2. + 0.5).save('test_normals_readout.png')
