import torch
import torch as th
import itertools
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from DRGAN import NetD, NetG
from data_pre import CASIABDataset
import torch.optim as optim
import os
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from keras.utils import to_categorical
Nd = 100
Np = 10
Nz = 50
channel_num = 1
save_dir = '../save/'

device = th.device("cuda:0")

dataset = CASIABDataset(data_dir='../gei/')
save_freq = 20000
iteration = 0
lr = 0.0002
batchsize = 8
pose = th.LongTensor([5, 5, 5, 5, 5, 5, 5, 5]).to('cuda:0')
p = to_categorical(pose, 11)
p = th.FloatTensor(p).to('cuda:0')
z = th.zeros((batchsize, 50), requires_grad=False).to(device)
model_G = torch.load('../save/p_epoch_last_G_1.pt')
netg = model_G
for iteration in tqdm(range(5)):
    ass_label, img, label1, angles = dataset.getbatch(batchsize)
    ass_label = ass_label.to(device).to(th.float32)
    img = img.to(device).to(th.float32)
    label1 = th.from_numpy(label1)
    print(label1)
    print(angles)
    label1 = label1.to(device).long()
    fake = netg(img, p, z)   # generate fake
    for i in range(len(fake)):
        refake = fake[i]
        refake1 = refake[0]
        img1 = refake1.detach().cpu().numpy()*255
        cv2.imshow("img", np.uint8(img1))
        cv2.waitKey()
        real = img[i]
        img2 = real[0].cpu().numpy()*255
        cv2.imshow('img2', np.uint8(img2))
        cv2.waitKey()




