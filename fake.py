# speed up the loading of the training data
import cv2
import torch as th
from DRGAN import NetG, NetD
from data_pre import CASIABDatasetGenerate
from keras.utils import to_categorical
import numpy as np

Nd = 24
Np = 11
Nz = 50
channel_num = 1
netg = NetG(Np=Np, Nz=Nz, channel_num=channel_num)
netd = NetD(Nd=Nd, Np=Np, channel_num=channel_num)

device = th.device("cuda:0")
model_G = th.load('../save/p_epoch_last_G1.pt')
netg = model_G
netg.eval()
netd.eval()
batchsize = 11
angles = ['000', '018', '036', '054', '072', '090',
          '108', '126', '144', '162', '180']
pcode = 7
pose = th.LongTensor([pcode, pcode, pcode, pcode, pcode, pcode, pcode, pcode, pcode, pcode, pcode]).to(device)
p = to_categorical(pose, 11)
p = th.FloatTensor(p).to('cuda:0')
z = th.FloatTensor(np.random.uniform(-1, 1, (batchsize, 50))).to(device)
for cond in ['nm-01', 'nm-02', 'nm-03', 'nm-04', 'nm-05',
             'nm-06']:
    dataset = CASIABDatasetGenerate(data_dir='../gei/',
                                    cond=cond)
    for i in range(101, 125):
        ass_label, img = dataset.getbatch(i)
        img = img.to(device).to(th.float32)
        with th.no_grad():
            fake_list = netg(img, p, z)
            fake_list = (fake_list + 1) / 2 * 255
            real_list = (img + 1) / 2 * 255
            view_list = (ass_label + 1) / 2 * 255
            view = view_list[0]
            view_ = view[0].squeeze().cpu().numpy()
            for j in range(11):
                fake = fake_list[j]
                fake_ = fake[0].squeeze().cpu().numpy()
                ang = angles[j]
                real = real_list[j]
                real_ = real[0].squeeze().cpu().numpy()
                cv2.imwrite('../fake/%03d-%s-%s-fake.png' % (i, cond, ang), fake_)
                cv2.imwrite('../fake/%03d-%s-%s-real.png' % (i, cond, ang), real_)

