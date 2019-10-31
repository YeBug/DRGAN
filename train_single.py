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
import joblib
import numpy as np
from keras.utils import to_categorical
Nd = 100
Np = 11
Nz = 50
channel_num = 1
save_dir = '../save/'

netg = NetG(Np=Np, Nz=Nz, channel_num=channel_num)
netd = NetD(Nd=Nd, Np=Np, channel_num=channel_num)

device = th.device("cuda:0")
# weights init
all_mods = itertools.chain()
all_mods = itertools.chain(all_mods, [
    list(netg.children())[0].children(),
    list(netd.children())[0].children(),
])
for mod in all_mods:
    if isinstance(mod, nn.Conv2d) or isinstance(mod, nn.ConvTranspose2d):
        init.normal_(mod.weight, 0.0, 0.02)
    elif isinstance(mod, nn.BatchNorm2d):
        init.normal_(mod.weight, 1.0, 0.02)
        init.constant_(mod.bias, 0.0)

netg = netg.to(device)
netd = netd.to(device)
netg.train()
netd.train()
netg = th.load('../save/p_epoch_last_G1.pt')
netd = th.load('../save/p_epoch_last_D1.pt')
dataset = CASIABDataset(data_dir='../gei/')
save_freq = 20000
iteration = 0
lr = 0.0002
batchsize = 64

pose = th.LongTensor(np.random.randint(Np, size=batchsize)).to(device)
p = to_categorical(pose, Np)
p = th.FloatTensor(p).to(device)
'''pose = th.LongTensor([5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]).to(device)
p = to_categorical(pose, 11)
p = th.FloatTensor(p).to(device)'''
z = th.FloatTensor(np.random.uniform(-1, 1, (batchsize, Nz))).to(device)
zeros = th.zeros(batchsize).to(device)
ones = th.ones(batchsize).to(device)
optimG = optim.Adam(netg.parameters(), lr=lr)
optimD = optim.Adam(netd.parameters(), lr=lr)
BECLoss = nn.BCEWithLogitsLoss()
print('Training starts')
Dloss_list = []
Gloss_list = []
D_tag_list = []
D_gan_list = []
D_pose_list = []
G_tag_list = []
G_gan_list = []
G_pose_list = []
neck = 8
op1 = 0
op2 = 0
for iteration in tqdm(range(60000)):
    view, img, label1, angles = dataset.getbatch(batchsize)
    img = img.to(device).to(th.float32)
    label1 = th.from_numpy(label1)
    label1 = label1.to(device).long()
    angles = th.from_numpy(angles)
    angles = angles.to(device).long()
    output_real = netd(img)
    fake = netg(img, p, z)   # generate fake
    output_fake = netd(fake)
    # update D
    if iteration % neck == 0:
        optimD.zero_grad()
        lossD_tag = F.cross_entropy(output_real[:, :Nd], label1)    # id label criterion
        lossD_gan = BECLoss(output_real[:, Nd], ones)+BECLoss(output_fake[:, Nd], zeros)    # GAN generate criterion
        lossD_pose = F.cross_entropy(output_real[:, Nd+1:], angles)   # pose label criterion
        lossD = lossD_gan+lossD_tag+lossD_pose
        Dloss_list.append(lossD.item())
        D_tag_list.append(lossD_tag.item())
        D_gan_list.append(lossD_gan.item())
        D_pose_list.append(lossD_pose.item())
        lossD.backward()
        print('the D loss is:', lossD, '  neck = ', neck)
        op1 = lossD.item()
        optimD.step()
    else:
        # update G
        optimG.zero_grad()
        lossG_tag = F.cross_entropy(output_fake[:, :Nd], label1)    # fake id
        lossG_gan = BECLoss(output_fake[:, Nd], ones)   # fake to true
        lossG_pose = F.cross_entropy(output_fake[:, Nd+1:], pose)
        lossG = lossG_tag+lossG_gan+lossG_pose
        Gloss_list.append(lossG.item())
        G_tag_list.append(lossG_tag.item())
        G_gan_list.append(lossG_gan.item())
        G_pose_list.append(lossG_pose.item())
        lossG.backward()
        print('the G loss is:', lossG, '  neck = ', neck)
        op2 = lossG.item()
        optimG.step()
    '''if abs(op1-op2) < 0.5:
        neck = 2
    elif abs(op1-op2) > 5:
        neck = 5
    else:x
        neck = 3'''
    if iteration % save_freq == 0:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        else:
            save_path_D = os.path.join(save_dir, 'p_epoch{}_D.pt'.format(iteration))
            th.save(netd, save_path_D)
            save_path_G = os.path.join(save_dir, 'p_epoch{}_G.pt'.format(iteration))
            th.save(netg, save_path_G)

save_path_D = os.path.join(save_dir, 'p_epoch_last_D2.pt')
th.save(netd, save_path_D)
save_path_G = os.path.join(save_dir, 'p_epoch_last_G2.pt')
th.save(netg, save_path_G)
joblib.dump(Dloss_list, '../save/p_Dloss.pkl')
joblib.dump(Gloss_list, '../save/p_Gloss.pkl')
joblib.dump(G_tag_list, '../save/p_G_tag_list.pkl')
joblib.dump(G_gan_list, '../save/p_G_gan_list.pkl')
joblib.dump(G_pose_list, '../save/p_G_pose_list.pkl')
joblib.dump(D_tag_list, '../save/p_D_tag_list.pkl')
joblib.dump(D_gan_list, '../save/p_D_gan_list.pkl')
joblib.dump(D_pose_list, '../save/p_D_pose_list.pkl')
