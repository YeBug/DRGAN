import glob as gb
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split, ShuffleSplit
import tensorflow as tf
from keras.utils import to_categorical
import joblib
import matplotlib.pyplot as plt
import math

Dloss = joblib.load('../save/90_Dloss.pkl')
Dtag = joblib.load('../save/90_D_tag_list.pkl')
Dgan = joblib.load('../save/90_D_gan_list.pkl')
Dpose = joblib.load('../save/90_D_pose_list.pkl')
Gloss = joblib.load('../save/90_Gloss.pkl')
Gtag = joblib.load('../save/90_G_tag_list.pkl')
Ggan = joblib.load('../save/90_G_gan_list.pkl')
Gpose = joblib.load('../save/90_G_pose_list.pkl')
D_epoch = 33334
G_epoch = 66666


fig = plt.figure(figsize=(60, 120))
ax1 = fig.add_subplot(421)
x = range(D_epoch)
p1, = ax1.plot(x, Dloss, "-.", markersize=2, label="D Cross Entropy")
handles, labels = ax1.get_legend_handles_labels()
ax1.set_ylabel('DLoss')
ax1.set_xlabel('Epoch')
ax1.set_title(
    "|Last D Loss:%.3f" % Dloss[-1])
ax1.legend(handles[::-1], labels[::-1])
ax1 = fig.add_subplot(422)
x = range(D_epoch)
p1, = ax1.plot(x, Dtag, "-.", markersize=2, label="D TAG Entropy")
handles, labels = ax1.get_legend_handles_labels()
ax1.set_ylabel('Dtag')
ax1.set_xlabel('Epoch')
ax1.set_title(
    "|Last Dtag Loss:%.3f" % Dtag[-1])
ax1.legend(handles[::-1], labels[::-1])
ax1 = fig.add_subplot(423)
x = range(D_epoch)
p1, = ax1.plot(x, Dgan, "-.", markersize=2, label="D GAN Entropy")
handles, labels = ax1.get_legend_handles_labels()
ax1.set_ylabel('Dgan')
ax1.set_xlabel('Epoch')
ax1.set_title(
    "|Last Dgan Loss:%.3f" % Dgan[-1])
ax1.legend(handles[::-1], labels[::-1])
ax1 = fig.add_subplot(424)
x = range(D_epoch)
p1, = ax1.plot(x, Dpose, "-.", markersize=2, label="D POSE Entropy")
handles, labels = ax1.get_legend_handles_labels()
ax1.set_ylabel('Dpose')
ax1.set_xlabel('Epoch')
ax1.set_title(
    "|Last Dpose Loss:%.3f" % Dpose[-1])
ax1.legend(handles[::-1], labels[::-1])

x = range(G_epoch)
ax2 = fig.add_subplot(425)
p3, = ax2.plot(x, Gloss, "-.", markersize=2, label="G Cross Entropy")

handles1, labels1 = ax2.get_legend_handles_labels()
ax2.set_ylabel('GLoss')
ax2.set_xlabel('Epoch')
ax2.set_title(
    "|Last G Loss:" + str(Gloss[-1]) )
ax2.legend(handles1[::-1], labels1[::-1])
ax2 = fig.add_subplot(426)
p3, = ax2.plot(x, Gtag, "-.", markersize=2, label="G TAG Entropy")

handles1, labels1 = ax2.get_legend_handles_labels()
ax2.set_ylabel('GLoss')
ax2.set_xlabel('Epoch')
ax2.set_title(
    "|Last Gtag Loss:" + str(Gtag[-1]) )
ax2.legend(handles1[::-1], labels1[::-1])
ax2 = fig.add_subplot(427)
p3, = ax2.plot(x, Ggan, "-.", markersize=2, label="G GAN Entropy")

handles1, labels1 = ax2.get_legend_handles_labels()
ax2.set_ylabel('GLoss')
ax2.set_xlabel('Epoch')
ax2.set_title(
    "|Last Ggan Loss:" + str(Ggan[-1]) )
ax2.legend(handles1[::-1], labels1[::-1])
ax2 = fig.add_subplot(428)
p3, = ax2.plot(x, Gpose, "-.", markersize=2, label="G POSE Entropy")

handles1, labels1 = ax2.get_legend_handles_labels()
ax2.set_ylabel('GLoss')
ax2.set_xlabel('Epoch')
ax2.set_title(
    "|Last Gpose Loss:" + str(Gpose[-1]) )
ax2.legend(handles1[::-1], labels1[::-1])
plt.savefig("../save/Loss7.png")
plt.show()

