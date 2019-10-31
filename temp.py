import torch as th
import numpy as np
from keras.utils import to_categorical

'''x = np.array([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]], )
x = th.from_numpy(x)
x = x.float()
print(x)
pose = th.LongTensor(np.random.randint(10, size=8))
p = to_categorical(pose, 10)
p = th.FloatTensor(p)
print(p)'''
pose = th.LongTensor([5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]).to('cuda:0')
p = to_categorical(pose, 11)
p = th.FloatTensor(p).to('cuda:0')
print(pose)
print(p)



