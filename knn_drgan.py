from sklearn.neighbors import KNeighborsClassifier
import cv2
import os
import numpy as np

result = np.zeros((11, 11))
angles_gallery = ['000', '018', '036', '054', '072',
                  '090',
                  '108', '126', '144', '162', '180']
angles_probe = angles_gallery
ix = 0
iy = 0
for g_ang in angles_gallery:
    iy = 0
    for p_ang in angles_probe:
        pid = 101
        X = []
        y = []
        for cond in ['nm-01', 'nm-02', 'nm-03', 'nm-04']:
            for p in range(pid, 125):
                path = '../fake/%03d-%s-%s-real.png' % (
                    p, cond, g_ang)
                path1 = '../fake/%03d-%s-%s-fake.png' % (p, cond, g_ang)
                if not os.path.exists(path):
                    continue
                if g_ang == '090':
                    img = cv2.imread(path, 0)
                else:
                    img = cv2.imread(path1, 0)
                img = cv2.resize(img, (64, 64))
                img = img.flatten().astype(np.float32)
                X.append(img)
                y.append(p-pid)

        nbrs = KNeighborsClassifier(n_neighbors=1, p=1, weights='distance')
        X = np.asarray(X)
        y = np.asarray(y).astype(np.int32)
        nbrs.fit(X, y)

        testX = []
        testy = []
        for cond in ['nm-05', 'nm-06']:
            for p in range(pid, 125):
                path = '../fake/%03d-%s-%s-real.png' % (
                    p, cond, g_ang)
                path1 = '../fake/%03d-%s-%s-fake.png' % (p, cond, g_ang)
                if not os.path.exists(path):
                    continue
                if p_ang == '090':
                    img = cv2.imread(path, 0)
                else:
                    img = cv2.imread(path1, 0)
                img = cv2.resize(img, (64, 64))
                img = img.flatten().astype(np.float32)
                testX.append(img)
                testy.append(p-pid)

        testX = np.asarray(testX).astype(np.float32)
        s = nbrs.score(testX, testy)
        s = round(s, 3)
        result[ix][iy] = s
        print(s)
        iy += 1
    ix += 1
print(result)
np.savetxt("view_analysis.csv", result)
