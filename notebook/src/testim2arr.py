from scipy.misc import imread
import os
import numpy as np

dir = '../../data/test_small'
imgs = os.listdir(dir)
imgs = [img for img in imgs if img.endswith('.jpg')]
np.save(os.path.join(dir, 'img_ids'), imgs)
imgs = np.array([imread(os.path.join(dir, img_name)) for img_name in imgs]) / 255
np.save(os.path.join(dir, 'img.npy'), imgs)