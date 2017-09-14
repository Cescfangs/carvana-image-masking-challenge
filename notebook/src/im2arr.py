import cv2
import os
import argparse
import numpy as np
from scipy.misc import imread

parser = argparse.ArgumentParser(description='resize image to cut down memory')
parser.add_argument('input_dir', type=str, help='input_directory')
parser.add_argument('output_dir', type=str, help='output_directory')
parser.add_argument('-W', metavar='width', default=240, action='store',
                    dest='w', help='output width')
parser.add_argument('-H', metavar='height', default=160, action='store',
                    dest='h', help='output height')
parser.add_argument('--mask', action='store_true', default=False, help='weather images are mask', dest='mask')

args = parser.parse_args()

is_mask = args.mask
w, h = args.w, args.h
input_dir = args.input_dir
output_dir = args.output_dir

print('is mask: {}\ninput dir: {}\noutput dir: {}\noutput size: {} x {}\n'.format(is_mask, input_dir, output_dir, w, h))
assert os.path.exists(input_dir)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

imgs = os.listdir(input_dir)
if is_mask:
    imgs = list(np.load(os.path.join(input_dir, 'img_ids.npy')))
    imgs = [img_id[:-4] + '_mask.gif' for img_id in imgs]
else:
    imgs = [img for img in imgs if img[-3:] == 'jpg']
    np.save(os.path.join(output_dir, 'img_ids.npy'), imgs)

i = 0
img_array = []
while imgs:
    if i % 1000 == 0:
        print('resize image {}'.format(i))
    i += 1
    img_name = imgs.pop()
    img_dir = os.path.join(input_dir, img_name)
    img = imread(img_dir)
    img = cv2.resize(img, (w, h)) / 255
    img_array.append(img)
img_array = np.array(img_array)
if is_mask:
    img_array = img_array[...,0]
np.save(os.path.join(output_dir, 'img.npy'), img_array)
