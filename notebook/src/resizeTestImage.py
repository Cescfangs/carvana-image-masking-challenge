import cv2
import os

w, h = 240, 160
input_dir = '../../data/test/'
output_dir = '../../data/test_small/'

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

imgs = os.listdir(input_dir)
imgs = [img for img in imgs if img.endswith('.jpg')]

i = 0
while imgs:
    if i % 100 == 0:
        print('resize image {}'.format(i))
    img_name = imgs.pop()
    img = cv2.imread(input_dir + img_name)
    img = cv2.resize(img, (w, h))
    cv2.imwrite(output_dir + img_name, img)
