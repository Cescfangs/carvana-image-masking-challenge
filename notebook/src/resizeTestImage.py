import cv2
import os
import argparse


parser = argparse.ArgumentParser(description='resize image to cut down memory')
parser.add_argument('input_dir', type=str, help='input_directory')
parser.add_argument('output_dir', type=str, help='output_directory')
parser.add_argument('-W', metavar='width', default=240, action='store',
                    dest='w', help='output width')
parser.add_argument('-H',metavar='height', default=160, action='store',
                    dest='h', help='output height')

args = parser.parse_args()

w, h = args.w, args.h
input_dir = args.input_dir
output_dir = args.output_dir

print('input dir: {}\noutput dir: {}\noutput size: {} x {}\n'.format(input_dir, output_dir, w, h))
assert os.path.exists(input_dir)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

imgs = os.listdir(input_dir)
imgs = [img for img in imgs if img.endswith('.jpg')]

i = 0
while imgs:
    if i % 1000 == 0:
        print('resize image {}'.format(i))
    i += 1
    img_name = imgs.pop()
    img = cv2.imread(input_dir + img_name)
    img = cv2.resize(img, (w, h))
    cv2.imwrite(output_dir + img_name, img)
