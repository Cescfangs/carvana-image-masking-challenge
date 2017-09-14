import keras.backend as K
import pandas as pd
import time
import numpy as np
from keras.losses import binary_crossentropy
from functools import wraps


def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        print(' ..... Executing {} .....'.format(function.__name__))
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print('  ...Total time running %s: --> %s seconds <--' %
              (function.__name__, str(t1 - t0)))
        print(' ..... func {} returned ..... '.format(function.__name__))
        return result
    return function_timer


def dice_coef(y_true, y_pred, smooth=0.1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    bytes = np.where(img.flatten() == 1)[0]
    runs = []
    prev = -2
    for b in bytes:
        if (b > prev + 1):
            runs.extend((b + 1, 0))
        runs[-1] += 1
        prev = b

    return ' '.join([str(i) for i in runs])



def bce_dice_loss(y_true, y_pred):
    return 0.5 * binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)


def make_submission(pred, name, compression='gzip'):
    OUTPUT_PATH = '../../data/results/'
    print('...creating submission...')
    img_name, masks = pred
    # print(img_name, masks)
    # res = [rle(mask) for mask in masks]
    # print(res)
    df = pd.DataFrame({'img': img_name, 'rle_mask': masks})
    df.to_csv(OUTPUT_PATH + name + '.gz', index=None, compression=compression)
