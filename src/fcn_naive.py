import pandas as pd
import numpy as np
from skimage.io import imread
from skimage.transform import downscale_local_mean
from os.path import join
# import cv2
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Input, concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
import keras.backend as K

input_folder = join('..', 'data')

df_mask = pd.read_csv(join(input_folder, 'train_masks.csv'), usecols=['img'])
ids_train = df_mask['img'].map(lambda s: s.split('_')[0]).unique()

imgs_idx = list(range(1, 17))


def load_img(im, idx):
    return imread(join(input_folder, 'train', '{}_{:02d}.jpg'.format(im, idx)))


def load_mask(im, idx):
    return imread(join(input_folder, 'train_masks', '{}_{:02d}_mask.gif'.format(im, idx)))


def resize(im):
    return downscale_local_mean(im, (4, 4) if im.ndim == 2 else (4, 4, 1))


def mask_image(im, mask):
    return (im * np.expand_dims(mask, 2))


def dice_coef(y_true, y_pred, smooth=0.1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def bce_dice_loss(y_true, y_pred):
    return 0.5 * binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)


def dice_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def create_model(input_shape):
    # Create simple model
    inp = Input(input_shape)
    conv1 = Conv2D(filters=3, kernel_size=3, activation='relu', padding='same')(inp)
    max1 = MaxPooling2D(2)(conv1)
    conv2 = Conv2D(3, 3, activation='relu', padding='same')(max1)
    max2 = MaxPooling2D(2)(conv2)
    conv3 = Conv2D(3, 3, activation='relu', padding='same')(max2)

    deconv3 = Conv2DTranspose(3, 3, strides=4, activation='relu', padding='same')(conv3)
    deconv2 = Conv2DTranspose(3, 3, strides=2, activation='relu', padding='same')(conv2)

    deconvs = concatenate([conv1, deconv2, deconv3])

    out = Conv2D(1, 7, activation='sigmoid', padding='same')(deconvs)

    model = Model(inp, out)
    model.summary()
    return model




# From here: https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py



num_train = 2  # len(ids_train)

# Load data for position id=1
# X = np.empty((num_train * 16, 320, 480, 3), dtype=np.float32)
# y = np.empty((num_train * 16, 320, 480, 1), dtype=np.float32)
X = []
y = []
for i, img_id in enumerate(ids_train[:num_train]):
    x_batch = np.array([load_img(img_id, j) for j in imgs_idx])
    y_batch = np.array([np.expand_dims(load_mask(img_id, j), 2) / 255.0 for j in imgs_idx])
    X.append(x_batch)
    y.append(y_batch)
    del x_batch, y_batch
    if i % 2 == 0:
        print('batch {}'.format(i))
X = np.concatenate(X, axis=0)
y = np.concatenate(y, axis=0)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)
print(X_train.shape)
model = create_model(X_train.shape[1:])
model.compile(Adam(lr=1e-3), dice_loss, metrics=['accuracy', dice_coef])
history = model.fit(X_train, y_train, epochs=1, validation_data=(X_val, y_val), batch_size=8, verbose=1)

X_test_name = pd.read_csv('../data/sample_submission.csv').img[:10]
X_test = np.array([load_img(name[:-3], name[-2:]) for name in X_test_name])
pred = model.predict(X_test, batch_size=4)
