import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Input, MaxPool2D, UpSampling2D, Concatenate, Conv2DTranspose
from keras.optimizers import Adam
from scipy.misc import imresize
from cv2 import resize
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import array_to_img, img_to_array, load_img, ImageDataGenerator
from naive import make_submission


data_dir = "../../data/train/"
mask_dir = "../../data/train_masks/"
all_images = os.listdir(data_dir)
output_width, output_height = 1918, 1280
input_width, input_height = 240, 160
# pick which images we will use for testing and which for validation
train_images, validation_images = train_test_split(all_images, train_size=0.8, test_size=0.2)

# utility function to convert greyscale images to rgb
# Now let's use Tensorflow to write our own dice_coeficcient metric


def dice_coef(y_true, y_pred):
    smooth = 1e-5

    y_true = tf.round(tf.reshape(y_true, [-1]))
    y_pred = tf.round(tf.reshape(y_pred, [-1]))

    isct = tf.reduce_sum(y_true * y_pred)

    return (2 * isct + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)


def grey2rgb(img):
    new_img = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            new_img.append(list(img[i][j]) * 3)
    new_img = np.array(new_img).reshape(img.shape[0], img.shape[1], 3)
    return new_img


# generator that we will use to read the data from the directory
def data_gen_small(data_dir, mask_dir, images, batch_size, dims):
    """
    data_dir: where the actual images are kept
    mask_dir: where the actual masks are kept
    images: the filenames of the images we want to generate batches from
    batch_size: self explanatory
    dims: the dimensions in which we want to rescale our images
    """
    while True:
        ix = np.random.choice(np.arange(len(images)), batch_size)
        imgs = []
        labels = []
        for i in ix:
            # images
            original_img = load_img(data_dir + images[i])
            resized_img = imresize(original_img, dims + [3])
            array_img = img_to_array(resized_img) / 255
            imgs.append(array_img)

            # masks
            original_mask = load_img(mask_dir + images[i].split(".")[0] + '_mask.gif')
            resized_mask = imresize(original_mask, dims + [3])
            array_mask = img_to_array(resized_mask) / 255
            labels.append(array_mask[:, :, 0])
        imgs = np.array(imgs)
        labels = np.array(labels)
        yield imgs, labels.reshape(-1, dims[0], dims[1], 1)


def test_generator(data_dir, batch_size, dims):
    all_images = os.listdir(data_dir)
    while 1:
        img_ids = all_images[:batch_size]
        all_images = all_images[batch_size:]
        imgs = [load_img(data_dir + img_id) for img_id in img_ids]
        imgs = np.array([imresize(img, dims + [3]) for img in imgs]) / 255
        yield imgs, img_ids

    # datagen = ImageDataGenerator(rescale=1./255)

    # generator = datagen.flow_from_directory(
    #     data_dir,
    #     target_size=(128, 128),
    #     batch_size=batch_size,
    #     class_mode=None,
    #     shuffle=False)
    # return generator


def down(input_layer, filters, pool=True):
    conv1 = Conv2D(filters, (3, 3), padding='same', activation='relu')(input_layer)
    residual = Conv2D(filters, (3, 3), padding='same', activation='relu')(conv1)
    if pool:
        max_pool = MaxPool2D()(residual)
        return max_pool, residual
    else:
        return residual


def up(input_layer, residual, filters):
    filters = int(filters)
    upsample = UpSampling2D()(input_layer)
    upconv = Conv2D(filters, kernel_size=(2, 2), padding="same")(upsample)
    concat = Concatenate(axis=3)([residual, upconv])
    conv1 = Conv2D(filters, (3, 3), padding='same', activation='relu')(concat)
    conv2 = Conv2D(filters, (3, 3), padding='same', activation='relu')(conv1)
    return conv2


def create_model():
    # Make a custom U-nets implementation.
    filters = 64
    input_layer = Input(shape=[input_width, input_height, 3])
    layers = [input_layer]
    residuals = []

    # Down 1, 128
    d1, res1 = down(input_layer, filters)
    residuals.append(res1)

    filters *= 2

    # Down 2, 64
    d2, res2 = down(d1, filters)
    residuals.append(res2)

    filters *= 2

    # Down 3, 32
    d3, res3 = down(d2, filters)
    residuals.append(res3)

    filters *= 2

    # Down 4, 16
    d4, res4 = down(d3, filters)
    residuals.append(res4)

    filters *= 2

    # Down 5, 8
    d5 = down(d4, filters, pool=False)

    # Up 1, 16
    up1 = up(d5, residual=residuals[-1], filters=filters / 2)

    filters /= 2

    # Up 2,  32
    up2 = up(up1, residual=residuals[-2], filters=filters / 2)

    filters /= 2

    # Up 3, 64
    up3 = up(up2, residual=residuals[-3], filters=filters / 2)

    filters /= 2

    # Up 4, 128
    up4 = up(up3, residual=residuals[-4], filters=filters / 2)

    out = Conv2D(filters=1, kernel_size=(1, 1), activation="sigmoid")(up4)

    model = Model(input_layer, out)

    model.summary()
    return model


def predict_mask(model, name, batch_size=32):
    res = []
    ids = []
    gen = test_generator('../../data/test/test_dir/', batch_size, [input_width, input_height])
    imgs, img_ids = next(gen)
    i = 0
    while img_ids:
        i += 1
        print('step {}, len {}'.format(i, len(img_ids)))
        res_batch = model.predict(imgs)
        print(res_batch.shape)
        res_batch = [resize(img, (output_width, output_height)) > 0.5 for img in res_batch]
        res.extend(res_batch)
        ids.extend(list(img_ids))
        imgs, img_ids = next(gen)
    make_submission([ids, res], name)


if __name__ == '__main__':
    # example use
    train_gen = data_gen_small(data_dir, mask_dir, train_images, 32, [input_width, input_height])
    img, msk = next(train_gen)
    model = create_model()
    # model.load_weights('../../model/1.weights')
    model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=[dice_coef])
    model.fit_generator(train_gen, steps_per_epoch=10, epochs=1)
    model.save_weights('../../model/1.weights')
    # test_gen = test_generator('../../data/test/', batch_size=2)
    # pred = model.predict_generator(test_gen, steps=1)
    # make_submission()
    # print(pred)
    predict_mask(model, '1', 100)
