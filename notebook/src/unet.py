import numpy as np
from keras.models import *
from keras.layers import Input, merge, Conv2D, MaxPool2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, Concatenate, BatchNormalization
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from util import *
from cv2 import resize, imread
# from data import dataProcess


def test_generator(data_dir, batch_size, dims=None):
    all_images = os.listdir(data_dir)
    all_images = [img for img in all_images if img.endswith('.jpg')]
    while 1:
        img_ids = all_images[:batch_size]
        all_images = all_images[batch_size:]
        imgs = np.array([imread(data_dir + img_id) for img_id in img_ids])

        # imgs = np.concatenate(imgs) / 255
        # imgs = np.array([imresize(img, dims + [3]) for img in imgs]) / 255
        yield imgs, img_ids


def down(input_layer, filters, pool=True):
    conv1 = Conv2D(filters, (3, 3), padding='same', activation='relu')(input_layer)
    conv1  = BatchNormalization()(conv1)
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
    upconv = BatchNormalization()(upconv)
    concat = Concatenate(axis=3)([residual, upconv])
    conv1 = Conv2D(filters, (3, 3), padding='same', activation='relu')(concat)
    conv2 = Conv2D(filters, (3, 3), padding='same', activation='relu')(conv1)
    return conv2


def create_model(input_width, input_height):
    # Make a custom U-nets implementation.
    filters = 32
    input_layer = Input(shape=(input_height, input_width, 3))
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
    model.compile(optimizer=Adam(lr=6e-5), loss=bce_dice_loss, metrics=[dice_coef])
    model.summary()

    return model


class myUnet(object):

    def __init__(self, img_rows=512, img_cols=512, res_dir='../../data/results'):

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.output_dir = res_dir
        self.get_unet()

    def load_data(self, train_dir='../../data/train_small/',
                  mask_dir='../../data/train_masks_small/',
                  test_dir='../../data/test_small/'):

        train = np.load(os.path.join(train_dir, 'img.npy'))
        mask = np.load(os.path.join(mask_dir, 'img.npy'))
        # test = np.load(os.path.join(test_dir, 'img.npy'))

        assert (train.ndim == 4 and
                mask.ndim == 3 and
                train.shape[:-1] == mask.shape)
        mask = np.expand_dims(mask, axis=3)
        self.test_dir = test_dir
        self.train_dir = train_dir

        return train, mask

    def get_unet(self):
        self.model = create_model(self.img_cols, self.img_rows)

        # filters = 16
        # inputs = Input((self.img_rows, self.img_cols, 3))

        # conv1 = Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        # conv1 = Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        # pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        # conv2 = Conv2D(filters * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        # conv2 = Conv2D(filters * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        # pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        # conv3 = Conv2D(filters * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        # conv3 = Conv2D(filters * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        # pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        # conv4 = Conv2D(filters * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        # conv4 = Conv2D(filters * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        # drop4 = Dropout(0.5)(conv4)
        # pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        # conv5 = Conv2D(filters * 16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        # conv5 = Conv2D(filters * 16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        # # drop5 = Dropout(0.5)(conv5)

        # up6 = Conv2D(filters * 8, 2, activation='relu', padding='same',
        #              kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
        # merge6 = merge([drop4, up6], mode='concat', concat_axis=3)
        # conv6 = Conv2D(filters * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        # conv6 = Conv2D(filters * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

        # up7 = Conv2D(filters * 4, 2, activation='relu', padding='same',
        #              kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
        # merge7 = merge([conv3, up7], mode='concat', concat_axis=3)
        # conv7 = Conv2D(filters * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        # conv7 = Conv2D(filters * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

        # up8 = Conv2D(filters * 2, 2, activation='relu', padding='same',
        #              kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
        # merge8 = merge([conv2, up8], mode='concat', concat_axis=3)
        # conv8 = Conv2D(filters * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        # conv8 = Conv2D(filters * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        # up9 = Conv2D(filters, 2, activation='relu', padding='same',
        #              kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
        # merge9 = merge([conv1, up9], mode='concat', concat_axis=3)
        # conv9 = Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        # conv9 = Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        # conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        # conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

        # model = Model(input=inputs, output=conv10)
        # model.compile(optimizer=Adam(lr=1e-4), loss=bce_dice_loss, metrics=[dice_coef])
        # self.model = model

    @fn_timer
    def train(self, batch_size=32, epochs=10, load_weights=False, **kwargs):
        print("loading data")
        imgs_train, imgs_mask_train = self.load_data()
        print("loading data done")
        self.get_unet()
        print("got unet")

        if load_weights:
            if 'weights_path' not in list(kwargs.keys()):
                raise KeyError('always provide weights_path when `load_weights` set to `True`')
            self.model.load_weights(kwargs['weights_path'])

        else:
            model_checkpoint = ModelCheckpoint('../../model/unet.hdf5', monitor='loss', verbose=1, save_best_only=True)
            print('Fitting model...')
            self.model.fit(imgs_train, imgs_mask_train, batch_size=batch_size, epochs=epochs,
                           verbose=1, shuffle=True, callbacks=[model_checkpoint])

        # print('predict test data')
        # imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
        # np.save('imgs_mask_test.npy', imgs_mask_test)

    @fn_timer
    def predict_mask(self, name, batch_size=32, output_shape=(1918, 1280)):
        res = []
        step = 0
        data = self.test_data
        ids = np.load('../../data/test_small/img_ids.npy')
        while len(data):
            step += 1
            imgs = data[: batch_size]
            data = data[batch_size:]
            print(('step {}, len {}'.format(step, len(imgs))))
            res_batch = self.model.predict(imgs)
            res_batch = [resize(img, output_shape) > 0.5 for img in res_batch]
            res_batch = [rle(img) for img in res_batch]
            res.extend(res_batch)
        make_submission((ids, res), name)

    @fn_timer
    def predict_batch(self, name, batch_size, output_shape=(1918, 1280)):
        res = []
        ids = []
        gen = test_generator(self.test_dir, batch_size)
        imgs, img_ids = next(gen)
        step = 0
        while img_ids:
            step += 1
            print('step {}, len {}'.format(step, len(img_ids)))
            res_batch = self.model.predict(imgs)
            res_batch = [resize(img, output_shape) > 0.5 for img in res_batch]
            res_batch = [rle(img) for img in res_batch]
            res.extend(res_batch)
            ids.extend(list(img_ids))
            imgs, img_ids = next(gen)
        make_submission([ids, res], name)


if __name__ == '__main__':
    myunet = myUnet(img_rows=160, img_cols=240)
    myunet.train()
