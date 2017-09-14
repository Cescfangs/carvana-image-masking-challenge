# import os
from unet import myUnet
data_dir = "../../data/train/"
mask_dir = "../../data/train_masks/"
# all_images = os.listdir(data_dir)
output_width, output_height = 1918, 1280
input_width, input_height = 240, 160
# pick which images we will use for testing and which for validation
# train_images, validation_images = train_test_split(all_images, train_size=0.8, test_size=0.2)



if __name__ == '__main__':
    # example use
    # train_gen = data_gen_small(data_dir, mask_dir, train_images, 32, (input_width, input_height))
    # img, msk = next(train_gen)
    # model = create_model()
    # # model.load_weights('../../model/1.weights')
    # model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=[dice_coef])
    # model.fit_generator(train_gen, steps_per_epoch=32, epochs=1)
    # model.save_weights('../../model/1.weights')
    # predict_mask(model, '1', 25)
    net = myUnet(img_rows=input_height, img_cols=input_width)
    net.train(batch_size=32, epochs=10, load_weights=False, weights_path='../../model/unet.hdf5')
    net.predict_batch('2', batch_size=256)
