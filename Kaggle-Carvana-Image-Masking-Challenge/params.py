from model.u_net import get_unet_128, get_unet_256, get_unet_512, get_unet_1024


max_epochs = 100
batch_size = 32

orig_width = 1918
orig_height = 1280

input_size = 128
input_w = 1920 // 2
input_h = 1280 // 2
threshold = 0.5

model_factory = get_unet_128(input_shape=(input_h, input_w, 3))
