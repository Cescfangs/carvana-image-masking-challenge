import argparse

input_height, input_width = 160, 240

parser = argparse.ArgumentParser(description='main')
parser.add_argument('-b', metavar='batch_size', required=True, type=int, dest='batch_size')
parser.add_argument('-e', metavar='epochs', type=int, required=True, dest='epochs')
parser.add_argument('--load', action='store_true', default=False, dest='load')

args = parser.parse_args()
batch_size = args.batch_size
epochs = args.epochs
load_weights = args.load
print('batch_size: {}\nepochs: {}\nload from disk: {}'.format(batch_size, epochs, load_weights))

if __name__ == '__main__':
    from unet import myUnet
    net = myUnet(img_rows=input_height, img_cols=input_width)
    net.train(batch_size=batch_size, epochs=epochs, load_weights=load_weights, weights_path='../../model/unet.hdf5')
    net.predict_batch('2', batch_size=256)
