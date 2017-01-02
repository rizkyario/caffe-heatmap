import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imresize
import caffe

# Make sure that caffe is on the python path:
caffe_root = '/Users/rizkyario/Documents/Codes/DeepLearning/caffe/'
caffe_heatmap_root = '/Users/rizkyario/Documents/Codes/DeepLearning/caffe-heatmap/'
# import sys
# sys.path.insert(0, caffe_root + 'python')

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

caffe.set_mode_cpu()
net = caffe.Net(caffe_heatmap_root + 'models/heatmap-flic-fusion/matlab.prototxt',
                caffe_heatmap_root + 'models/heatmap-flic-fusion/caffe-heatmap-flic.caffemodel',
                caffe.TEST)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
# transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
# transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

# set net to batch size of 50
net.blobs['data'].reshape(1, 3, 256, 256)

net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(caffe_heatmap_root + 'python/pose/sign/2.png'))
net.forward()
features = net.blobs['conv5_fusion'].data[...][0]

heatmapResized = np.zeros((7, 256, 256))

for i in range(0, 7):
    heatmapResized[i] = imresize(features[i], (256, 256), mode='F') - 1

joints = np.zeros((7, 2))

for i in range(0, 7):
    sub_img = heatmapResized[i]
    vec = sub_img.flatten()
    print vec
    idx = np.argmax(vec)

    y = (idx.astype('int') / 256)
    x = (idx.astype('int') % 256)
    joints[i] = np.array([x, y])

plt.imshow(transformer.deprocess('data', net.blobs['data'].data[0]))
plt.scatter(joints[0][0], joints[0][1])
plt.scatter(joints[1][0], joints[1][1])
plt.scatter(joints[2][0], joints[2][1])
plt.scatter(joints[3][0], joints[3][1])
plt.scatter(joints[4][0], joints[4][1])
plt.scatter(joints[5][0], joints[5][1])
plt.scatter(joints[6][0], joints[6][1])

plt.show()
