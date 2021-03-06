import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.misc import imresize
from SaveFigureAsImage import SaveFigureAsImage
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
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]

# set net to batch size of 50
net.blobs['data'].reshape(1, 3, 256, 256)
net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(caffe_heatmap_root + 'python/pose/sign/1.png'))
net.forward()
features = net.blobs['conv5_fusion'].data[...][0]

heatmapResized = np.zeros((7, 256, 256))

for i in range(0, 7):
    heatmapResized[i] = imresize(features[i], (256, 256), mode='F') - 1

joints = np.zeros((7, 2))

for i in range(0, 7):
    sub_img = heatmapResized[i]
    vec = sub_img.flatten()
    idx = np.argmax(vec)

    y = (idx.astype('int') / 256)
    x = (idx.astype('int') % 256)
    joints[i] = np.array([x, y])

plt.imshow(transformer.deprocess('data', net.blobs['data'].data[0]))

plt.plot([joints[1][0], joints[3][0]], [joints[1][1], joints[3][1]], '.r-', linewidth=3, zorder=1)
plt.plot([joints[3][0], joints[5][0]], [joints[3][1], joints[5][1]], '.g-', linewidth=3, zorder=1)

plt.plot([joints[2][0], joints[4][0]], [joints[2][1], joints[4][1]], '.r-', linewidth=3, zorder=1)
plt.plot([joints[4][0], joints[6][0]], [joints[4][1], joints[6][1]], '.g-', linewidth=3, zorder=1)

cmap = cm.get_cmap(name='rainbow')
for i in range(0, 7):
    plt.scatter(joints[i][0], joints[i][1], color=cmap(i * 256 / 7), s=40, zorder=2)

plt.show()
plt.savefig(caffe_heatmap_root + 'python/pose/output/1.png')