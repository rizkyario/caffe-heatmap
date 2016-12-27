% This file uses a FLIC trained model and applies it to a video sequence from Poses in the Wild
%
% Download the model:
%    wget http://tomas.pfister.fi/models/caffe-heatmap-flic.caffemodel -P ../../models/heatmap-flic-fusion/

% Options

opt.visualise = true;		% Visualise predictions?
opt.useGPU = true; 			% Run on GPU
opt.dims = [256 256]; 		% Input dimensions (needs to match matlab.txt)
opt.numJoints = 7; 			% Number of joints
opt.layerName = 'conv5_fusion'; % Output layer name
opt.modelDefFile = '/Users/rizkyario/Documents/Codes/DeepLearning/caffe-heatmap/models/heatmap-flic-fusion/matlab.prototxt'; % Model definition
opt.modelFile = '/Users/rizkyario/Documents/Codes/DeepLearning/caffe-heatmap/models/heatmap-flic-fusion/caffe-heatmap-flic.caffemodel'; % Model weights

% Add caffe matlab into path
addpath('/Users/rizkyario/Documents/Codes/DeepLearning/caffe/matlab')

% Image input directory
opt.inputDir = '/Users/rizkyario/Documents/Codes/DeepLearning/caffe-heatmap/matlab/pose/sign/';

% Create image file list
imInds = 1:22;
for ind = 1:numel(imInds); files{ind} = [num2str(imInds(ind)) '.png']; end

% Apply network
joints = applyNet(files, opt)