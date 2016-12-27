% Apply network to a single image
function joints = applyNetImage(imgFile, net, opt)

% Read & reformat input image
img = imread(imgFile);
input_data = prepareImagePose(img, opt);

% Forward pass
tic
net.forward({input_data});
features = net.blobs(opt.layerName).get_data();
[joints, heatmaps] = processHeatmap(features, opt);
disp(toc); 

% Visualisation
if opt.visualise
    % Heatmap
    heatmapVis = getConfidenceImage(heatmaps, img);
    figure(2),imshow(heatmapVis);
    [pathstr,name,ext] = fileparts(imgFile)
    imwrite(heatmapVis,strcat('heatmap_',name,ext));

    % Original image overlaid with joints
    figure(1),imshow(uint8(img));
    
    hold on
    plotSkeleton(joints, [], []);
    f = getframe(gca);
    img = frame2im(f);

    imwrite(uint8(img),strcat('join_',name,ext));
    hold off
    
end

