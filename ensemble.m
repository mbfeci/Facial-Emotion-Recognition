function [accuracy,results] = ensemble(models,testDir)
%ENSEMBLE Summary of this function goes here
% types:
% 1 -> feature of net (type, net, classifier)
% 2 -> feature of both net and landmark (type, net, classifier)
% 3 -> Only landmark (type, classifier)
% 4 -> convnet (type, net)
%   Detailed explanation goes here
device = gpuDevice();
nmodels = length(models);

[testImages, testLabels] = getImagesAndLabels(testDir);

testLabels = categorical(testLabels, [1 2 3 4 5 6 7], {'Angry' 'Happy' 'Sad' 'Suprised' 'Neutral' 'Scared' 'Disgusted'});

testData = imageDatastore(testImages);

testData.ReadFcn = @reader;
%testData.Labels = testLabels;
results = categorical(ones(nmodels, length(testLabels)), [1 2 3 4 5 6 7], {'Angry' 'Happy' 'Sad' 'Suprised' 'Neutral' 'Scared' 'Disgusted'});

disp(['Available GPU Memory: ', num2str(device.AvailableMemory)]);
for i=1:nmodels
    model = models{i};
    type = model{1};
    disp(['Type: ', num2str(type)]);
    if type==4
        results(i,:) = classify(model{2}, testData, 'MiniBatchSize', 16);     
        disp(['Available GPU Memory: ', num2str(device.AvailableMemory)]);
    elseif type==3
        landmarkFeatures = getLandmarkFeatures(testData);
        results(i,:) = predict(model{2}, landmarkFeatures);
        disp(['Available GPU Memory: ', num2str(device.AvailableMemory)]);
    elseif type==2
        netFeatures = activations(model{2}, testData, layer);
        features = horzcat(netFeatures,landMarkFeatures);
        results(i,:) = predict(model{3}, features)';
        disp(['Available GPU Memory: ', num2str(device.AvailableMemory)]);
    elseif type==1
        netFeatures = activations(model{2}, testData, layer);
        results(i,:) = predict(model{3}, netFeatures);
        disp(['Available GPU Memory: ', num2str(device.AvailableMemory)]);
    end
end

predictedLabels = mode(results);
accuracy = mean(predictedLabels == testLabels');

function y = reader(img)
    y = imread(img);
    if size(y,1) ~=224 || size(y,2) ~=224
       y = getFace(y); 
    end
    if size(y,3)==1
        y = cat(3,y,y,y);
    end
end
end

