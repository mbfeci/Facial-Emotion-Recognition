function y = ensemblePredict(models,testDir)
%ENSEMBLE Summary of this function goes here
% types:
% 1 -> feature of net (type, net, classifier)
% 2 -> feature of both net and landmark (type, net, classifier)
% 3 -> Only landmark (type, classifier)
% 4 -> convnet (type, net)
%   Detailed explanation goes here
% gpuDevice(1)

%testDir = fullfile('test', testDir);
I = readIm(testDir);
face = getFace(I);
imshow(face);

nmodels = length(models);
results = categorical(ones(nmodels, 1), [1 2 3 4 5 6 7], {'Angry' 'Happy' 'Sad' 'Suprised' 'Neutral' 'Scared' 'Disgusted'});


for i=1:nmodels
    model = models{i};
    type = model{1};
    if type==4
        results(i) = classify(model{2}, face);     
    elseif type==3
        landmarkFeatures = double(getLandmarkFeatures(I));
        results(i) = predict(model{2}, landmarkFeatures);
    elseif type==2
        netFeatures = activations(model{2}, testData, layer);
        features = horzcat(netFeatures,landmarkFeatures);
        results(i,:) = predict(model{3}, features)';
    elseif type==1
        netFeatures = activations(model{2}, testData, layer);
        results(i,:) = predict(model{3}, netFeatures);
    end
end

y = mode(results);
end

