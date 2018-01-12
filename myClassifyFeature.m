function y = myClassifyFeature(classifier, filename, type, net, layer)
%CLASSÝFY Summary of this function goes here
%   Detailed explanation goes here
if ~exist('type', 'var')
   type = 3; 
end

I = lulread(fullfile('test', filename));
faceReal = getFace(I);
imshow(faceReal);

if type==3
    feature = getLandmarkFeatures(I);
else
    if ~exist('layer', 'var')
        feature = getFeatures(net, faceReal, type);
    else
        feature = getFeatures(net, faceReal, type, layer);
    end
end

y = predict(classifier, double(feature));
end

