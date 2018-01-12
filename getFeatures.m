function features = getFeatures(model, face, type, layer)
%GETFEATURES 
%   type=1 -> feature of the model extracted from layer
%   type=2 -> feature extracted from model and landmarks

if ~exist('layer','var')
   layer = length(model.Layers)-3;
end

if ~exist('type','var')
   type = 1;
end

features = activations(model, face, layer);
if type~=1
    landmarks = getLandmarkFeatures(face);
    features = horzcat(features,landmarks);
end

end

