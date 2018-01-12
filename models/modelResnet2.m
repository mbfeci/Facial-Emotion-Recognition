%%
% Type ~=1 means including fear and disgust as well.
type = 2;

%%

if type==1
    base = fullfile('Data', 'data5');
else
    base = fullfile('Data', 'data7');
end

trainDataDir = fullfile(base, 'data_noise_train');
testDataDir = fullfile(base, 'data_noise_test');

[trainImages, trainLabels] = getImagesAndLabels(trainDataDir);
[testImages, testLabels] = getImagesAndLabels(testDataDir);

trainData = imageDatastore(trainImages);
trainData.ReadFcn = @reader;

if type==1
    trainData.Labels = categorical(trainLabels, [1 2 3 4 5], {'Angry' 'Happy' 'Sad' 'Suprised' 'Neutral'});
else
    trainData.Labels = categorical(trainLabels, [1 2 3 4 5 6 7], {'Angry' 'Happy' 'Sad' 'Suprised' 'Neutral' 'Scared' 'Disgusted'});
end

testData = imageDatastore(testImages);
testData.ReadFcn = @reader;

if type==1
    testData.Labels = categorical(testLabels, [1 2 3 4 5], {'Angry' 'Happy' 'Sad' 'Suprised' 'Neutral'});
else
    testData.Labels = categorical(testLabels, [1 2 3 4 5 6 7], {'Angry' 'Happy' 'Sad' 'Suprised' 'Neutral' 'Scared' 'Disgusted'});
end

disp(['Training image set size is: ', num2str(length(trainData.Files))]);
disp(['Test image set size is: ', num2str(length(testData.Files))]);

netRes = resnet50;
lgraph = layerGraph(netRes);
lgraph = removeLayers(lgraph, { 'fc1000' 'fc1000_softmax' 'ClassificationLayer_fc1000'});

if type==1
    newLayers = [
        fullyConnectedLayer(5,'Name','fc','WeightLearnRateFactor',9,'BiasLearnRateFactor', 9)
        softmaxLayer('Name','softmax')
        classificationLayer('Name','classoutput')];
else
    newLayers = [
        fullyConnectedLayer(7,'Name','fc','WeightLearnRateFactor',9,'BiasLearnRateFactor', 9)
        softmaxLayer('Name','softmax')
        classificationLayer('Name','classoutput')];
end

lgraph = addLayers(lgraph,newLayers);

lgraph = connectLayers(lgraph, 'avg_pool', 'fc');

miniBatchSize = 16;

options = trainingOptions('sgdm',...
    'MiniBatchSize',miniBatchSize,...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropFactor',0.8,...
    'LearnRateDropPeriod',2,...
    'MaxEpochs',12,...
    'InitialLearnRate',1e-4,...
    'Plots','training-progress',...
    'ValidationData',testData,...
    'ValidationFrequency',839, ...
    'ValidationPatience', 2);

netRes = trainNetwork(trainData, lgraph, options);

function y = reader(img)
    y = imread(img);
    if size(y,3)==1
        y = cat(3,y,y,y);
    end
end


%{
%}