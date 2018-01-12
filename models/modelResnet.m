dataDir = fullfile('Data', 'data_all');

data = imageDatastore(dataDir);

labelDir = fullfile(dataDir, 'label.txt');
fileID = fopen(labelDir,'r');
labels = fscanf(fileID, '%d');
fclose(fileID);
data.ReadFcn = @reader;

data.Labels = categorical(labels, [1 2 3 4 5], {'Angry' 'Happy' 'Sad' 'Suprised' 'Neutral'});

[trainImages, validationImages] = splitEachLabel(data, 0.8, 'randomized');

disp(['Training image set size is: ', num2str(length(trainImages.Files))]);
disp(['Validation image set size is: ', num2str(length(validationImages.Files))]);

netRes = resnet50;
lgraph = layerGraph(netRes);
lgraph = removeLayers(lgraph, { 'fc1000' 'fc1000_softmax' 'ClassificationLayer_fc1000'});

newLayers = [
    fullyConnectedLayer(5,'Name','fc','WeightLearnRateFactor',8,'BiasLearnRateFactor', 8)
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')];

lgraph = addLayers(lgraph,newLayers);

lgraph = connectLayers(lgraph, 'avg_pool', 'fc');

miniBatchSize = 8;

options = trainingOptions('sgdm',...
    'MiniBatchSize',miniBatchSize,...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropFactor',0.4,...
    'LearnRateDropPeriod',4,...
    'MaxEpochs',5,...
    'InitialLearnRate',1e-4,...
    'Plots','training-progress',...
    'ValidationData',validationImages,...
    'ValidationFrequency',86, ...
    'ValidationPatience', 2);

netRes = trainNetwork(trainImages, lgraph, options);

function y = reader(img)
    y = imread(img);
    if size(y,3)==1
        y = cat(3,y,y,y);
    end
end


%{
Run 1:
    - LR: 1e-4
    - LRDrop: 0.4
    - LRDropPeriod: 4 (Epochs)
    - Epoch: 5
    - LastLayerLRWeight: 8
Log 1:
Training image set size is: 690
Validation image set size is: 172
Training on single GPU.
Initializing image normalization.
|=======================================================================================================================|
|     Epoch    |   Iteration  | Time Elapsed |  Mini-batch  |  Validation  |  Mini-batch  |  Validation  | Base Learning|
|              |              |  (seconds)   |     Loss     |     Loss     |   Accuracy   |   Accuracy   |     Rate     |
|=======================================================================================================================|
|            1 |            1 |         1.27 |       1.5194 |       1.6042 |       50.00% |       22.67% |     1.00e-04 |
|            1 |           50 |        31.74 |       1.4489 |              |       50.00% |              |     1.00e-04 |
|            1 |           86 |        52.96 |       0.8478 |       1.0345 |       75.00% |       63.37% |     1.00e-04 |
|            2 |          100 |        64.81 |       1.1984 |              |       75.00% |              |     1.00e-04 |
|            2 |          150 |        93.79 |       0.8859 |              |       62.50% |              |     1.00e-04 |
|            2 |          172 |       106.53 |       0.4373 |       0.8425 |       87.50% |       70.93% |     1.00e-04 |
|            3 |          200 |       124.41 |       0.4908 |              |       75.00% |              |     1.00e-04 |
|            3 |          250 |       153.40 |       0.3847 |              |       87.50% |              |     1.00e-04 |
|            3 |          258 |       158.06 |       0.2766 |       0.7284 |      100.00% |       73.26% |     1.00e-04 |
|            4 |          300 |       185.89 |       0.3053 |              |      100.00% |              |     1.00e-04 |
|            4 |          344 |       212.19 |       0.1985 |       0.6726 |      100.00% |       77.33% |     1.00e-04 |
|            5 |          350 |       220.30 |       0.1754 |              |      100.00% |              |     4.00e-05 |
|            5 |          400 |       251.29 |       0.2388 |              |      100.00% |              |     4.00e-05 |
|            5 |          430 |       269.82 |       0.1121 |       0.6688 |      100.00% |       76.16% |     4.00e-05 |
|=======================================================================================================================|
%}