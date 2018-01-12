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

netVGG = vgg16;
layersTransfer = netVGG.Layers(1:end-3);

if type==1
    layers = [layersTransfer
        fullyConnectedLayer(5, 'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10)
        softmaxLayer
        classificationLayer];
else
    layers = [layersTransfer
        fullyConnectedLayer(7, 'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10)
        softmaxLayer
        classificationLayer];
end

miniBatchSize = 64;

options = trainingOptions('sgdm',...
    'MiniBatchSize',miniBatchSize,...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropFactor',0.8,...
    'LearnRateDropPeriod',3,...
    'MaxEpochs',21,...
    'InitialLearnRate',1e-4,...
    'Plots','training-progress',...
    'ValidationData',testData,...
    'ValidationFrequency',209, ...
    'ValidationPatience', 2);

netVGG = trainNetwork(trainData, layers, options);

function y = reader(img)
    y = imread(img);
    if size(y,3)==1
        y = cat(3,y,y,y);
    end
end

%{
Run 1:
    - data noise
    - LR: 1e-4
    - LRDrop: 0.8
    - LRDropPeriod: 2 (Epochs)
    - Epoch: 12
    - LastLayerLRWeight: 10

Log 1:
Training image set size is: 8340
Validation image set size is: 1788
Training on single GPU.
Initializing image normalization.
|=======================================================================================================================|
|     Epoch    |   Iteration  | Time Elapsed |  Mini-batch  |  Validation  |  Mini-batch  |  Validation  | Base Learning|
|              |              |  (seconds)   |     Loss     |     Loss     |   Accuracy   |   Accuracy   |     Rate     |
|=======================================================================================================================|
|            1 |            1 |         5.08 |       1.8964 |       1.5247 |       29.69% |       36.74% |     1.00e-04 |
|            1 |           50 |       264.37 |       0.9459 |              |       67.19% |              |     1.00e-04 |
|            1 |          100 |       475.53 |       0.9208 |              |       67.19% |              |     1.00e-04 |
|            1 |          130 |       602.31 |       0.6287 |       0.7488 |       81.25% |       70.97% |     1.00e-04 |
|            2 |          150 |       733.84 |       0.7807 |              |       68.75% |              |     1.00e-04 |
|            2 |          200 |       944.85 |       0.6298 |              |       79.69% |              |     1.00e-04 |
|            2 |          250 |      1152.73 |       0.3352 |              |       89.06% |              |     1.00e-04 |
|            2 |          260 |      1194.67 |       0.3212 |       0.5726 |       89.06% |       78.80% |     1.00e-04 |
|            3 |          300 |      1409.42 |       0.3410 |              |       87.50% |              |     8.00e-05 |
|            3 |          350 |      1617.86 |       0.2174 |              |       89.06% |              |     8.00e-05 |
|            3 |          390 |      1785.29 |       0.2584 |       0.5158 |       92.19% |       82.10% |     8.00e-05 |
|            4 |          400 |      1876.53 |       0.2315 |              |       92.19% |              |     8.00e-05 |
|            4 |          450 |      2085.42 |       0.1433 |              |       98.44% |              |     8.00e-05 |
|            4 |          500 |      2294.29 |       0.0791 |              |       98.44% |              |     8.00e-05 |
|            4 |          520 |      2377.84 |       0.1074 |       0.4761 |       95.31% |       85.12% |     8.00e-05 |
|            5 |          550 |      2550.66 |       0.1571 |              |       95.31% |              |     6.40e-05 |
|            5 |          600 |      2759.32 |       0.1017 |              |       96.88% |              |     6.40e-05 |
|            5 |          650 |      2967.07 |       0.0817 |       0.4420 |       96.88% |       86.69% |     6.40e-05 |
|            6 |          700 |      3222.16 |       0.0295 |              |      100.00% |              |     6.40e-05 |
|            6 |          750 |      3434.70 |       0.0648 |              |      100.00% |              |     6.40e-05 |
|            6 |          780 |      3562.01 |       0.0710 |       0.4901 |       96.88% |       86.86% |     6.40e-05 |
|            7 |          800 |      3696.72 |       0.1138 |              |       95.31% |              |     5.12e-05 |
|            7 |          850 |      3906.62 |       0.0827 |              |       95.31% |              |     5.12e-05 |
|            7 |          900 |      4117.11 |       0.0377 |              |      100.00% |              |     5.12e-05 |
|            7 |          910 |      4158.76 |       0.0279 |       0.4663 |      100.00% |       87.30% |     5.12e-05 |
|=======================================================================================================================|

Run 2:
    - data noise
    - LR: 1e-4
    - LRDrop: 0.8
    - LRDropPeriod: 3 (Epochs)
    - Epoch: 21
    - LastLayerLRWeight: 10

Log 2:
Training image set size is: 13424
Test image set size is: 315
Training on single GPU.
Initializing image normalization.
|=======================================================================================================================|
|     Epoch    |   Iteration  | Time Elapsed |  Mini-batch  |  Validation  |  Mini-batch  |  Validation  | Base Learning|
|              |              |  (seconds)   |     Loss     |     Loss     |   Accuracy   |   Accuracy   |     Rate     |
|=======================================================================================================================|
|            1 |            1 |         6.43 |       2.6276 |       2.0756 |       15.63% |        8.89% |     1.00e-04 |
|            1 |           50 |       225.02 |       1.6236 |              |       48.44% |              |     1.00e-04 |
|            1 |          100 |       438.47 |       1.2090 |              |       59.38% |              |     1.00e-04 |
|            1 |          150 |       650.27 |       0.9390 |              |       73.44% |              |     1.00e-04 |
|            1 |          200 |       861.58 |       0.8311 |              |       68.75% |              |     1.00e-04 |
|            1 |          209 |       899.69 |       0.7758 |       0.6837 |       68.75% |       76.83% |     1.00e-04 |
|            2 |          250 |      1081.14 |       0.6395 |              |       79.69% |              |     1.00e-04 |
|            2 |          300 |      1291.03 |       0.5824 |              |       73.44% |              |     1.00e-04 |
|            2 |          350 |      1499.01 |       0.5224 |              |       78.13% |              |     1.00e-04 |
|            2 |          400 |      1708.90 |       0.5051 |              |       78.13% |              |     1.00e-04 |
|            2 |          418 |      1783.74 |       0.3806 |       0.4776 |       90.63% |       83.49% |     1.00e-04 |
|            3 |          450 |      1927.10 |       0.2590 |              |       92.19% |              |     1.00e-04 |
|            3 |          500 |      2137.85 |       0.3525 |              |       87.50% |              |     1.00e-04 |
|            3 |          550 |      2348.30 |       0.2881 |              |       85.94% |              |     1.00e-04 |
|            3 |          600 |      2557.57 |       0.3157 |              |       90.63% |              |     1.00e-04 |
|            3 |          627 |      2669.90 |       0.2106 |       0.4243 |       92.19% |       85.71% |     1.00e-04 |
|            4 |          650 |      2775.53 |       0.2068 |              |       93.75% |              |     8.00e-05 |
|            4 |          700 |      2984.59 |       0.2350 |              |       95.31% |              |     8.00e-05 |
|            4 |          750 |      3194.97 |       0.2216 |              |       95.31% |              |     8.00e-05 |
|            4 |          800 |      3404.93 |       0.1241 |              |       96.88% |              |     8.00e-05 |
|            4 |          836 |      3556.87 |       0.0733 |       0.4216 |       96.88% |       89.21% |     8.00e-05 |
|            5 |          850 |      3623.94 |       0.1692 |              |       93.75% |              |     8.00e-05 |
|            5 |          900 |      3834.23 |       0.1545 |              |       95.31% |              |     8.00e-05 |
|            5 |          950 |      4043.77 |       0.1594 |              |       92.19% |              |     8.00e-05 |
|            5 |         1000 |      4254.46 |       0.1012 |              |       96.88% |              |     8.00e-05 |
|            5 |         1045 |      4443.23 |       0.0663 |       0.4580 |       98.44% |       89.21% |     8.00e-05 |
|            6 |         1050 |      4473.48 |       0.0509 |              |       98.44% |              |     8.00e-05 |
|            6 |         1100 |      4681.80 |       0.0976 |              |       96.88% |              |     8.00e-05 |
|            6 |         1150 |      4891.29 |       0.0804 |              |       96.88% |              |     8.00e-05 |
|            6 |         1200 |      5101.95 |       0.1316 |              |       95.31% |              |     8.00e-05 |
|            6 |         1250 |      5312.80 |       0.0672 |              |       96.88% |              |     8.00e-05 |
|            6 |         1254 |      5329.56 |       0.0588 |       0.5035 |       96.88% |       89.84% |     8.00e-05 |
|=======================================================================================================================|


%}