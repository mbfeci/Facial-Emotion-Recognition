base = fullfile('Data/data5', 'data_all');
fid = fopen(fullfile(base,'alabel.txt'), 'r');
count = 1;
images = {};
labels = [];
while ~feof(fid)
    line = fgetl(fid);
    cell = split(line);
    if ~exist(fullfile(base, cell{1}))
        continue;
    end
    images{count} = fullfile(base, cell{1}); %isim
    labels(count) = str2num(cell{2}); %label
    count = count+1;
end
fclose(fid);

data = imageDatastore(images);
data.Labels = categorical(labels, [1 2 3 4 5], {'Angry' 'Happy' 'Sad' 'Suprised' 'Neutral'});
data.ReadFcn = @reader;

[trainData, testData] = splitEachLabel(data, 0.8, 'randomized');

disp(['Training image set size is: ', num2str(length(trainData.Files))]);
disp(['Validation image set size is: ', num2str(length(testData.Files))]);

leNet = googlenet;
lgraph = layerGraph(leNet);
lgraph = removeLayers(lgraph, {'loss3-classifier','prob','output'});

newLayers = [
    fullyConnectedLayer(5,'Name','fc','WeightLearnRateFactor',8,'BiasLearnRateFactor', 8)
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')];

lgraph = addLayers(lgraph,newLayers);

lgraph = connectLayers(lgraph, 'pool5-drop_7x7_s1', 'fc');

miniBatchSize = 8;

options = trainingOptions('sgdm',...
    'MiniBatchSize',miniBatchSize,...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropFactor',0.8,...
    'LearnRateDropPeriod',4,...
    'MaxEpochs',60,...
    'InitialLearnRate',1e-4,...
    'Plots','training-progress',...
    'ValidationData',testData,...
    'ValidationFrequency',95, ...
    'ValidationPatience', 2);

leNet = trainNetwork(trainData, lgraph, options);

function y = reader(img)
    y = imread(img);
    if size(y,3)==1
        y = cat(3,y,y,y);
    end
end


%{
%}