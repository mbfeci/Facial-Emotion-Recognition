trainDir = fullfile('Data', 'data_all_train');
testDir = fullfile('Data', 'data_all_test');

trainData = imageDatastore(trainDir);
testData = imageDatastore(testDir);
trainLabelDir = fullfile(trainDir, 'label.txt');
testLabelDir = fullfile(testDir, 'label.txt');

fileID = fopen(trainLabelDir,'r');
trainLabels = fscanf(fileID, '%d');
fclose(fileID);
fileID = fopen(testLabelDir, 'r');
testLabels = fscanf(fileID, '%d');
fclose(fileID);
trainData.ReadFcn = @reader;
testData.ReadFcn = @reader;

trainData.Labels = categorical(trainLabels, [1 2 3 4 5], {'Angry' 'Happy' 'Sad' 'Suprised' 'Neutral'});
testData.Labels = categorical(testLabels, [1 2 3 4 5], {'Angry' 'Happy' 'Sad' 'Suprised' 'Neutral'});

disp(['Training image set size is: ', num2str(length(trainData.Files))]);
disp(['Validation image set size is: ', num2str(length(testData.Files))]);

netVGG = vgg16;
layer = 'fc7';

trainingLandmarks = getLandmarkFeatures(trainData);
testLandmarks = getLandmarkFeatures(testData);
disp('Landmark features have been extracted');

trainingFeatures = activations(netVGG, trainData, layer);
testFeatures = activations(netVGG, testData, layer);

disp('VGG Features have been extracted');

trainingFeatures = horzcat(trainingFeatures, trainingLandmarks);
testFeatures = horzcat(testFeatures, testLandmarks);

trainingLabels = trainData.Labels;
testLabels = testData.Labels;

classifier = fitcecoc(trainingFeatures,trainingLabels);
disp('SVM learning is finished.');

predictedLabels = predict(classifier,testFeatures);
accuracy = mean(predictedLabels == testLabels);
disp(['Test accuracy is: ', num2str(accuracy)]);

function y = reader(img)
    y = imread(img);
    if size(y,3)==1
        y = cat(3,y,y,y);
    end
end


%{

%}