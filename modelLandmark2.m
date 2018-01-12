%%
% Type ~=1 means including fear and disgust as well.
type = 2;

%%

if type==1
    base = fullfile('Data', 'data5');
else
    base = fullfile('Data', 'data7');
end

modelFile = 'shape_predictor_68_face_landmarks.dat';

%%
%Train landmark features and labels
trainDataDir = fullfile(base, 'data_noise_train');
fid = fopen(fullfile(trainDataDir,'alabel.txt'), 'r');
count = 1;
trainImages = {};
trainLabels = [];
c = 1;
while ~feof(fid)
    line = fgetl(fid);
    cell = split(line);
    if mod(c, 12) ~= 1
        c = c+1;
        continue;
    end
    if ~exist(fullfile(trainDataDir, cell{1}), 'file')
        continue;
    end
    c = c+1;
    cell{1}
    trainImages{count} = fullfile(trainDataDir, cell{1}); %isim
    trainLabels(count) = str2num(cell{2}); %label
    count = count+1;
end

ntrain = count-1;
trainLandmarks = zeros(ntrain, 136);

unfoundFacesTrain = [];
for ii=1:ntrain
    I = imread(trainImages{ii});
    res = find_face_landmarks(modelFile, I);
    if(length(res.faces)<1)
        unfoundFacesTrain = [unfoundFacesTrain ii];
        trainLandmarks(ii,:) = zeros(1,136);
        continue;
    end
    trainLandmarks(ii,:) = res.faces(1).landmarks(:);
end
length(unfoundFacesTrain)
%%
% Test landmark feature extraction and labels.

testDataDir = fullfile(base, 'data_noise_test');
fid = fopen(fullfile(testDataDir,'alabel.txt'), 'r');
count = 1;
testImages = {};
testLabels = [];
while ~feof(fid)
    line = fgetl(fid);
    cell = split(line);
    if ~exist(fullfile(testDataDir, cell{1}), 'file')
        continue;
    end
    testImages{count} = fullfile(testDataDir, cell{1}); %isim
    testLabels(count) = str2num(cell{2}); %label
    count = count+1;
end

ntest = count-1;
testLandmarks = zeros(ntest, 136);

unfoundFacesTest = [];
for ii=1:ntest
    I = imread(testImages{ii});
    res = find_face_landmarks(modelFile, I);
    if(length(res.faces)<1)
        unfoundFacesTest = [unfoundFacesTest ii];
        testLandmarks(ii,:) = zeros(1,136);
        continue;
    end
    testLandmarks(ii,:) = res.faces(1).landmarks(:);
end
length(unfoundFacesTest)
%%

permTrain = randperm(ntrain);
permTest = randperm(ntest);

trainIndex = setdiff(permTrain,unfoundFacesTrain);
testIndex = setdiff(permTest, unfoundFacesTest);

disp(['Training image set size is: ', num2str(ntrain)]);
disp(['Test image set size is: ', num2str(ntest)]);

trainLabels = trainLabels(trainIndex);
testLabels = testLabels(testIndex);

if type==1
    trainLabels = categorical(trainLabels, [1 2 3 4 5], {'Angry' 'Happy' 'Sad' 'Suprised' 'Neutral'});
    testLabels = categorical(testLabels, [1 2 3 4 5], {'Angry' 'Happy' 'Sad' 'Suprised' 'Neutral'});
else
    trainLabels = categorical(trainLabels, [1 2 3 4 5 6 7], {'Angry' 'Happy' 'Sad' 'Suprised' 'Neutral' 'Scared' 'Disgusted'});
    testLabels = categorical(testLabels, [1 2 3 4 5 6 7], {'Angry' 'Happy' 'Sad' 'Suprised' 'Neutral' 'Scared' 'Disgusted'});
end

trainLandmarks = trainLandmarks(trainIndex,:);
testLandmarks = trainLandmarks(testIndex,:);

classifier = fitcecoc(trainLandmarks,trainLabels);
predictedLabels = predict(classifier,testLandmarks);
accuracy = mean(predictedLabels == testLabels')

function y = reader(img)
    y = imread(img);
    if size(y,3)==1
        y = cat(3,y,y,y);
    end
end


%{

%}