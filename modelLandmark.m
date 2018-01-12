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
base = fullfile(base, 'data_all');
fid = fopen(fullfile(base,'alabel.txt'), 'r');
count = 1;
images = {};
labels = [];
while ~feof(fid)
    line = fgetl(fid);
    cell = split(line);
    if ~exist(fullfile(base, cell{1}), 'file')
        continue;
    end
    images{count} = fullfile(base, cell{1}); %isim
    labels(count) = str2num(cell{2}); %label
    count = count+1;
end
nimages = count-1;
imageLandmarks = zeros(nimages, 136);

unfoundFaces = [];
for ii=1:nimages
    I = imread(images{ii});
    res = find_face_landmarks(modelFile, I);
    if(length(res.faces)~=1)
        unfoundFaces = [unfoundFaces ii];
        continue;
    end
    imageLandmarks(ii,:) = res.faces(1).landmarks(:);
end

perm = randperm(nimages);
ntrain = floor(0.8*nimages);
ntest = nimages-ntrain;

trainIndex = perm(setdiff(1:ntrain,unfoundFaces));
testIndex = perm(setdiff((ntrain+1):length(perm),unfoundFaces));

disp(['Training image set size is: ', num2str(ntrain)]);
disp(['Validation image set size is: ', num2str(ntest)]);

trainLabels = labels(trainIndex);
testLabels = labels(testIndex);

if type==1
    trainLabels = categorical(trainLabels, [1 2 3 4 5], {'Angry' 'Happy' 'Sad' 'Suprised' 'Neutral'});
    testLabels = categorical(testLabels, [1 2 3 4 5], {'Angry' 'Happy' 'Sad' 'Suprised' 'Neutral'});
else
    trainLabels = categorical(trainLabels, [1 2 3 4 5 6 7], {'Angry' 'Happy' 'Sad' 'Suprised' 'Neutral' 'Scared' 'Disgusted'});
    testLabels = categorical(testLabels, [1 2 3 4 5 6 7], {'Angry' 'Happy' 'Sad' 'Suprised' 'Neutral' 'Scared' 'Disgusted'});
end


trainLandmarks = imageLandmarks(trainIndex,:);
testLandmarks = imageLandmarks(testIndex,:);

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
Training image set size is: 1165
Validation image set size is: 292

accuracy =

    0.7692
%}