gpuDevice(1)

loaded = true;

if ~loaded
    %These are the models that we have trained
    load('eLandmark7.mat')
    load('eLenet7.mat')
    load('eResnet7.mat')
    load('eVGG7.mat')
    load('ensembleLenetClass7.mat')
    disp('All models are loaded');
end

models = {vgg7 lenet7 resnet7 landmark7 lenetClass7};

%base = fullfile('Data', 'data7');
base = 'test';

%testDataDir = fullfile(base, 'data_noise_test');
testDataDir = fullfile(base, 'alltest');
[~,testLabels] = getImagesAndLabels(testDataDir);
testLabels = categorical(testLabels, [1 2 3 4 5 6 7], {'Angry' 'Happy' 'Sad' 'Suprised' 'Neutral' 'Scared' 'Disgusted'});
[acc, res] = ensemble(models, testDataDir);

[C, order] = confusionmat(testLabels', mode(res));