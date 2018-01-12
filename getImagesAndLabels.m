function [images, labels] = getImagesAndLabels(dataDir)
%GETIMAGESANDLABELS Summary of this function goes here
%   Detailed explanation goes here
fid = fopen(fullfile(dataDir,'alabel.txt'), 'r');
count = 1;
start = 14000;
images = cell(1,start);
labels = zeros(start,1);
while ~feof(fid)
    line = fgetl(fid);
    lines = split(line);
    if ~exist(fullfile(dataDir, lines{1}), 'file')
        continue;
    end
    images{count} = fullfile(dataDir, lines{1}); %isim
    labels(count) = str2double(lines{2}); %label
    count = count+1;
end
images = images(1:(count-1));
labels = labels(1:count-1);
fclose(fid);

end

