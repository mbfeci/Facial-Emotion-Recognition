function out = getLandmarkFeatures(input)
%GETLANDMARKFEATURES
% 
modelFile = 'shape_predictor_68_face_landmarks.dat';
find_face_landmarks(modelFile);
if isa(input, 'matlab.io.datastore.ImageDatastore')
    images = input.Files;
    nimages = length(images);
    out = zeros(nimages, 136);
    count = 0;
    for ii=1:nimages
        dataDir = cell2mat(images(ii));
        I = imread(dataDir);
        res = find_face_landmarks(modelFile, I);
        if(length(res.faces)~=1)
            count = count+1;
            continue;
        end
        out(ii,:) = res.faces(1).landmarks(:);
    end
    disp([num2str(count), ' landmarks could not be extracted']);
else
    res = find_face_landmarks(input);
    if length(res.faces) < 1
        out = zeros(1,136);
    else
        out = res.faces(1).landmarks;
        out = out-[res.faces(1).bbox(1), res.faces(1).bbox(2)];
        out(:,1) = round(224*out(:,1)/res.faces(1).bbox(3));
        out(:,2) = round(224*out(:,2)/res.faces(1).bbox(4));
        out = out(:)';
    end
end
end

