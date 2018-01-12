function y = classifyLandmark(classifier,filename,modelFile)
%CLASSÝFYLANDMARK Summary of this function goes here
%   Detailed explanation goes here
imgDir = fullfile('test', filename);
faceDetector = vision.CascadeObjectDetector('FrontalFaceLBP');
I = readIm(imgDir);
bboxes = step(faceDetector, I);
if size(bboxes,1)==0
    faceDetector2 = vision.CascadeObjectDetector();
    bboxes = step(faceDetector2, I)
end

[dummy,idx] = max(bboxes(:, 3).*bboxes(:, 4));
I = I(bboxes(idx,2):bboxes(idx,2)+bboxes(idx,4), bboxes(idx,1):bboxes(idx,1)+bboxes(idx,3), :);

faceReal = imresize(I, [224, 224]);
imshow(faceReal);

res = find_face_landmarks(modelFile, faceReal);
y = predict(classifier, double(res.faces(1).landmarks(:))');
end