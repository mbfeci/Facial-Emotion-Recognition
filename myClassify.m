function y = myClassify(net, filename)
%CLASSÝFY Summary of this function goes here
%   Detailed explanation goes here
%{
faceDetector = vision.CascadeObjectDetector('FrontalFaceLBP');
I = lulread(fullfile('test', filename));
bboxes = step(faceDetector, I);
if size(bboxes,1)==0
    faceDetector2 = vision.CascadeObjectDetector();
    bboxes = step(faceDetector2, I);
end
[dummy,idx] = max(bboxes(:, 3).*bboxes(:, 4));
%}
I = lulread(fullfile('test', filename));
faceReal = getFace(I);
imshow(faceReal);
y = classify(net, faceReal);
end

