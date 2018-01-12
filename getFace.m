function faceReal = getFace(I)
%GETRESÝZEDFACE Summary of this function goes here
%   Detailed explanation goes here
bboxes = ultFaceDetector(I);
face = I(bboxes(2):bboxes(2)+bboxes(4), bboxes(1):bboxes(1)+bboxes(3), :);
faceReal = imresize(face, [224, 224]);
end