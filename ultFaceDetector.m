function box = ultFaceDetector(I)
%ULTFACEDETECTOR Summary of this function goes here
%   Detailed explanation goes here
modelFile = 'shape_predictor_68_face_landmarks.dat';
res = find_face_landmarks(modelFile, I);
if length(res.faces) > 0
    box = res.faces(1).bbox;
    if box(1) < 1
        box(3) = box(3)+box(1)-1;
        box(1) = 1;
    end
    
    if box(2) < 1
        box(4) = box(4)+box(2)-1;
        box(2) = 1;
    end
    
    if box(1)+box(3)>size(I,2)
        box(3) = size(I,2)-box(1);
    end
    if box(2)+box(4)>size(I,1)
        box(4) =size(I,1)-box(2);
    end
else
    faceDetector = vision.CascadeObjectDetector('FrontalFaceLBP');
    box = step(faceDetector, I);
    if size(box,1)==0
        faceDetector2 = vision.CascadeObjectDetector();
        box = step(faceDetector2, I);
    end
    [dummy,idx] = max(box(:, 3).*box(:, 4));
    box = box(idx,:);
end
end

