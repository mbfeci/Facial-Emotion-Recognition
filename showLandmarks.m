function y = showLandmarks(filename)
%SHOWLANDMARKS Summary of this function goes here
%   Detailed explanation goes here

I = readIm(filename);
faceReal = getFace(I);
subplot(1,2,1);
imshow(I);

res = find_face_landmarks(I);
l = res.faces(1).landmarks;

subplot(1,2,2);
imshow(I);
hold on;
out = l;
plot(out(:,1), out(:,2), '*g');
end

