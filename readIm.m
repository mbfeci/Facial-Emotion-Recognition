function y = readIm(filename)
y = imread(filename);
if size(y,3)==1
    y = cat(3, y, y, y);
end
end


