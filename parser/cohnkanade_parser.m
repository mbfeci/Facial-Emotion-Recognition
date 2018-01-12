%%%% LABELS %%%% CK %
% ANGRY    = 1 %  1 %
% HAPPY    = 2 %  5 %
% SAD      = 3 %  6 %
% SURPRISE = 4 %  7 %
% NEUTRAL  = 5 %  0 %
% FEAR     = 6 %  4 %
% DISGUST  = 7 %  3 %
%%%%%%%%%%%%%%%%%%%%%

%% Parser for CK+ database
% Returns a cell array in which each row contains [Name, Image, Label].
%   type: if 1, ignore DISGUST and FEAR.
function y = cohnkanade_parser(root, type)
    if (~exist('type', 'var'))
        type = 0;
    end
    c = 1;
    y = {};
    faceDetector = vision.CascadeObjectDetector;
    % Browse emotion folder first because not every image is labeled.
    files1 = dir(fullfile(root,'Emotion','S*.*'));
    for i=1:length(files1)
        disp(strcat(num2str(floor(i*100/length(files1))), '%'));
        files2 = dir(fullfile(files1(i).folder,files1(i).name,'0*.*'));
        for j=1:length(files2)
            % Read the txt file, if exists.
            files3 = dir(fullfile(files2(j).folder,files2(j).name,'*.txt'));
            if isempty(files3)
                continue;
            end
            txtfid = fopen(fullfile(files3(1).folder, files3(1).name), 'r');
            label = fscanf(txtfid, '%f');
            fclose(txtfid);
            % Ignore CONTEMPT.
            if label == 2
                continue;
            end
            if type == 1
                % Ignore FEAR and DISGUST.
                if label == 4 || label == 3
                    continue;
                end
            end
            % Detect the face if possible. If multiple faces are found, 
            % get the biggest one which happens to be a heuristic. Note 
            % that CK+ database contains only one face in each image. 
            imfiles = dir(fullfile(root,'cohn-kanade-images', ...
                files1(i).name,files2(j).name,'*.png'));
            % First image is always NEUTRAL.
            I = imread(fullfile(imfiles(1).folder,imfiles(1).name));
            f = step(faceDetector, I);
            if (size(f, 1) > 0)
                y{c,1} = strcat('ck_', imfiles(1).name, '.png');
                [~,x] = max(f(:, 3).*f(:, 4));
                y{c,2} = imresize(I(f(x,2):f(x,2)+f(x,4), f(x,1):f(x,1)+f(x,3), :),[224, 224]);
                y{c,3} = 5;
                c = c + 1;
            end
            % Last image is what label says.
            I = imread(fullfile(imfiles(length(imfiles)).folder,imfiles(length(imfiles)).name));
            f = step(faceDetector, I);
            if (size(f, 1) > 0)
                y{c,1} = strcat('ck_', imfiles(length(imfiles)).name, '.png');
                [~,x] = max(f(:, 3).*f(:, 4));
                y{c,2} = imresize(I(f(x,2):f(x,2)+f(x,4), f(x,1):f(x,1)+f(x,3), :),[224, 224]);
                y{c,3} = correctLabel(label);
                c = c + 1;
            end
        end
    end
end

function y = correctLabel(label)
    if label == 1
        y = 1;
    elseif label == 5
        y = 2;
    elseif label == 6
        y = 3;
    elseif label == 7
        y = 4;
    elseif label == 0
        y = 5;
    elseif label == 4
        y = 6;
    elseif label == 3
        y = 7;
    end
end