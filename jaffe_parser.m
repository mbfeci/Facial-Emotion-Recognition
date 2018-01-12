%%%% LABELS %%%%
% ANGRY    = 1 %
% HAPPY    = 2 %
% SAD      = 3 %
% SURPRISE = 4 %
% NEUTRAL  = 5 %
% FEAR     = 6 %
% DISGUST  = 7 %
%%%%%%%%%%%%%%%%

%% Parser for JAFFE database
% Returns a cell array in which each row contains [Name, Image, Label].
%   type: if 1, ignore DISGUST and FEAR.
function y = jaffe_parser(root, type)
    if (~exist('type', 'var'))
        type = 0;
    end
    c = 1;
    y = {};
    faceDetector = vision.CascadeObjectDetector;
    files = dir(fullfile(root,'jaffe','*.tiff'));
    for i=1:length(files)
        if type == 1
            % Ignore DISGUST and FEAR
            if contains(files(i).name, '.DI')
                continue;
            end
            if contains(files(i).name, '.FE')
                continue;
            end
        end
        % Read the image and detect the face, if can be detected.
        I = imread(fullfile(files(i).folder, files(i).name));
        f = step(faceDetector, I);
        if (size(f, 1) == 0)
            continue;
        end
        % If multiple faces are found, get the biggest one which happens
        % to be a heuristic. Note that JAFFE database contains only one
        % face in each image.
        [~,x] = max(f(:, 3).*f(:, 4));
        y{c,1} = strcat('jaffe_', files(i).name, '.png');
        y{c,2} = imresize(I(f(x,2):f(x,2)+f(x,4), f(x,1):f(x,1)+f(x,3), :),[224, 224]);
        y{c,3} = getLabel(files(i).name);
        c = c + 1;
    end
end

% Returns the correct emotion for the given image.
function l = getLabel(name)
    if contains(name, '.AN')
        l = 1;
    elseif contains(name, '.HA')
        l = 2;
    elseif contains(name, '.SA')
        l = 3;
    elseif contains(name, '.SU')
        l = 4;
    elseif contains(name, '.NE')
        l = 5;
    elseif contains(name, '.FE')
        l = 6;
    elseif contains(name, '.DI')
        l = 7;
    end
end

