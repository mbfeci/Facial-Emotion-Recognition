%%%% LABELS %%%%
% ANGRY    = 1 %
% HAPPY    = 2 %
% SAD      = 3 %
% SURPRISE = 4 %
% NEUTRAL  = 5 %
% FEAR     = 6 %
% DISGUST  = 7 %
%%%%%%%%%%%%%%%%

%% Parser for homedata database
% Returns a cell array in which each row contains [Name, Image, Label].
%   type: if 1, ignore DISGUST and FEAR.
function y = homedata_parser(root, type)
   if (~exist('type', 'var'))
        type = 0;
    end
    c = 1;
    y = {};
    faceDetector = vision.CascadeObjectDetector;
    % Browse emotion folder first because not every image is labeled.
    files1 = dir(fullfile(root,'homedata','0*'));
    for i=1:length(files1)
        disp(strcat(num2str(floor(i*100/length(files1))), '%'));
        % Ignore FEAR and DISGUST.
        if type == 1 && i == 6
            break;
        end
        files2 = dir(fullfile(files1(i).folder,files1(i).name,'*.jpg'));
        for j=1:length(files2)
            % Detect the face if possible. If multiple faces are found, 
            % get the biggest one which happens to be a heuristic. Note 
            % that homedata database contains only one face in each image.
            I = imread(fullfile(files2(j).folder,files2(j).name));
            f = step(faceDetector, I);
            if (size(f, 1) > 0)
                y{c,1} = strcat('home_',num2str(i),'_',files2(j).name,'.png');
                [~,x] = max(f(:, 3).*f(:, 4));
                y{c,2} = imresize(I(f(x,2):f(x,2)+f(x,4), f(x,1):f(x,1)+f(x,3), :),[224, 224]);
                y{c,3} = i;
                c = c + 1;
            end
        end
    end
end