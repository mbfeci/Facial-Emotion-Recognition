%%%% LABELS %%%%
% ANGRY    = 1 %
% HAPPY    = 2 %
% SAD      = 3 %
% SURPRISE = 4 %
% NEUTRAL  = 5 %
% FEAR     = 6 %
% DISGUST  = 7 %
%%%%%%%%%%%%%%%%

%% Parser for KDEF database
% Returns a cell array in which each row contains [Name, Image, Label].
%   type: if 1, ignore DISGUST and FEAR.
function y = kdef_parser(root, type)
    if (~exist('type', 'var'))
        type = 0;
    end
    y = {};
    c = 1;
    [~,names] = xlsread(fullfile(root,'KDEF','DATAMATRIXKDEF.xls'), 'A:A');
    [~,emotion] = xlsread(fullfile(root,'KDEF','DATAMATRIXKDEF.xls'), 'B:B');
    for i=2:size(emotion,1)
        if getLabel(emotion{i}) == -1
            continue;
        end
        if type == 1
            % Ignore FEAR and DISGUST.
            if getLabel(emotion{i}) > 5
                continue;
            end
        end
        % Read the image
        filename = fullfile(root,'KDEF',strcat(names{i},'.jpg'));
        if ~exist(filename, 'file')
            continue;
        end
        I = imread(filename);
        y{c,1} = strcat('kdef_',names{i},'.png');
        y{c,2} = imresize(I,[224, 224]);
        y{c,3} = getLabel(emotion{i});
        c = c + 1;
    end
end

function y = getLabel(emotion)
    if isequal(emotion, 'Angry')
        y = 1;
    elseif isequal(emotion, 'Happy')
        y = 2;
    elseif isequal(emotion, 'Sad')
        y = 3;
    elseif isequal(emotion, 'Surprised')
        y = 4;
    elseif isequal(emotion, 'Neutral')
        y = 5;
    elseif isequal(emotion, 'Fearful')
        y = 6;
    elseif isequal(emotion, 'Disgusted')
        y = 7;
    else
        y = -1;
    end
end