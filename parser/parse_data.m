%% Parse all the data and put them into the appropriate folder.
%% Determine constants.
% If equals to 1, ignore FEAR and DISGUST.
type = 0;
% Base folders for databases.
base_jaffe = '';
base_ck = 'cohn-kanade';
base_kdef = '';
base_home = '';
% Target folder for each type.
target0 = 'data7';
target1 = 'data5';

%% Parse all datasets.
data = {};
data = [data; jaffe_parser(base_jaffe,type)];
disp('Jaffe OK!');
data = [data; cohnkanade_parser(base_ck,type)];
disp('CK+ OK!');
data = [data; kdef_parser(base_kdef,type)];
disp('KDEF OK!');
data = [data; homedata_parser(base_home,type)];
disp('HOME OK!');
disp('Writing...');
slice = size(data, 1)/100;

%% Write images to the target folder.
c = 0;
percent = 0;
if type == 1
    target = target1;
else
    target = target0;
end

fid_all = fopen(fullfile(target,'data_all','alabel.txt'), 'w');
fid_test = fopen(fullfile(target, 'data_noise_test','alabel.txt'), 'w');
fid_train = fopen(fullfile(target, 'data_noise_train','alabel.txt'), 'w');
for i=1:size(data,1)
    if i > c
        disp(strcat(num2str(percent), '%'));
        c = c + slice;
        percent = percent + 1;
    end
    % Write all the images to the data_all folder.
    imwrite(data{i,2}, fullfile(target,'data_all',data{i,1}));
    fprintf(fid_all, '%s %d\n', data{i,1}, data{i,3});
    
    % Add noise and seperate train and test images.
    if rand < 0.8
        isTrain = true;
        fid = fid_train;
        base = fullfile(target,'data_noise_train');
    else
        isTrain = false;
        fid = fid_test;
        base = fullfile(target,'data_noise_test');
    end
    imwrite(data{i,2}, fullfile(base, data{i,1}));
    if isTrain
        imwrite(imnoise(data{i,2}, 'salt & pepper'), fullfile(base, strcat(data{i,1}, '01sp.png')));
        imwrite(imnoise(data{i,2}, 'gaussian', 0, 0.05), fullfile(base, strcat(data{i,1}, '02g1.png')));
        imwrite(imnoise(data{i,2}, 'gaussian', 0, 0.1), fullfile(base, strcat(data{i,1}, '03g2.png')));
        imwrite(imnoise(data{i,2}, 'gaussian', 0.15, 0), fullfile(base, strcat(data{i,1}, '04g3.png')));
        imwrite(imnoise(data{i,2}, 'gaussian', 0.15, 0.05), fullfile(base, strcat(data{i,1}, '05g4.png')));
        imwrite(imnoise(data{i,2}, 'gaussian', 0.15, 0.1), fullfile(base, strcat(data{i,1}, '06g5.png')));
        imwrite(imnoise(data{i,2}, 'gaussian', 0.3, 0), fullfile(base, strcat(data{i,1}, '07g6.png')));
        imwrite(imnoise(data{i,2}, 'gaussian', 0.3, 0.05), fullfile(base, strcat(data{i,1}, '08g7.png')));
        imwrite(imnoise(data{i,2}, 'gaussian', 0.3, 0.1), fullfile(base, strcat(data{i,1}, '09g8.png')));
        imwrite(imnoise(data{i,2}, 'speckle', 0.2), fullfile(base, strcat(data{i,1}, '10s1.png')));
        imwrite(imnoise(data{i,2}, 'speckle', 0.4), fullfile(base, strcat(data{i,1}, '11s2.png')));
    end

    fprintf(fid, '%s %d\n', data{i,1}, data{i,3});
    if isTrain
        fprintf(fid, '%s %d\n', strcat(data{i,1}, '01sp.png'), data{i,3});
        fprintf(fid, '%s %d\n', strcat(data{i,1}, '02g1.png'), data{i,3});
        fprintf(fid, '%s %d\n', strcat(data{i,1}, '03g2.png'), data{i,3});
        fprintf(fid, '%s %d\n', strcat(data{i,1}, '04g3.png'), data{i,3});
        fprintf(fid, '%s %d\n', strcat(data{i,1}, '05g4.png'), data{i,3});
        fprintf(fid, '%s %d\n', strcat(data{i,1}, '06g5.png'), data{i,3});
        fprintf(fid, '%s %d\n', strcat(data{i,1}, '07g6.png'), data{i,3});
        fprintf(fid, '%s %d\n', strcat(data{i,1}, '08g7.png'), data{i,3});
        fprintf(fid, '%s %d\n', strcat(data{i,1}, '09g8.png'), data{i,3});
        fprintf(fid, '%s %d\n', strcat(data{i,1}, '10s1.png'), data{i,3});
        fprintf(fid, '%s %d\n', strcat(data{i,1}, '11s2.png'), data{i,3});
    end
end
fclose(fid_test);
fclose(fid_train);
fclose(fid_all);