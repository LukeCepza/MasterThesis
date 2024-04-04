% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %% Generate pp01 .set data (preprocessed) eval1
% % CONFIGURATION VARIABLES
% dataPath     = 'D:\shared_git\MaestriaThesis\NeuroSenseDatabase';
% savePathcwt = 'D:\shared_git\MaestriaThesis\cwt_imags';
% type_of_pp   = 'pp01';
% ICA_reject   = true;
% recomp_ICA   = false;
% LapReference = false;
% interpolate = true;
% do_extract_cw_plot = true;
% listStimuli  = {'Air1','Air2','Air3','Air4',...
%                'Vib1','Vib2','Vib3','Vib4',...
%                'Car1','Car2','Car3','Car4'};
% channels     = {'Fp1';'Fp2';'F3'; 'F4';'C3';'C4';'P3'; ...
%                 'P4';'O1';'O2';'F7';'F8';'T7';'T8';'P7'; ...
%                 'P8';'Fz';'Cz';'Pz';'AFz';'CPz'; 'POz'};
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% output_mat = zeros(1,22+4);
% imag = 0;
% Class_nums = zeros(8000,1);
% for type = 1:12
%     tstimul = listStimuli{type};
%     for id = 1:34
%         sub_id = sprintf('sub-%02d', id);
%         nameInE = [sub_id, '_' , type_of_pp , '_e', tstimul ,'.set'];
%         nameInEPath = fullfile(dataPath, sub_id, type_of_pp, tstimul);
%         for chan = 5
%             EEG = image_feature_maker_process( nameInE,        ...
%                     nameInEPath, ...
%                     chan, ...
%                     'interpolate', interpolate, ...
%                     'ICA_reject',ICA_reject,              ...
%                     'recomp_ICA',recomp_ICA, ...
%                     'do_extract_cw_plot',do_extract_cw_plot);
%             epoch = uint16(EEG.cw_plot); 
%             for c_epoch = 1:size(epoch,1)
%                 imag = imag + 1;
%                 cwt_im = squeeze(epoch(c_epoch,:,:));
%                 imwrite(cwt_im, fullfile(savePathcwt, 'cwt_png',['cwt' num2str(imag) '.png']), bitdepth = 16)
%                 imwrite(cwt_im, fullfile(savePathcwt,'cwt_tiff', ['cwt' num2str(imag) '.tiff']))
%                 Class_nums(imag) = type; 
%             end
%         end
%     end
% end
% Class_nums = Class_nums(Class_nums == 0);
% writematrix(Class_nums, fullfile(savePathcwt,[type_of_pp, 'imagesClass.csv']));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Generate pp01 .set data (preprocessed) eval2
% CONFIGURATION VARIABLES
dataPath     = 'D:\shared_git\MaestriaThesis\NeuroSenseDatabase';
savePathmat = 'D:\shared_git\MaestriaThesis\mat';
type_of_pp   = 'pp01';
ICA_reject   = true;
recomp_ICA   = false;
LapReference = false;
interpolate = true;
do_extract_cw_plot = false;
listStimuli  = {'Air1','Air2','Air3','Air4',...
               'Vib1','Vib2','Vib3','Vib4',...
               'Car1','Car2','Car3','Car4'};
channels     = {'Fp1';'Fp2';'F3'; 'F4';'C3';'C4';'P3'; ...
                'P4';'O1';'O2';'F7';'F8';'T7';'T8';'P7'; ...
                'P8';'Fz';'Cz';'Pz';'AFz';'CPz'; 'POz'};
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
output_mat = zeros(1,22,1250);
imag = 0;
Class_nums = 0;
for type = 1:12
    tstimul = listStimuli{type};
    for id = 1:34
        sub_id = sprintf('sub-%02d', id);
        nameInE = [sub_id, '_' , type_of_pp , '_e', tstimul ,'.set'];
        nameInEPath = fullfile(dataPath, sub_id, type_of_pp, tstimul);
        for chan = 5
            EEG = image_feature_maker_process( nameInE,        ...
                    nameInEPath, ...
                    chan, ...
                    'interpolate', interpolate, ...
                    'ICA_reject',ICA_reject,              ...
                    'recomp_ICA',recomp_ICA, ...
                    'do_extract_cw_plot',do_extract_cw_plot);
            epoch = EEG.data;
            rows_num = min(size(epoch));
            epoch = permute(epoch, [3,1,2]);
            Class_nums = cat(1,Class_nums, ones(rows_num,1)*type);
            output_mat = cat(1,output_mat, epoch);
       end
    end
end
ERPs = output_mat(2:end,:,:);
classes = Class_nums(2:end);
save("ERPs.mat","ERPs")
save("classes.mat","classes")
%writematrix(Class_nums, fullfile(savePathcwt,[type_of_pp, 'imagesClass.csv']));