addpath("D:\NYNGroup\eeglab2023.1\")
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Generate pp01 .set data (preprocessed) eval1
% CONFIGURATION VARIABLES
dataPath    = 'D:\shared_git\MaestriaThesis\NeuroSenseDatabase';
type_of_pp  = 'pp01';
ICA_reject  = true;
recomp_ICA  = true;
LapReference= true;
listStimuli = {'Air1','Air2','Air3','Air4',...
               'Vib1','Vib2','Vib3','Vib4',...
               'Car1','Car2','Car3','Car4'};
channels    = {'Fp1';'Fp2';'F3'; 'F4';'C3';'C4';'P3'; ...
                'P4';'O1';'O2';'F7';'F8';'T7';'T8';'P7'; ...
                'P8';'Fz';'Cz';'Pz';'AFz';'CPz'; 'POz'};
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

output_mat = zeros(1,1254);
for type = 1:12
    tstimul = listStimuli{type};
    for id = 1:34
        sub_id = sprintf('sub-%02d', id);
        nameInE = [sub_id, '_' , type_of_pp , '_e', tstimul ,'.set'];
        nameInEPath = fullfile(dataPath, sub_id, type_of_pp, tstimul);

        for chan = 5
            try
                EEG = feature_maker_process( nameInE,        ...
                        nameInEPath, ...
                        chan, ...
                        'ICA_reject',ICA_reject,              ...
                        'recomp_ICA',recomp_ICA,              ...
                        'LapReference',LapReference);

                epoch = squeeze(EEG.data)';
                rows_num = min(size(epoch));
                data = ones(rows_num,1)*[chan,id,type];
                output_mat = cat(1,output_mat, [data, [1:rows_num]', epoch]);
            catch
                disp("Channel " + channels{chan} + " not found at" + nameInE)
            end
        end
    end
end
%%
for id = 1:34
    sub_id = sprintf('sub-%02d', id);
    nameInE = [sub_id, '_E_' , type_of_pp ,'.set'];
    nameInEPath = fullfile(dataPath, sub_id, type_of_pp);

    for chan = 5
        try
            EEG = feature_maker_process( nameInE,        ...
                    nameInEPath, ...
                    chan, ...
                    'ICA_reject',ICA_reject,              ...
                    'recomp_ICA',recomp_ICA,              ...
                    'LapReference',LapReference);

            epoch = squeeze(EEG.data)';
            rows_num = min(size(epoch));
            data = ones(rows_num,1)*[chan,id,13];
            output_mat = cat(1,output_mat, [data, [1:rows_num]', epoch]);
        catch
            disp("Channel " + channels{chan} + " not found at" + nameInE)
        end
    end
end
%% Export data
%###% Export times
%output_mat = output_mat(2:end,:);
writematrix(output_mat, fullfile('D:\shared_git\MaestriaThesis\FeaturesTabs',[type_of_pp, '_t1.csv']));

%###% PSD
[pxx,f] = pwelch(output_mat(:,5:end)', 500,250,[], 250);
output_pxx = [output_mat(:,1:4),pxx'] 
writematrix(output_mat, fullfile('D:\shared_git\MaestriaThesis\FeaturesTabs',[type_of_pp, '_t2.csv']));
