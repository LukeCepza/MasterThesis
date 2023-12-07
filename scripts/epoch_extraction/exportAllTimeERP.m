addpath("D:\NYNGroup\eeglab2023.1\")
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Generate PP03 .set data (preprocessed)
% CONFIGURATION VARIABLES
type_of_pp  = 'pp03';
dataPath    = 'D:\shared_git\MaestriaThesis\data';
outPath    =  fullfile('D:\shared_git\MaestriaThesis\results', type_of_pp, 'eeglabStudy');
listStimuli = {'Air1','Air2','Air3','Air4',...
                'Vib1','Vib2','Vib3','Vib4',...
                'Car1','Car2','Car3','Car4'};
channels = {'Fp1';'Fp2';'F3'; 'F4';'C3';'C4';'P3'; ...
                'P4';'O1';'O2';'F7';'F8';'T7';'T8';'P7'; ...
                'P8';'Fz';'Cz';'Pz';'AFz';'CPz'; 'POz'};
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
eeglab

output_mat = zeros(1,1004);
for type = 1:12
    tstimul = listStimuli{type};
    for id = 13:47
        id_str = sprintf('ID%02d', id);
        nameInE = fullfile(dataPath,id_str, type_of_pp,tstimul,[id_str,'_', type_of_pp , '_e',tstimul,'.set' ]);
        try
            EEG = pop_loadset('filename',nameInE);
            for chan = [5,6,18,21]%1:22
                try
                    EEG_temp = pop_select( EEG, 'channel',channels(chan));
                    epoch = squeeze(EEG_temp.data)';
                    rows_num = min(size(epoch));
                    data = ones(rows_num,1)*[chan,id,type];
                    output_mat = cat(1,output_mat, [data, [1:rows_num]', epoch]);
                catch
                    disp("Channel " + channels{chan}+" not found at"+ nameInE)
                end
            end
        catch
            disp(nameInE+ " not found")
        end
    end
end

writematrix(output_mat, 'output_mat.csv');