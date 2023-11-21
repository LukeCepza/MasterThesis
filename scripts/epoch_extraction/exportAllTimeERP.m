addpath("D:\NYNGroup\eeglab2023.1\")
dataPath = 'D:\shared_git\MaestriaThesis\data';
%here we just need to change the name of the data.
listStimuli = {'Air1','Air2','Air3','Air4',...
               'Vib1','Vib2','Vib3','Vib4',...
               'Car1','Car2','Car3','Car4'};

channels = {'Fp1';'Fp2';'F3'; 'F4';'C3';'C4';'P3'; ...
'P4';'O1';'O2';'F7';'F8';'T7';'T8';'P7'; ...
'P8';'Fz';'Cz';'Pz';'AFz';'CPz'; 'POz'};

%channels = {'C3';'C4';'P3';'P4';'T7';'T8';'P7'; 'P8';'Cz';'Pz';'AFz';'CPz'};
typeOfPreprocessData = '_pp_epochs';%'_pp_epochs';
output_mat = zeros(1,1004);
for type = 1:12
    tstimul = listStimuli{type};
    for id = 13:50
        id_str = sprintf('ID%02d', id);    
        file = fullfile('D:\shared_git\MaestriaThesis\data',id_str,typeOfPreprocessData,tstimul, ...
        [id_str ,typeOfPreprocessData,tstimul,'.set']);
        try
            EEG = pop_loadset('filename',file);
            for chan = [5,6,18,21]%1:22
                try
                    EEG_temp = pop_select( EEG, 'channel',channels(chan));
                    epoch = squeeze(EEG_temp.data)';
                    rows_num = min(size(epoch));
                    data = ones(rows_num,1)*[chan,id,type];
                    output_mat = cat(1,output_mat, [data, [1:rows_num]', epoch]);
                catch
                    disp("Channel " + channels{chan}+" not found at"+ file)
                end
            end
        catch
            disp(file+ " not found")
        end
    end
end