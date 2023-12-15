addpath("D:\NYNGroup\eeglab2023.1\")
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Generate PP03 .set data (preprocessed)
% CONFIGURATION VARIABLES
type_of_pp  = 'pp03';
dataPath    = 'D:\shared_git\MaestriaThesis\data';
outPath    =  fullfile('D:\shared_git\MaestriaThesis\results', type_of_pp, 'eeglabStudy');
listStimuli = {'Air',...
                'Vib',...
                'Car'};
channels = {'Fp1';'Fp2';'F3'; 'F4';'C3';'C4';'P3'; ...
                'P4';'O1';'O2';'F7';'F8';'T7';'T8';'P7'; ...
                'P8';'Fz';'Cz';'Pz';'AFz';'CPz'; 'POz'};
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
MI = zeros(3,50,22,22);
MIa = zeros(3,50,22,22);
MIb = zeros(3,50,22,22);

tstimul1 = listStimuli{1};
tstimul2 = listStimuli{2};
tstimul3 = listStimuli{3};

for id = 13:50

    id_str = sprintf('ID%02d', id);
    nameInE1 = fullfile(dataPath,id_str, type_of_pp, tstimul1, [id_str,'_', type_of_pp , '_e',tstimul1,'.set' ]);
    nameInE2 = fullfile(dataPath,id_str, type_of_pp, tstimul2, [id_str,'_', type_of_pp , '_e',tstimul2,'.set' ]);
    nameInE3 = fullfile(dataPath,id_str, type_of_pp, tstimul3, [id_str,'_', type_of_pp , '_e',tstimul3,'.set' ]);
    nameInEB = fullfile(dataPath,id_str, type_of_pp,  ['B_', id_str , '_',type_of_pp,'.set' ]);
    try
        EEG1 = pop_loadset('filename',nameInE1);
        EEG1 = pop_subcomp(EEG1,find(EEG1.reject.gcompreject), 0,0);
        EEG2 = pop_loadset('filename',nameInE2);
        EEG2 = pop_subcomp(EEG2,find(EEG2.reject.gcompreject), 0,0);
        EEG3 = pop_loadset('filename',nameInE3);
        EEG3 = pop_subcomp(EEG3,find(EEG3.reject.gcompreject), 0,0);
        EEGB = pop_loadset('filename',nameInEB);
        EEGB = pop_subcomp(EEGB,find(EEGB.reject.gcompreject), 0,0);
        MIa(:,id,:,:) = MI_3matrix_avgERP(EEG1,EEG2,EEG3);
        MI(:,id,:,:) = MI_3matrix(EEG1,EEG2,EEG3);
        MIb(:,id,:,:) = MI_3matrix_basal(EEG1,EEG2,EEG3,EEGB);

    catch
        disp(nameInE1+ " not found")
   end
end
%%
plot_MIacross(MI, 'ColorLimits', [0 3], 'fig_n', 10, 'Title', 'Spontaneus')
plot_MIacross(MIa, 'ColorLimits', [0 3], 'fig_n', 11, 'Title', 'Averaged')
plot_MIacross(MIb, 'ColorLimits', [0 3], 'fig_n', 12, 'Title', 'Agains Basal')
