%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Generate PPvalidation .set data (preprocessed)
% CONFIGURATION VARIABLES
dataPath    = 'D:\shared_git\MaestriaThesis\NeuroSenseDatabase';
type_of_pp  = 'pp01';
listStimuli = {'Air','Air1','Air2','Air3','Air4',...
               'Vib','Vib1','Vib2','Vib3','Vib4',...
               'Car','Car1','Car2','Car3','Car4'};
channel     = {'Fp1';'Fp2';'F3'; 'F4';'C3';'C4';'P3'; ...
                'P4';'O1';'O2';'F7';'F8';'T7';'T8';'P7'; ...
                'P8';'Fz';'Cz';'Pz';'AFz';'CPz'; 'POz'};
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for type = 5
    tstimul = listStimuli{type};
    for id = 26
        sub_id = sprintf('sub-%02d', id);
        nameInEPath = [sub_id, '_' , type_of_pp , '_e', tstimul ,'.set'];
        nameInE = fullfile(dataPath, sub_id, type_of_pp, tstimul);
        EEG = pop_loadset('filename',nameInE,'filepath',nameInEPath);     
        data = reshape([EEG.data],EEG.nbchan,[],1);
        EEG.data = reshape(applyLaplacianReference(data,channel),EEG.nbchan,1250,[]);
    end
end