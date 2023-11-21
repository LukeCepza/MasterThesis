[ALLEEG EEG CURRENTSET ALLCOM] = eeglab;
%
listStimuli = {'Air'}
tstimul = 'Car'
addpath("D:\NYNGroup\eeglab2023.1\")
dataPath = 'D:\shared_git\MaestriaThesis\data';
cells = cell(1,35);
idx = 0;
for i = 0:35
    id_str = sprintf('ID%02d', i);    
    file = fullfile('D:\shared_git\MaestriaThesis\data',id_str ,tstimul, ...
        [id_str ,'_pp_epochs',tstimul,'.set']);
    if exist(file, 'file') == 2 
        idx = idx + 1;
        cells{idx} = {'index',idx,'load',file,'subject',id_str,...
            'condition',tstimul,'session',1,'group','1'};
        disp("attaching to list " + id_str);
    else
        disp("Skipping " + file + " - file does not exist.");
    end
end
cells = cells(~cellfun('isempty', cells));%%
%
[STUDY ALLEEG] = std_editset( STUDY, [], 'name',tstimul,'task',tstimul,...
    'commands', cells,'updatedat','on','rmclust','on' );
[EEG ALLEEG CURRENTSET] = eeg_retrieve(ALLEEG,1);
[STUDY ALLEEG] = std_editset( STUDY, ALLEEG, 'updatedat','on','rmclust','on' );
[STUDY ALLEEG] = std_checkset(STUDY, ALLEEG);
CURRENTSTUDY = 1; EEG = ALLEEG; CURRENTSET = [1:length(EEG)];
[STUDY, ALLEEG] = std_precomp(STUDY, ALLEEG, {},'savetrials','on','recompute','on','erp','on');
eeglab redraw
%%
ERP = zeros(22,1000);
for i = 1:22
    channels = {'Fp1';'Fp2';'F3'; 'F4';'C3';'C4';'P3'; ...
        'P4';'O1';'O2';'F7';'F8';'T7';'T8';'P7'; ...
        'P8';'Fz';'Cz';'Pz';'AFz';'CPz'; 'POz'};
    [~,erp]=std_erpplot(STUDY,ALLEEG,'channels',channels(i), 'design', 1,'noplot','on');
    ERP(i,:) = mean(cell2mat(erp),2);
end


