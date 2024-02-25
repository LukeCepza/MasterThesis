addpath("D:\NYNGroup\eeglab2023.1\");
clear;
rng default 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Generate PPValidation .set data (preprocessed)
% CONFIGURATION VARIABLES
type_of_pp  = 'pp_validation';
dataPath    = 'D:\shared_git\MaestriaThesis\NeuroSenseDatabase';
ChanLocsBesa = 'D:\NYNGroup\eeglab2023\plugins\dipfit\standard_BESA\standard-10-5-cap385.elp';
PE          = false;                    
ICA_reject  = false;              
ASR         = true;                     
Cleanline   = false;             
Filt60Hz    = true;                
Interpolate = false; 
GlobalRef   = true;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic
parfor id = 1:34
    % Format 'i' with leading zeros (e.g., sub-01, sub-02, etc.)
    sub_id = sprintf('sub-%02d', id);
    nameInE = fullfile(dataPath, sub_id, 'eeg', [sub_id '_E.edf']);
    nameOutpp = fullfile(dataPath, sub_id, type_of_pp, [sub_id, '_E_' , type_of_pp , '.set']);
    
    mkdir(fullfile(dataPath, sub_id, type_of_pp))
    if exist(nameInE, 'file') == 2     
        % input file exists
        disp("Processing " + sub_id);
        preproEEG( dataPath,...
            nameInE,        ...
            nameOutpp,           ...
            sub_id, ...
            type_of_pp,...
            ChanLocsBesa, ...
            'PE', PE,                    ...
            'ICA_reject',ICA_reject,              ...
            'ASR',ASR,                     ...
            'Cleanline', Cleanline,             ...
            'Filt60Hz', Filt60Hz,               ...
            'Interpolate',Interpolate,            ...
            'GlobalRef',GlobalRef);
    else
        % file doesn't exist
        disp("Skipping " + nameInE + " - file does not exist.");
    end
end
 
%% Preprocessing of basal data
parfor id = 1:34
    % Format 'i' with leading zeros (e.g., sub-01, sub-02, etc.)
    sub_id = sprintf('sub-%02d', id);
    nameInE = fullfile(dataPath, sub_id, 'eeg', [sub_id '_B.edf']);
    nameOutpp = fullfile(dataPath, sub_id, type_of_pp, [sub_id, '_B_' , type_of_pp , '.set']);
    if exist(nameInE, 'file') == 2 
        % input file exists
        disp("Processing " + sub_id);
        preproEEG( dataPath,...
            nameInE,        ...
            nameOutpp,           ...
            sub_id, ...
            type_of_pp,...
            ChanLocsBesa, ...
            'PE', PE,                    ...
            'ICA_reject',ICA_reject,              ...
            'ASR',ASR,                     ...
            'Cleanline', Cleanline,             ...
            'Filt60Hz', Filt60Hz,               ...
            'Interpolate',Interpolate,            ...
            'GlobalRef',GlobalRef);
    else
        % file doesn't exist
        disp("Skipping " + nameInE + " - file does not exist.");
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Generate pp01 .set data (preprocessed)
% CONFIGURATION VARIABLES
type_of_pp  = 'pp01';
dataPath    = 'D:\shared_git\MaestriaThesis\NeuroSenseDatabase';
ChanLocsBesa = 'D:\NYNGroup\eeglab2023\plugins\dipfit\standard_BESA\standard-10-5-cap385.elp';
PE          = false;                    
ICA_reject  = false;              
ASR         = true;                     
Cleanline   = false;             
Filt60Hz    = true;                
Interpolate = false; 
GlobalRef   = true;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic
parfor id = 1:34
    % Format 'i' with leading zeros (e.g., sub-01, sub-02, etc.)
    sub_id = sprintf('sub-%02d', id);
    nameInE = fullfile(dataPath, sub_id, 'eeg', [sub_id '_E.edf']);
    nameOutpp = fullfile(dataPath, sub_id, type_of_pp, [sub_id, '_E_' , type_of_pp , '.set']);
    
    mkdir(fullfile(dataPath, sub_id, type_of_pp))
    if exist(nameInE, 'file') == 2     
        % input file exists
        disp("Processing " + sub_id);
        preproEEG( dataPath,...
            nameInE,        ...
            nameOutpp,           ...
            sub_id, ...
            type_of_pp,...
            ChanLocsBesa, ...
            'PE', PE,                    ...
            'ICA_reject',ICA_reject,              ...
            'ASR',ASR,                     ...
            'Cleanline', Cleanline,             ...
            'Filt60Hz', Filt60Hz,               ...
            'Interpolate',Interpolate,            ...
            'GlobalRef',GlobalRef);
    else
        % file doesn't exist
        disp("Skipping " + nameInE + " - file does not exist.");
    end
end
 
%% Preprocessing of basal data
parfor id = 1:34
    % Format 'i' with leading zeros (e.g., sub-01, sub-02, etc.)
    sub_id = sprintf('sub-%02d', id);
    nameInE = fullfile(dataPath, sub_id, 'eeg', [sub_id '_B.edf']);
    nameOutpp = fullfile(dataPath, sub_id, type_of_pp, [sub_id, '_B_' , type_of_pp , '.set']);
    if exist(nameInE, 'file') == 2 
        % input file exists
        disp("Processing " + sub_id);
        preproEEG( dataPath,...
            nameInE,        ...
            nameOutpp,           ...
            sub_id, ...
            type_of_pp,...
            ChanLocsBesa, ...
            'PE', PE,                    ...
            'ICA_reject',ICA_reject,              ...
            'ASR',ASR,                     ...
            'Cleanline', Cleanline,             ...
            'Filt60Hz', Filt60Hz,               ...
            'Interpolate',Interpolate,            ...
            'GlobalRef',GlobalRef);
    else
        % file doesn't exist
        disp("Skipping " + nameInE + " - file does not exist.");
    end
end
toc