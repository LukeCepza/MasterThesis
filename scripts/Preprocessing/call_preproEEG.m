addpath("D:\NYNGroup\eeglab2023.1\");
clear;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Generate PP01 .set data (preprocessed)
% CONFIGURATION VARIABLES
type_of_pp  = 'pp01';
dataPath    = 'D:\shared_git\MaestriaThesis\data';
PE          = false;                    
ICA_reject  = true;              
ASR         = true;                     
Cleanline   = false;             
Filt60Hz    = true;                
Interpolate = false; 
GlobalRef   = false;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic
parfor i = 12:50
    % Format 'i' with leading zeros (e.g., ID01, ID02, etc.)
    id_str = sprintf('ID%02d', i);

    nameInE = fullfile(dataPath, id_str, ['E_' id_str '.edf']);
    nameOutpp = fullfile(dataPath, id_str, type_of_pp, ['E_', id_str, '_' , type_of_pp , '.set']);
    mkdir(fullfile(dataPath, id_str, type_of_pp))
    if exist(nameInE, 'file') == 2 
        
        % input file exists
        disp("Processing " + id_str);
        preproEEG( nameInE,        ...
            nameOutpp,           ...
            id_str, ...
            type_of_pp,...
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
%%
% Generate PP .set data of basal data (preprocessed)
parfor i = 12:50
    % Format 'i' with leading zeros (e.g., ID01, ID02, etc.)
    id_str = sprintf('ID%02d', i);

    nameInE = fullfile(dataPath, id_str, ['B_' id_str '.edf']);
    nameOutpp = fullfile(dataPath, id_str, type_of_pp, ['B_', id_str, '_' , type_of_pp , '.set']);
    mkdir(fullfile(dataPath, id_str, type_of_pp))
    if exist(nameInE, 'file') == 2 
        % input file exists
        disp("Processing " + id_str);
        preproEEG( nameInE,        ...
            nameOutpp,           ...
            id_str, ...
            type_of_pp,...
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Generate pp02 .set data (perceived epoch preprocessed) 
% CONFIGURATION VARIABLES
type_of_pp  = 'pp02';
dataPath    = 'D:\shared_git\MaestriaThesis\data';
PE          = true;                    
ICA_reject  = true;              
ASR         = true;                     
Cleanline   = false;             
Filt60Hz    = true;               
Interpolate = false; 
GlobalRef   = false;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[~,~,~] = eeglab;
parfor i = 12:50
    id_str = sprintf('ID%02d', i);
    
    nameInE = fullfile(dataPath, id_str, ['E_' id_str '.edf']);
    nameOutpp = fullfile(dataPath, id_str, type_of_pp, ['E_', id_str, '_' , type_of_pp , '.set']);
    try
        rmdir(fullfile(dataPath, id_str, type_of_pp), 's')
    catch
    end
    mkdir(fullfile(dataPath, id_str, type_of_pp))
    if exist(nameInE, 'file') == 2 
        % input file exists
        disp("Processing " + id_str);
        preproEEG( nameInE,        ...
            nameOutpp,           ...
            id_str, ...
            type_of_pp,...
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Generate PP03 .set data (preprocessed)
% CONFIGURATION VARIABLES
type_of_pp  = 'pp03';
dataPath    = 'D:\shared_git\MaestriaThesis\data';
PE          = false;                    
ICA_reject  = false;              
ASR         = true;                     
Cleanline   = false;             
Filt60Hz    = true;                
Interpolate = false; 
GlobalRef   = true;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic
parfor i = 12:50
    % Format 'i' with leading zeros (e.g., ID01, ID02, etc.)
    id_str = sprintf('ID%02d', i);

    nameInE = fullfile(dataPath, id_str, ['E_' id_str '.edf']);
    nameOutpp = fullfile(dataPath, id_str, type_of_pp, ['E_', id_str, '_' , type_of_pp , '.set']);
    mkdir(fullfile(dataPath, id_str, type_of_pp))
    if exist(nameInE, 'file') == 2 
        
        % input file exists
        disp("Processing " + id_str);
        preproEEG( nameInE,        ...
            nameOutpp,           ...
            id_str, ...
            type_of_pp,...
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

%% Generate PP .set data of basal data (preprocessed)
% Generate PP03 .set data (preprocessed)
% CONFIGURATION VARIABLES
type_of_pp  = 'pp03';
dataPath    = 'D:\shared_git\MaestriaThesis\data';
PE          = false;                    
ICA_reject  = false;              
ASR         = true;                     
Cleanline   = false;             
Filt60Hz    = true;                
Interpolate = false; 
GlobalRef   = true;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic
parfor i = 12:50
    % Format 'i' with leading zeros (e.g., ID01, ID02, etc.)
    id_str = sprintf('ID%02d', i);

    nameInE = fullfile(dataPath, id_str, ['B_' id_str '.edf']);
    nameOutpp = fullfile(dataPath, id_str, type_of_pp, ['B_', id_str, '_' , type_of_pp , '.set']);
    mkdir(fullfile(dataPath, id_str, type_of_pp))
    if exist(nameInE, 'file') == 2 
        % input file exists
        disp("Processing " + id_str);
        preproEEG( nameInE,        ...
            nameOutpp,           ...
            id_str, ...
            type_of_pp,...
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