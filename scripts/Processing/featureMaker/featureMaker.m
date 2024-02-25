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


writematrix(output_mat, 'output_mat.csv');

function EEG = feature_maker_process(varargin)
    eeglab;

    % Create an instance of the inputParser class
    p = inputParser;

    % Define the mandatory arguments
    addRequired(p, 'nameInE', @ischar);
    addRequired(p, 'nameInEPath', @ischar);
%addRequired(p, 'channels', @iscell);
    addRequired(p, 'chan', @isnumeric);

    % Define optional arguments with default values
    addParameter(p, 'ICA_reject',  true, @islogical);
    addParameter(p, 'recomp_ICA',  true, @islogical);
    addParameter(p, 'LapReference', false, @islogical);

    % Parse the input arguments
    parse(p, varargin{:});

    % Access the parsed values
    nameInE = p.Results.nameInE;
    nameInEPath = p.Results.nameInEPath;
%channels = p.Results.channels;
    chan = p.Results.chan;

    ICA_reject = p.Results.ICA_reject;
    LapReference = p.Results.LapReference;
    recomp_ICA = p.Results.recomp_ICA;

    EEG = pop_loadset('filename', nameInE, 'filepath', nameInEPath);    

    if recomp_ICA
        EEG = iclabel(EEG);
        EEG = pop_icflag(EEG, [NaN NaN;0.6 1;0.6 1;0.6 1;0.6 1;0.6 1;0.6 1]);
    end

    if ICA_reject
        EEG = pop_subcomp(EEG,find(EEG.reject.gcompreject), 0,0);
    end

    if LapReference
        channels = {EEG.chanlocs.labels}';
        % Laplacian reference
        data = reshape([EEG.data],EEG.nbchan,[],1);
        EEG.data = reshape(applyLaplacianReference(data, channels), EEG.nbchan,1250,[]);
    end
    
    EEG = pop_select(EEG, 'channel', channels(chan));
end