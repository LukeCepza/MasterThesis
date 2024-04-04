function EEG = image_feature_maker_process(varargin)
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
    addParameter(p, 'basal_eeg', false, @islogical);
    addParameter(p,'do_extract_cw_plot',false, @islogical);
    addParameter(p,'Interpolate', false, @islogical)
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
    basal_eeg = p.Results.basal_eeg;
    Interpolate = p.Results.Interpolate;
    do_extract_cw_plot = p.Results.do_extract_cw_plot;

    EEG = pop_loadset('filename', nameInE, 'filepath', nameInEPath);    
    channels = {EEG.chanlocs.labels}';

    if Interpolate
        load('D:\shared_git\MasterThesis\scripts\Processing\featureMaker\utils\chanlocs_original.mat', 'chanlocs_original')
        EEG = pop_interp(EEG, chanlocs_original, 'spherical');  
        channels = {EEG.chanlocs.labels}';
    end
    
    if recomp_ICA
        EEG = iclabel(EEG);
        EEG = pop_icflag(EEG, [NaN NaN;0.6 1;0.6 1;0.6 1;0.6 1;0.6 1;0.6 1]);
    end

    if LapReference
        data = reshape([EEG.data],EEG.nbchan,[],1);
        if basal_eeg
            EEG.data = applyLaplacianReference(EEG.data, channels);
        else
            EEG.data = reshape(applyLaplacianReference(data, channels), EEG.nbchan,1250,[]);
        end
    end

    if do_extract_cw_plot
        if ICA_reject
            EEG = pop_subcomp(EEG,find(EEG.reject.gcompreject), 0,0);
        end
        EEG = extract_cw_plot(EEG, chan);
        return

    end
end