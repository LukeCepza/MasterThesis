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

    if LapReference
        channels = {EEG.chanlocs.labels}';
        % Laplacian reference
        data = reshape([EEG.data],EEG.nbchan,[],1);
        EEG.data = reshape(applyLaplacianReference(data, channels), EEG.nbchan,1250,[]);
    end

    if recomp_ICA
        EEG = iclabel(EEG);
        EEG = pop_icflag(EEG, [NaN NaN;0.6 1;0.6 1;0.6 1;0.6 1;0.6 1;0.6 1]);
    end

    if ICA_reject
        EEG = pop_subcomp(EEG,find(EEG.reject.gcompreject), 0,0);
    end

    if is_basal
       EEG = pop_subcomp(EEG,find(EEG.reject.gcompreject), 0,0);

   
    EEG = pop_select(EEG, 'channel', channels(chan));
end