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
    addParameter(p, 'basal_eeg', false, @islogical);
    addParameter(p, 'Bipolar_P3', false, @islogical);
    addParameter(p,'do_dwtEnergy',false, @islogical);
    addParameter(p,'do_dwtEnergy_tf',false, @islogical);

    % Parse the input arguments
    parse(p, varargin{:});

    % Access the parsed values
    nameInE = p.Results.nameInE;
    nameInEPath = p.Results.nameInEPath;
%channels = p.Results.channels;
    chan = p.Results.chan;
    do_dwtEnergy = p.Results.do_dwtEnergy;
    ICA_reject = p.Results.ICA_reject;
    LapReference = p.Results.LapReference;
    recomp_ICA = p.Results.recomp_ICA;
    basal_eeg = p.Results.basal_eeg;
    Bipolar_P3 = p.Results.Bipolar_P3;
    do_dwtEnergy_tf =p.Results.do_dwtEnergy_tf;

    EEG = pop_loadset('filename', nameInE, 'filepath', nameInEPath);    
    channels = {EEG.chanlocs.labels}';

    if LapReference
        data = reshape([EEG.data],EEG.nbchan,[],1);
        if basal_eeg
            EEG.data = applyLaplacianReference(EEG.data, channels);
        else
            EEG.data = reshape(applyLaplacianReference(data, channels), EEG.nbchan,1250,[]);
        end
    end

    if do_dwtEnergy
       EEG = dwtEnergy(EEG);
       return
    end

    if do_dwtEnergy_tf
       EEG = dwtEnergy_tf(EEG);
       return
    end

    if recomp_ICA
        EEG = iclabel(EEG);
        EEG = pop_icflag(EEG, [NaN NaN;0.6 1;0.6 1;0.6 1;0.6 1;0.6 1;0.6 1]);
    end

    if ICA_reject
        EEG = pop_subcomp(EEG,find(EEG.reject.gcompreject), 0,0);
    end

    if Bipolar_P3
        
       EEG_P3 = pop_select(EEG, 'channel', channels(7));
       EEG_C3 = pop_select(EEG, 'channel', channels(chan));
    end 

    EEG = pop_select(EEG, 'channel', channels(chan));

    if Bipolar_P3
        EEG.data = EEG_C3.data - EEG_P3.data;
    end 


end