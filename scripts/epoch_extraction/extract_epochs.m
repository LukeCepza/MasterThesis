function extract_epochs(ppnum, epochNamesPairs, dataPath, id_str, varargin)
    % Create an instance of the inputParser class
    p = inputParser;
    
    % Define the mandatory argument
    addRequired(p, 'ppnum', @ischar);
    addRequired(p, 'epochNamesPairs', @iscell);
    addRequired(p, 'dataPath', @ischar);
    addRequired(p, 'id_str', @ischar);
    addParameter(p, 'do_dipolefit', false, @islogical);
    addParameter(p,'rerunAMICA', false, @islogical)
    addParameter(p,'rerunInfoMaxICA', false, @islogical)

    % Parse the input arguments
    parse(p, ppnum, epochNamesPairs, dataPath, id_str, varargin{:});


    % Access the parsed values
    ppnum = p.Results.ppnum;
    dataPath = p.Results.dataPath;
    id_str = p.Results.id_str;
    epochNamesPairs = p.Results.epochNamesPairs;
    do_dipolefit = p.Results.do_dipolefit;
    rerunAMICA = p.Results.rerunAMICA;
    rerunInfoMaxICA = p.Results.rerunInfoMaxICA;

    EEG = pop_loadset('filename',['E_', id_str,'_', ppnum ,'.set' ],'filepath',fullfile(dataPath,id_str, ppnum));

    for idx = 1:length(epochNamesPairs)
        
        try
            rmdir(fullfile(dataPath, id_str, ppnum, epochNamesPairs{idx,2}), 's')
        catch
        end
        mkdir(fullfile(dataPath, id_str, ppnum, epochNamesPairs{idx,2}))

        EEGt = pop_epoch( EEG,epochNamesPairs{idx,1}, ...
            [-1,  3],  'epochinfo', 'yes');
        EEGt = pop_rmbase( EEGt, [-1000 0] ,[]);
        
        if rerunAMICA
            outdir = fullfile(dataPath, id_str, ppnum, epochNamesPairs{idx,2}, 'amicaouttmp');

            [weights,sphere,mods] = runamica15(EEGt.data, 'outdir',outdir);
            EEGt.etc.amicaResultStructure = mods;
            EEGt.icaweights = weights;
            EEGt.icasphere  = sphere;
            
            EEGt = iclabel(EEGt);
            EEGt = pop_icflag(EEGt, [NaN NaN;0.6 1;0.6 1;0.6 1;0.6 1;0.6 1;0.6 1]);
            %EEGt = pop_subcomp(EEGt,find(EEGt.reject.gcompreject), 0,0);
        end

        if rerunInfoMaxICA
            EEG = pop_runica(EEG, 'icatype', 'runica', 'extended', 1, ...
            'lrate', 1e-5, 'maxsteps', 2000,'interrupt','off'); % 1300th iterations to converge.
            EEGt = iclabel(EEGt);
            EEGt = pop_icflag(EEGt, [NaN NaN;0.6 1;0.6 1;0.6 1;0.6 1;0.6 1;0.6 1]);
        end

        if do_dipolefit
            EEGt = pop_dipfit_settings( EEGt, 'hdmfile', ...
                'D:\\NYNGroup\\eeglab2023.1\\plugins\\dipfit\\standard_BEM\\standard_vol.mat',...
                'mrifile','D:\\NYNGroup\\eeglab2023.1\\plugins\\dipfit\\standard_BEM\\standard_mri.mat',...
                'chanfile','D:\\NYNGroup\\eeglab2023.1\\plugins\\dipfit\\standard_BEM\\elec\\standard_1005.elc',...
                'coordformat','MNI','coord_transform','warpfiducials');
            EEGt = pop_multifit(EEGt, [] ,'threshold',10);
        end
        EEGt.setname = [id_str, epochNamesPairs{idx,3}];
        pop_saveset(EEGt, 'filename', fullfile(dataPath, id_str, ppnum, epochNamesPairs{idx,2},[EEGt.setname, '.set']));
    end
end
