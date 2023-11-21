function extract_epochs(ppnum, epochNamesPairs, dataPath, id_str, varargin)
    % Create an instance of the inputParser class
    p = inputParser;

    % Define the mandatory argument
    addRequired(p, 'ppnum', @ischar);
    addRequired(p, 'epochNamesPairs', @iscell);
    addRequired(p, 'dataPath', @ischar);
    addRequired(p, 'id_str', @ischar);
    addParameter(p, 'do_dipolefit', false, @islogical);

    % Parse the input arguments
    parse(p, ppnum, epochNamesPairs, dataPath, id_str, varargin{:});


    % Access the parsed values
    ppnum = p.Results.ppnum;
    dataPath = p.Results.dataPath;
    id_str = p.Results.id_str;
    epochNamesPairs = p.Results.epochNamesPairs;
    do_dipolefit = p.Results.do_dipolefit;

    EEG = pop_loadset('filename',['E_', id_str,'_', ppnum ,'.set' ],'filepath',fullfile(dataPath,id_str, ppnum));

    for idx = 1:length(epochNamesPairs)
        EEGt = pop_epoch( EEG,epochNamesPairs{idx,1}, ...
            [-1,  3],  'epochinfo', 'yes');
        EEGt = pop_rmbase( EEGt, [-1000 0] ,[]);
        if do_dipolefit
            EEGt = pop_dipfit_settings( EEGt, 'hdmfile', ...
                'D:\\NYNGroup\\eeglab2023.1\\plugins\\dipfit\\standard_BEM\\standard_vol.mat',...
                'mrifile','D:\\NYNGroup\\eeglab2023.1\\plugins\\dipfit\\standard_BEM\\standard_mri.mat',...
                'chanfile','D:\\NYNGroup\\eeglab2023.1\\plugins\\dipfit\\standard_BEM\\elec\\standard_1005.elc',...
                'coordformat','MNI','coord_transform','warpfiducials');
            EEGt = pop_multifit(EEGt, [] ,'threshold',10);
        end
        EEGt.setname = [id_str, epochNamesPairs{idx,3}];
        try
            rmdir(fullfile(dataPath, id_str, ppnum, epochNamesPairs{idx,2}), 's')
        catch
        end
        mkdir(fullfile(dataPath, id_str, ppnum, epochNamesPairs{idx,2}))
        pop_saveset(EEGt, 'filename', fullfile(dataPath, id_str, ppnum, epochNamesPairs{idx,2},[EEGt.setname, '.set']));
    end
end
% Deprecated version
% function extract_epochs(EEG,dataPath, id_str)
% 
%     EEGt = pop_epoch( EEG, {  '33028'  '33029'  '33030'  '33031'  }, ...
%         [-1  3],  'epochinfo', 'yes');
%     EEGt = pop_rmbase( EEGt, [-1000 0] ,[]);
%     EEGt.setname = [id_str, '_pp_epochsAir'];
%     mkdir(fullfile(dataPath, id_str,"Air"))
%     pop_saveset(EEGt, 'filename', fullfile(dataPath, id_str, 'Air',[EEGt.setname, '.set']));
% 
%     EEGt = pop_epoch( EEG, {  '33028' }, ...
%         [-1  3],  'epochinfo', 'yes');
%     EEGt = pop_rmbase( EEGt, [-1000 0] ,[]);
%     EEGt.setname = [id_str, '_pp_epochsAir1'];
%     mkdir(fullfile(dataPath, id_str,"Air1"))
%     pop_saveset(EEGt, 'filename',  fullfile(dataPath, id_str,'Air1', [EEGt.setname, '.set']));
% 
%     EEGt = pop_epoch( EEG, {  '33029' }, ...
%     [-1  3],  'epochinfo', 'yes');
%     EEGt = pop_rmbase( EEGt, [-1000 0] ,[]);
%     mkdir(fullfile(dataPath, id_str,"Air2"))
% 
%     EEGt.setname = [id_str, '_pp_epochsAir2'];
%     pop_saveset(EEGt, 'filename',  fullfile(dataPath, id_str, 'Air2',[EEGt.setname, '.set']));
% 
%     EEGt = pop_epoch( EEG, {  '33030' }, ...
%         [-1  3],  'epochinfo', 'yes');
%     EEGt = pop_rmbase( EEGt, [-1000 0] ,[]);
%     mkdir(fullfile(dataPath, id_str,"Air3"))
%     EEGt.setname = [id_str, '_pp_epochsAir3'];
%     pop_saveset(EEGt, 'filename',  fullfile(dataPath, id_str, 'Air3',[EEGt.setname, '.set']));
% 
%     EEGt = pop_epoch( EEG, {  '33031' }, ...
%         [-1  3],  'epochinfo', 'yes');
%     EEGt = pop_rmbase( EEGt, [-1000 0] ,[]);
%     mkdir(fullfile(dataPath, id_str,"Air4"))
%     EEGt.setname = [id_str, '_pp_epochsAir4'];
%     pop_saveset(EEGt, 'filename',  fullfile(dataPath, id_str,'Air4', [EEGt.setname, '.set']));
% 
%     %car
%     EEGt = pop_epoch( EEG, {  '33024'  '33025'  '33026'  '33027'  }, ...
%         [-1  3], 'epochinfo', 'yes');
%     EEGt = pop_rmbase( EEGt, [-1000 0] ,[]);
%         mkdir(fullfile(dataPath, id_str,"Car"))
% 
%     EEGt.setname = [id_str, '_pp_epochsCar'];
%     pop_saveset(EEGt, 'filename',  fullfile(dataPath, id_str, 'Car',[EEGt.setname, '.set']));
% 
%     EEGt = pop_epoch( EEG, {  '33024' }, ...
%         [-1  3],  'epochinfo', 'yes');
%     EEGt = pop_rmbase( EEGt, [-1000 0] ,[]);
%             mkdir(fullfile(dataPath, id_str,"Car1"))
% 
%     EEGt.setname = [id_str, '_pp_epochsCar1'];
%     pop_saveset(EEGt, 'filename',  fullfile(dataPath, id_str, 'Car1',[EEGt.setname, '.set']));
% 
%     EEGt = pop_epoch( EEG, {  '33025' }, ...
%         [-1  3],  'epochinfo', 'yes');
%     EEGt = pop_rmbase( EEGt, [-1000 0] ,[]);
%                 mkdir(fullfile(dataPath, id_str,"Car2"));
% 
%     EEGt.setname = [id_str, '_pp_epochsCar2'];
%     pop_saveset(EEGt, 'filename', fullfile(dataPath, id_str, 'Car2',[EEGt.setname, '.set']));
% 
%     EEGt = pop_epoch( EEG, {  '33026' }, ...
%         [-1  3],  'epochinfo', 'yes');
%     EEGt = pop_rmbase( EEGt, [-1000 0] ,[]);
%     EEGt.setname = [id_str, '_pp_epochsCar3'];           
%     mkdir(fullfile(dataPath, id_str,"Car3"));
% 
%     pop_saveset(EEGt, 'filename',  fullfile(dataPath, id_str, 'Car3',[EEGt.setname, '.set']));
% 
%     EEGt = pop_epoch( EEG, {  '33027' }, ...
%         [-1  3],  'epochinfo', 'yes');
%     EEGt = pop_rmbase( EEGt, [-1000 0] ,[]);
%         mkdir(fullfile(dataPath, id_str,"Car4"));
%     EEGt.setname = [id_str, '_pp_epochsCar4'];
%     pop_saveset(EEGt, 'filename',  fullfile(dataPath, id_str, 'Car4',[EEGt.setname, '.set']));
% 
%     %vib
%     EEGt = pop_epoch( EEG, {  '33032'  '33033'  '33034'  '33035'  }, ...
%         [-1  3], 'epochinfo', 'yes');
%     EEGt = pop_rmbase( EEGt, [-1000 0] ,[]);
%             mkdir(fullfile(dataPath, id_str,"Vib"));
% 
%     EEGt.setname = [id_str, '_pp_epochsVib'];
%     pop_saveset(EEGt, 'filename',  fullfile(dataPath, id_str, 'Vib',[EEGt.setname, '.set']));
% 
%     EEGt = pop_epoch( EEG, {  '33032' }, ...
%         [-1  3],  'epochinfo', 'yes');
%     EEGt = pop_rmbase( EEGt, [-1000 0] ,[]);           
%     mkdir(fullfile(dataPath, id_str,"Vib1"));
% 
%     EEGt.setname = [id_str, '_pp_epochsVib1'];
%     pop_saveset(EEGt, 'filename',  fullfile(dataPath, id_str, 'Vib1',[EEGt.setname, '.set']));
% 
%     EEGt = pop_epoch( EEG, {  '33033' }, ...
%         [-1  3],  'epochinfo', 'yes');
%     EEGt = pop_rmbase( EEGt, [-1000 0] ,[]);
%         mkdir(fullfile(dataPath, id_str,"Vib2"));
% 
%     EEGt.setname = [id_str, '_pp_epochsVib2'];
%     pop_saveset(EEGt, 'filename',  fullfile(dataPath, id_str, 'Vib2',[EEGt.setname, '.set']));
% 
%     EEGt = pop_epoch( EEG, {  '33034' }, ...
%         [-1  3],  'epochinfo', 'yes');
%     EEGt = pop_rmbase( EEGt, [-1000 0] ,[]);   
%     mkdir(fullfile(dataPath, id_str,"Vib3"));
% 
%     EEGt.setname = [id_str, '_pp_epochsVib3'];
%     pop_saveset(EEGt, 'filename',  fullfile(dataPath, id_str, 'Vib3',[EEGt.setname, '.set']));
% 
%     EEGt = pop_epoch( EEG, {  '33035' }, ...
%         [-1  3],  'epochinfo', 'yes');
%     EEGt = pop_rmbase( EEGt, [-1000 0] ,[]);
%         mkdir(fullfile(dataPath, id_str,"Vib4"));
% 
%     EEGt.setname = [id_str, '_pp_epochsVib4'];
%     pop_saveset(EEGt, 'filename',  fullfile(dataPath, id_str, 'Vib4',[EEGt.setname, '.set']));
% end