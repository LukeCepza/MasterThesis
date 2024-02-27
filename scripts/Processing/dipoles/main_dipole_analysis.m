dataPath    = 'D:\shared_git\MaestriaThesis\NeuroSenseDatabase';
type_of_pp  = 'pp01';
listStimuli = {'Air1','Air2','Air3','Air4',...
               'Vib1','Vib2','Vib3','Vib4',...
               'Car1','Car2','Car3','Car4'};



for type = 1:12
    tstimul = listStimuli{type};
    EEG = EEG_concatenator(dataPath,type_of_pp,tstimul);

    % Save dataset
    EEG.setname = ['allEEG_' , listStimuli];

    nameOutpp = fullfile('D:\shared_git\MaestriaThesis\Dipoles', ...
    ['allEEG_' , tstimul , '.set']);
    pop_saveset(EEG, 'filename', nameOutpp);

    EEG = pop_runica(EEG, 'icatype', 'runica', 'extended', 1, ...
    'lrate', 1e-5, 'maxsteps', 2000,'interrupt','off'); % 1300th iterations to converge.
    EEG = iclabel(EEG);
    EEG = pop_icflag(EEG, [NaN NaN;0.6 1;0.6 1;0.6 1;0.6 1;0.6 1;0.6 1]);
    
    EEG = pop_dipfit_settings( EEG, 'hdmfile', ...
    'D:\\NYNGroup\\eeglab2023.1\\plugins\\dipfit\\standard_BEM\\standard_vol.mat',...
    'mrifile','D:\\NYNGroup\\eeglab2023.1\\plugins\\dipfit\\standard_BEM\\standard_mri.mat',...
    'chanfile','D:\\NYNGroup\\eeglab2023.1\\plugins\\dipfit\\standard_BEM\\elec\\standard_1005.elc',...
    'coordformat','MNI','coord_transform','warpfiducials');
    EEG = pop_multifit(EEG, [] ,'threshold',10);
        
    pop_saveset(EEG, 'filename', nameOutpp);
end
%%
EEG = pop_loadset('filename','allEEG.set','filepath','D:\shared_git\MaestriaThesis\Dipoles');

EEG = pop_runica(EEG, 'icatype', 'runica', 'extended', 1, ...
'lrate', 1e-5, 'maxsteps', 2000,'interrupt','off'); % 1300th iterations to converge.
EEG = iclabel(EEG);
EEG = pop_icflag(EEG, [NaN NaN;0.6 1;0.6 1;0.6 1;0.6 1;0.6 1;0.6 1]);

pop_saveset(EEG, 'filename', 'D:\shared_git\MaestriaThesis\Dipoles\allEEG1.set');

EEG = pop_dipfit_settings( EEG, 'hdmfile', ...
'D:\\NYNGroup\\eeglab2023.1\\plugins\\dipfit\\standard_BEM\\standard_vol.mat',...
'mrifile','D:\\NYNGroup\\eeglab2023.1\\plugins\\dipfit\\standard_BEM\\standard_mri.mat',...
'chanfile','D:\\NYNGroup\\eeglab2023.1\\plugins\\dipfit\\standard_BEM\\elec\\standard_1005.elc',...
'coordformat','MNI','coord_transform','warpfiducials');
EEG = pop_multifit(EEG, [] ,'threshold',10);

pop_saveset(EEG, 'filename', 'D:\shared_git\MaestriaThesis\Dipoles\allEEG2.set');

function EEG = EEG_concatenator(dataPath,type_of_pp,tstimul)
    for id = 1:34
        sub_id = sprintf('sub-%02d', id);
        nameInE = [sub_id, '_' , type_of_pp , '_e', tstimul ,'.set'];
        nameInEPath = fullfile(dataPath, sub_id, type_of_pp, tstimul);
        
        if id == 1
            EEG = pop_loadset('filename',nameInE,'filepath',nameInEPath);
            chanlocs_original = EEG.chanlocs; %Necessary for interpolation
        else
            mEEG = pop_loadset('filename',nameInE,'filepath',nameInEPath);
            mEEG = pop_interp(mEEG, chanlocs_original, 'spherical');  
            EEG = pop_mergeset( EEG, mEEG, 0);
        end
    end
end
