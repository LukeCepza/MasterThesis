addpath("D:\NYNGroup\eeglab2023.1\")
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Generate PP03 .set data (preprocessed)
% CONFIGURATION VARIABLES
type_of_pp  = 'pp03';
dataPath    = 'D:\shared_git\MaestriaThesis\data';
outPath    =  fullfile('D:\shared_git\MaestriaThesis\results', type_of_pp, 'eeglabStudy');
listStimuli = {'Air',...
                'Vib',...
                'Car'};
channels = {'Fp1';'Fp2';'F3'; 'F4';'C3';'C4';'P3'; ...
                'P4';'O1';'O2';'F7';'F8';'T7';'T8';'P7'; ...
                'P8';'Fz';'Cz';'Pz';'AFz';'CPz'; 'POz'};
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
MI = zeros(3,50,22,22);

tstimul1 = listStimuli{1};
tstimul2 = listStimuli{2};
tstimul3 = listStimuli{3};

for id = 13:50

    id_str = sprintf('ID%02d', id);
    nameInE1 = fullfile(dataPath,id_str, type_of_pp, tstimul1, [id_str,'_', type_of_pp , '_e',tstimul1,'.set' ]);
    nameInE2 = fullfile(dataPath,id_str, type_of_pp, tstimul2, [id_str,'_', type_of_pp , '_e',tstimul2,'.set' ]);
    nameInE3 = fullfile(dataPath,id_str, type_of_pp, tstimul3, [id_str,'_', type_of_pp , '_e',tstimul3,'.set' ]);

    try
        EEG1 = pop_loadset('filename',nameInE1);
        EEG1 = pop_subcomp(EEG1,find(EEG1.reject.gcompreject), 0,0);
        EEG2 = pop_loadset('filename',nameInE2);
        EEG2 = pop_subcomp(EEG2,find(EEG2.reject.gcompreject), 0,0);
        EEG3 = pop_loadset('filename',nameInE3);
        EEG3 = pop_subcomp(EEG3,find(EEG3.reject.gcompreject), 0,0);

        MI(:,id,:,:) = MI_3matrix(EEG1,EEG2,EEG3);
    catch
        disp(nameInE1+ " not found")
    end

end
%%
Dropped = mean(MI,[1,3,4],'omitnan') > 0;
MIM_all = squeeze(mean(MI(:,Dropped,:,:),2,"omitnan"));

f = figure(11); 
f.Name = 'MI several Plot'; 
f.Color = 'white'; 
pause(1); 
set(gcf, 'Position', [0 0 1500, 500]); % Set size
set(gcf, 'renderer', 'painters');

t = tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

%Air vs Vib
nexttile;
h = heatmap(squeeze(MIM_all(1, :, :)), ...
    'XLabel', 'Air', 'YLabel', 'Vib', ...
    'Title', 'Mutual Information Air vs Vib');

colormap(jet);

h.XDisplayLabels = string(channels);
h.YDisplayLabels = string(channels);
h.ColorLimits = [0.0013 0.0027];
%Vib vs Car
nexttile;
h = heatmap(squeeze(MIM_all(2, :, :)), ...
    'XLabel', 'Vib', 'YLabel', 'Car', ...
    'Title', 'Mutual Information Vib vs Car');

colormap(jet);

h.XDisplayLabels = string(channels);
h.YDisplayLabels = string(channels);
h.ColorLimits = [0.0013 0.0027];

%Car vs Air
nexttile;
h = heatmap(squeeze(MIM_all(3, :, :)), ...
    'XLabel', 'Car', 'YLabel', 'Air', ...
    'Title', 'Mutual Information Car vs Air');

colormap(jet);

h.XDisplayLabels = string(channels);
h.YDisplayLabels = string(channels);
h.ColorLimits = [0.0013 0.0027];

function MIM_triple = MI_3matrix_avgERP(EEG1,EEG2,EEG3)

   channels = {'Fp1';'Fp2';'F3'; 'F4';'C3';'C4';'P3'; ...
                    'P4';'O1';'O2';'F7';'F8';'T7';'T8';'P7'; ...
                    'P8';'Fz';'Cz';'Pz';'AFz';'CPz'; 'POz'};
    MIM_triple = nan(3,22,22);

    A = mean(EEG1.data,3);  
    B = mean(EEG2.data,3);  
    C = mean(EEG3.data,3);  

    d1 = 0;
    % Test A vs B.
    for idx_1 = 1:22
        if ismember(channels(idx_1),{EEG1.chanlocs.labels})
            d1 = 1+d1;
            d2 = 0;
            for idx_2 = 1:22
                if ismember(channels(idx_2),{EEG1.chanlocs.labels})
                    d2 = 1+d2;
                    MIM_triple(1,idx_1,idx_2) = mutual_information(A(d1,:),B(d2,:));   
                end
            end
        end
    end

    % Test B vs C.
    d1 = 0;
    for idx_1 = 1:22
        if ismember(channels(idx_1),{EEG2.chanlocs.labels})
            d1 = 1+d1;
            d2 = 0;
            for idx_2 = 1:22
                if ismember(channels(idx_2),{EEG2.chanlocs.labels})
                    d2 = 1+d2;
                    MIM_triple(2,idx_1,idx_2) = mutual_information(B(d1,:),C(d2,:));   
                end
            end
        end
    end

    % Test C vs. A
    d1 = 0;
    for idx_1 = 1:22
        if ismember(channels(idx_1),{EEG3.chanlocs.labels})
            d1 = 1+d1;
            d2 = 0;
            for idx_2 = 1:22
                if ismember(channels(idx_2),{EEG3.chanlocs.labels})
                    d2 = 1+d2;
                    MIM_triple(3,idx_1,idx_2) = mutual_information(C(d1,:),A(d2,:));   
                end
            end
        end
    end

end

function MIM_triple = MI_3matrix(EEG1,EEG2,EEG3)

   channels = {'Fp1';'Fp2';'F3'; 'F4';'C3';'C4';'P3'; ...
                    'P4';'O1';'O2';'F7';'F8';'T7';'T8';'P7'; ...
                    'P8';'Fz';'Cz';'Pz';'AFz';'CPz'; 'POz'};
    MIM_triple = nan(3,22,22);

    A = squeeze(reshape(EEG1.data,size(EEG1.data,1),[],1));  
    B = squeeze(reshape(EEG2.data,size(EEG2.data,1),[],1));  
    C = squeeze(reshape(EEG3.data,size(EEG3.data,1),[],1));  

    minlength = min([size(A,2),size(B,2),size(C,2)]);

    d1 = 0;
    % Test A vs B.
    for idx_1 = 1:22
        if ismember(channels(idx_1),{EEG1.chanlocs.labels})
            d1 = 1+d1;
            d2 = 0;
            for idx_2 = 1:22
                if ismember(channels(idx_2),{EEG1.chanlocs.labels})
                    d2 = 1+d2;
                    MIM_triple(1,idx_1,idx_2) = mutual_information(A(d1,1:minlength),B(d2,1:minlength),'freedmanDiaconisRule', true);                         
                    %MIM_triple(1,idx_1,idx_2) = mutual_information(A(d1,:),B(d2,:),'freedmanDiaconisRule', true);     
                end
            end
        end
    end

    % Test B vs C.
    d1 = 0;
    for idx_1 = 1:22
        if ismember(channels(idx_1),{EEG2.chanlocs.labels})
            d1 = 1+d1;
            d2 = 0;
            for idx_2 = 1:22
                if ismember(channels(idx_2),{EEG2.chanlocs.labels})
                    d2 = 1+d2;
                    MIM_triple(2,idx_1,idx_2) = mutual_information(B(d1,1:minlength),C(d2,1:minlength),'freedmanDiaconisRule', true);  
                    %MIM_triple(1,idx_1,idx_2) = mutual_information(B(d1,:),C(d2,:),'freedmanDiaconisRule', true);     
                
                end
            end
        end
    end

    % Test C vs. A
    d1 = 0;
    for idx_1 = 1:22
        if ismember(channels(idx_1),{EEG3.chanlocs.labels})
            d1 = 1+d1;
            d2 = 0;
            for idx_2 = 1:22
                if ismember(channels(idx_2),{EEG3.chanlocs.labels})
                    d2 = 1+d2;
                    MIM_triple(3,idx_1,idx_2) = mutual_information(C(d1,1:minlength),A(d2,1:minlength),'freedmanDiaconisRule', true);   
                    %MIM_triple(1,idx_1,idx_2) = mutual_information(C(d1,:),A(d2,:),'freedmanDiaconisRule', true);     

                end
            end
        end
    end

end