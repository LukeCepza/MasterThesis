addpath("D:\NYNGroup\eeglab2023.1\")
dataPath = 'D:\shared_git\MaestriaThesis\data';
%here we just need to change the name of the data.
listStimuli = {'Air','Air1','Air2','Air3','Air4',...
               'Vib','Vib1','Vib2','Vib3','Vib4',...
               'Car','Car1','Car2','Car3','Car4'};
typeOfPreprocessData = '_pp_epochs';%'_pp_epochs';
figure(5)
for type = 1:15
    tstimul = listStimuli{type};
    StudyInfo = getStudyInfo(tstimul,typeOfPreprocessData);
    [locs,ERP] = getERPandDIPFromStudy(StudyInfo,tstimul);
    save(['D:\shared_git\MaestriaThesis\results\ERPs_EEGLABSTUDY\', tstimul,'.mat'], 'ERP');
    save(['D:\shared_git\MaestriaThesis\results\EEGLAB_STUDY\', tstimul, '_locs.mat'], 'locs');
    %plotERP_single_loc(tstimul,ERP,{[1 0 0]},0.9,5)
    %saveas(gcf, ['D:\shared_git\MaestriaThesis\results\ERPs_EEGLABSTUDY\',tstimul,'.jpg']);% Functions
    clf(5)
end

function StudyInfo = getStudyInfo(tstimul,typeOfPreprocessData) 
% This function returns a cell of folders containing the .set data for
% given condition
    cells = cell(1,35);
    idx = 0;
    for i = 13:50
        id_str = sprintf('ID%02d', i);    
        file = fullfile('D:\shared_git\MaestriaThesis\data',id_str ,tstimul, ...
            [id_str ,typeOfPreprocessData,tstimul,'.set']);
        if exist(file, 'file') == 2 
            idx = idx + 1;
            cells{idx} = {'index',idx,'load',file,'subject',id_str,...
                'condition',tstimul,'session',1,'group','1'};
            disp("attaching to list " + id_str);
        else
            disp("Skipping " + file + " - file does not exist.");
        end
    end
    StudyInfo = cells(~cellfun('isempty', cells));%%
end

function [locs,ERP] = getERPandDIPFromStudy(StudyInfo,tstimul)
    %Based on the Study information, a new study is generated and the ERP
    % for the given coclear
    % ndition is calculated. 
    % The ERP is returned.
    numclust  = 15;
    [ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab; 
    [STUDY, ALLEEG] = std_editset( [], [], 'name',tstimul,'task',tstimul,...
        'commands', StudyInfo,'updatedat','on','rmclust','on' );
    [EEG, ALLEEG, CURRENTSET] = eeg_retrieve(ALLEEG,1);
    [STUDY, ALLEEG] = std_editset( STUDY, ALLEEG, 'updatedat','on','rmclust','on' );
    [STUDY, ALLEEG] = std_checkset(STUDY, ALLEEG);
    CURRENTSTUDY = 1; EEG = ALLEEG; CURRENTSET = 1:length(EEG);
    [STUDY ALLEEG] = std_editset( STUDY, ALLEEG, 'commands',{{'inbrain','on','dipselect',0.15}},'updatedat','on','rmclust','on' );
    [STUDY ALLEEG] = std_checkset(STUDY, ALLEEG);
    [STUDY, ALLEEG] = std_precomp(STUDY, ALLEEG, {},'savetrials','on','recompute','on','erp','on');
    [STUDY ALLEEG] = std_preclust(STUDY, ALLEEG, 1,{'dipoles','weight',1});
    [STUDY] = pop_clust(STUDY, ALLEEG, 'algorithm','kmeans','clus_num',  numclust  );
    STUDY = std_dipplot(STUDY,ALLEEG,'clusters','all');
    saveas(gcf, ['D:\shared_git\MaestriaThesis\results\EEGLAB_STUDY\',tstimul,'_dipoles.jpg']);
    clf(5)

    
    locs = zeros(numclust+1,3);
    for clust = 2:15+1
        locs(clust,:) = STUDY.cluster(clust).dipole.posxyz;
    end

    [STUDY EEG] = pop_savestudy( STUDY, EEG, 'filename', [tstimul , '.study'] ,...
      'filepath','D:\shared_git\MaestriaThesis\results\EEGLAB_STUDY\');

    ERP = zeros(22,1000);

    for i = 1:22
        channels = {'Fp1';'Fp2';'F3'; 'F4';'C3';'C4';'P3'; ...
            'P4';'O1';'O2';'F7';'F8';'T7';'T8';'P7'; ...
            'P8';'Fz';'Cz';'Pz';'AFz';'CPz'; 'POz'};
        [~,erp]=std_erpplot(STUDY,ALLEEG,'channels',channels(i), 'design', 1,'noplot','on');
        ERP(i,:) = mean(cell2mat(erp),2);
    end
end

function plotERP_single_loc(tstimul, ERP_list,figColor,LineW,fig_n)
     ts = -1000:1/250*1000:2999;
    
    f = figure(fig_n); f.Name = 'ERP Plot'; 
    f.Color ='white'; pause(1); f.Position; 
    set(gcf, 'Position', [0 0 1500, 700]); %<- Set size
    set(gcf, 'renderer', 'painters');

    Labels = {'FP1','FP2','F3','F4','C3','C4','P3','P4','O1','O2','F7','F8',...
    'T7','T8','P7','P8','Fz','Cz','Pz','AFz','CPz','POz'};
    Labels = convertCharsToStrings(Labels);%GEt Labels
    
    %Matrix of Mbrain CAp chan locations for subplot locations
    subplotchloc = ["FP1","2";"FP2","4";"F3","7";"F4","9";"C3","12";...
        "C4","14";"P3","22";"P4","24";"O1","27";"O2","29";"F7","6";...
        "F8","10";"T7","11";"T8","15";"P7","21";"P8","25";"Fz","8";...
        "Cz","13";"Pz","23";"AFz","3";"CPz","18";"POz","28"];
    ch = 0;
    for chpltloc = Labels %Plot per chanel
        ch = ch + 1;
        Gav = ERP_list(ch,:);

        locplt = str2double(subplotchloc(find(chpltloc == subplotchloc),2));
        subplot(6,5,locplt)
        %Plot ERP
        hold on
        plot(ts,Gav,'LineWidth',LineW,'Color',cell2mat(figColor))
        set(gca,'Xtick',[-200 , 0 , 400],'Ytick', [-4 0 6])
        set(gca,'FontUnits','points','FontName','Sans','FontSize',10)
        axis([-1000 2000 -3 3])
        line([0 0], [0 2],'Color',[0.1 0.1 0.1],'LineWidth', 0.8);
        yline(0, 'LineWidth', 1,'Color',[0.1 0.1 0.1],'Alpha', 0.4);
        title(chpltloc)
        
        ax = gca;
        ax.XAxisLocation = 'origin';
        ax.YAxisLocation = 'origin';
        set(gca,'XColor','none','YColor','none','TickDir','in')
    end
    subplot(6,5,1)
    set(gca,'XColor','none','YColor','none','TickDir','in')
    set(gca,'FontUnits','points','FontName','Sans','FontSize',10)
    text(0.3,0.5, [tstimul,' ERP'])
end

