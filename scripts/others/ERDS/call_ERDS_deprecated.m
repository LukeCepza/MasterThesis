%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Generate PP03 .set data (preprocessed)
% CONFIGURATION VARIABLES
dataPath    = 'D:\shared_git\MaestriaThesis\data';
type_of_pp  = 'pp03';
listStimuli = {'Air','Air1','Air2','Air3','Air4',...
               'Vib','Vib1','Vib2','Vib3','Vib4',...
               'Car','Car1','Car2','Car3','Car4'};
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Corregir canales faltantes, rellenar con Nans

ERDS = []
t = linspace(-1000,2996,500);                 
tic
for type = 10
    tstimul = listStimuli{type};
    for i = 13:20
        id_str = sprintf('ID%02d', i);    
        file = fullfile(dataPath,id_str , type_of_pp, tstimul, ...
            [id_str, '_' ,type_of_pp, '_e',tstimul,'.set']);
        if exist(file, 'file') == 2 
            %[ERD,f] = indivCWT(dataPath,id_str,type_of_pp,tstimul);
            iERDS = indivERDS(dataPath,id_str,type_of_pp,tstimul);
            ERDS = cat(4,ERDS, iERDS);
        else
            disp("Skipping " + file + " - file does not exist.");
        end
    end
    gERDS = groupERDS(ERDS,type_of_pp,tstimul,type,t,f);
end
%permute(ERDS,[1,3,2])
toc % 30 mins to process with parfor


function gERDS = groupERDS(ERDS, type_of_pp,tstimul,fignum,t,f)

    gERDS = squeeze(mean(ERDS, 4));
    figure(fignum+10)   
    plotERSPFull_TL('Sos',gERDS,fignum, ...
    'ylim', [1, 46] , 'clim', [-500, 100],'ts',t,'freq',f);
    saveas(fignum+10, fullfile('D:\shared_git\MaestriaThesis\results',...
    type_of_pp,'ERC','PlotERDS',[tstimul,'.jpg']));% Functions
    clf(fignum+10)
end

function ERDS = indivERDS(varargin)

    % Create an instance of the inputParser class
    p = inputParser;

    % Define the mandatory arguments
    addRequired(p, 'dataPath', @ischar);
    addRequired(p, 'id_str', @ischar);
    addRequired(p, 'type_of_pp', @ischar);
    addRequired(p, 'tstimul', @ischar);

    % Define optional arguments with default values
    addParameter(p, 'BaseLine_otherSignal', '', @ischar);

    % Parse the input arguments
    parse(p, varargin{:});

    % Access the parsed values
    dataPath = p.Results.dataPath;
    id_str = p.Results.id_str;
    type_of_pp = p.Results.type_of_pp;
    BaseLine_otherSignal = p.Results.BaseLine_otherSignal;
    tstimul = p.Results.tstimul;
    
    ERD = {};
    load(fullfile(dataPath, id_str, type_of_pp, tstimul, 'cwt.mat'), 'ERD');

    % for ch = 1:22
    %     all_ccwt = ERD{ch}
    %     for epoch = size(ERD,3)
    %         ccwt = squeeze(all_ccwt(epoch,:,:))
    %         baseline =  ccwt(:,1:125);
    %         ERDS = ccwt - mean(baseline, 2)
    %     end
    % end

    ERDS = zeros(22,61,500);

    for ch = 1:22
        all_ccwt = (ERD{ch}).^2;
        baseline = all_ccwt(:,1:125,:);
        epochERDS = (mean(baseline, 2) - all_ccwt)./mean(baseline, 2)*100;
        ERDS(ch,:,:) = squeeze(mean(epochERDS,3));    
    end

    % if ~isempty(BaseLine_otherSignal) 
    %     load(BaseLine_otherSignal)
    % else
    % 
    % end

    if ~isempty(BaseLine_otherSignal) 
        save(fullfile(dataPath,id_str,type_of_pp,'ERDS.mat'),'ERDS');
    end
end
 
function [ERD,f] = indivCWT(varargin)

    eeglab;

    % Create an instance of the inputParser class
    p = inputParser;

    % Define the mandatory arguments
    addRequired(p, 'dataPath', @ischar);
    addRequired(p, 'id_str', @ischar);
    addRequired(p, 'type_of_pp', @ischar);
    addRequired(p, 'tstimul', @ischar);

    % Define optional arguments with default values
    addParameter(p, 'BaseLine_otherSignal', false, @islogical);

    % Parse the input arguments
    parse(p, varargin{:});

    % Access the parsed values
    dataPath = p.Results.dataPath;
    id_str = p.Results.id_str;
    type_of_pp = p.Results.type_of_pp;
    tstimul = p.Results.tstimul;

    do_BaseLine_otherSignal = p.Results.BaseLine_otherSignal;
  
    chans = {'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', ...
        'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'AFz', 'CPz', 'POz'};
    filepath = fullfile(dataPath,id_str , type_of_pp, tstimul);
    filename =  [id_str, '_' ,type_of_pp, '_e',tstimul,'.set'];
    EEG = pop_loadset('filename',filename ,'filepath',filepath);

    EEG = pop_resample( EEG, 125);
    [~,f] = cwt(squeeze(EEG.data(1,:,1)),'amor', EEG.srate);

    wt_val = cell(22,1);
    epoch_len = size(EEG.data,3);
    wt_all = zeros(length(f),length(EEG.times),epoch_len);

    for ch = 1:22
        try
            EEGch = pop_select( EEG, 'channel',chans(ch));
            for epoch = 1:epoch_len
                wt_all(:,:,epoch) = abs(cwt(squeeze(EEGch.data(1,:,epoch)),'amor', EEGch.srate));
            end
        catch
            wt_val{ch} = NaN;
            disp(['channel', chans{ch} ,' not found'])
        end
        wt_val{ch} = wt_all
    end
    ERD = wt_val
    save( fullfile(filepath,'cwt.mat'),'ERD');
end
% 
% function [CWT,f] = BasalCWT(varargin)
% 
%     eeglab;
% 
%     % Create an instance of the inputParser class
%     p = inputParser;
% 
%     % Define the mandatory arguments
%     addRequired(p, 'dataPath', @ischar);
%     addRequired(p, 'id_str', @ischar);
%     addRequired(p, 'type_of_pp', @ischar);
%     addRequired(p, 'tstimul', @ischar);
% 
%     % Define optional arguments with default values
%     addParameter(p, 'BaseLine_otherSignal', false, @islogical);
% 
%     % Parse the input arguments
%     parse(p, varargin{:});
% 
%     % Access the parsed values
%     dataPath = p.Results.dataPath;
%     id_str = p.Results.id_str;
%     type_of_pp = p.Results.type_of_pp;
%     tstimul = p.Results.tstimul;
% 
%     do_BaseLine_otherSignal = p.Results.BaseLine_otherSignal;
% 
%     chans = {'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', ...
%         'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'AFz', 'CPz', 'POz'};
%     filepath = fullfile(dataPath,id_str , type_of_pp, tstimul);
%     filename =  ['B_', id_str, '_' , type_of_pp , '.set'];
%     EEG = pop_loadset('filename',filename ,'filepath',filepath);
%     EEG = pop_subcomp(EEG,find(EEG.reject.gcompreject), 0,0);
% 
%     wt_val = cell(22,1);
%     for ch = 1:22
%         try
%             EEGch = pop_select( EEG, 'channel',chans(ch));
%             for epoch = 1:size(EEGch.data,3)
%                 [wt_val{ch},f] = cwt(squeeze(EEGch.data(1,:,epoch)),'amor', EEGch.srate);
%             end
%         catch
%             wt_val{ch} = NaN;
%             disp(['channel', chans{ch} ,' not found'])
%         end
%     end
%     ERD = wt_val
%     save( fullfile(filepath,'cwt.mat'),'ERD');
% end
% 
% function ERD = globalERD()
% 
% end