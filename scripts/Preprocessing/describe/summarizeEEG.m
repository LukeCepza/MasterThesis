%% Luis Kevin Cepeda Zapata 22/10/2023
function [reps, len,chans] = summarizeEEG(pathIn)

    dataSet = pop_loadset('filename', pathIn);
% uniqueStrings = {'1998','1999','200','OVTK_StimulationId_ExperimentStop',...
%     'OVTK_StimulationId_Label_00','OVTK_StimulationId_Label_01',...
%     'OVTK_StimulationId_Label_02','OVTK_StimulationId_Label_03',...
% 	'OVTK_StimulationId_Label_04','OVTK_StimulationId_Label_05',...
% 	'OVTK_StimulationId_Label_06','OVTK_StimulationId_Label_07',...
% 	'OVTK_StimulationId_Label_08','OVTK_StimulationId_Label_09',...
% 	'OVTK_StimulationId_Label_0A','OVTK_StimulationId_Label_0B',...
% 	'OVTK_StimulationId_Label_12','OVTK_StimulationId_Label_13',...
% 	'boundary'	'condition 19'};

uniqueStrings = {'1998','1999','200','OVTK_StimulationId_ExperimentStop',...
'33024','33025','33026','33027','33028','33029','33030','33031',...
'33032','33033','33034','33035', '33042','33043','boundary','condition 19'};
    % Initialize a cell array to store unique strings and their counts
    uniqueStrings = unique(uniqueStrings);
    counts = cell(size(uniqueStrings));

    % Loop through the unique strings and count their occurrences
    for i = 1:numel(uniqueStrings)
        counts{i} = sum(strcmp({dataSet.event.type}, uniqueStrings{i}));
    end   
    % Display the unique strings and their respective counts
    for i = 1:numel(uniqueStrings)
        fprintf('%s appears %d times\n', uniqueStrings{i}, counts{i});
    end
    
    reps = cell2mat(counts);
    len = length(dataSet.times);

    channelsList = {'Fp1','Fp2','F3', 'F4','C3','C4','P3','P4', ...
    'O1','O2','F7','F8','T7','T8','P7', 'P8','Fz','Cz','Pz', ...
    'AFz','CPz', 'POz'};

    uniqueStrings = unique(channelsList);
    counts = cell(size(channelsList));

        % Loop through the unique strings and count their occurrences
    for i = 1:numel(uniqueStrings)
        counts{i} = sum(strcmp({dataSet.chanlocs.labels}, uniqueStrings{i}));
    end   
    % Display the unique strings and their respective counts
    for i = 1:numel(uniqueStrings)
        fprintf('%s appears %d times\n', uniqueStrings{i}, counts{i});
    end

    chans = cell2mat(counts);
end




