addpath("D:\NYNGroup\eeglab2023.1\")
eeglab
dataPath = 'D:\shared_git\MaestriaThesis\data';
reps = zeros(99,20);len = zeros(99,1);chans = zeros(99,22);

for i = 13:50
    % Format 'i' with leading zeros (e.g., ID01, ID02, etc.)
    id_str = sprintf('ID%02d', i);
    
    nameInE = fullfile(dataPath, id_str, ['E_' id_str '_pp.set']);
    outCSV = fullfile(dataPath, id_str, 'allDataSummary.csv');
    
    if exist(nameInE, 'file') == 2 
        % input file exists
        disp("Summarizeing " + id_str);
        [reps(i,:), len(i), chans(i,:)] = summarizeEEG(nameInE);   
    else
        % file doesn't exist
        disp("Skipping " + nameInE + " - file does not exist.");
    end
end

colNames = {'id', 'size' ,'pn26','pn22','StopCar','End',...
    'CarI1','CarI2','CarI3','CarI4','AirI1','AirI2','AirI3',...
    'AirI4','VibI1','VibI2','VibI3','VibI4','VibStop','AirStop',...
    'boundary','CMD', 'Fp1','Fp2','F3', 'F4','C3','C4','P3','P4', ...
    'O1','O2','F7','F8','T7','T8','P7', 'P8','Fz','Cz','Pz', ...
    'AFz','CPz', 'POz'};
dataframe = [(1:99)',len,reps,chans];
T = array2table(dataframe);
T.Properties.VariableNames = colNames;
channels = {};