addpath("D:\NYNGroup\eeglab2023.1\")
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Generate PP03 .set data (preprocessed)
% CONFIGURATION VARIABLES
type_of_pp  = 'pp03';
dataPath    = 'D:\shared_git\MaestriaThesis\data';
outPath    =  fullfile('D:\shared_git\MaestriaThesis\results', type_of_pp, 'eeglabStudy');
listStimuli = {'Air1','Air2','Air3','Air4',...
                'Vib1','Vib2','Vib3','Vib4',...
                'Car1','Car2','Car3','Car4'};
channels = {'Fp1';'Fp2';'F3'; 'F4';'C3';'C4';'P3'; ...
                'P4';'O1';'O2';'F7';'F8';'T7';'T8';'P7'; ...
                'P8';'Fz';'Cz';'Pz';'AFz';'CPz'; 'POz'};
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
eeglab
output = zeros(12,50,22,22);
for type = 1:12
    tstimul = listStimuli{type};
    for id = 13:50
        id_str = sprintf('ID%02d', id);
        nameInE = fullfile(dataPath,id_str, type_of_pp,tstimul,[id_str,'_', type_of_pp , '_e',tstimul,'.set' ]);
        try
            EEG = pop_loadset('filename',nameInE);
            EEG = pop_subcomp(EEG,find(EEG.reject.gcompreject), 0,0);
            output(type,id,:,:) = MI_matrix(EEG);
        catch
            disp(nameInE+ " not found")
        end

    end
end
%%
Dropped = mean(output,[1,3,4],'omitnan') > 0;
MIM_all = squeeze(mean(output(:,Dropped,:,:),2,"omitnan"));

f = figure(10); 
f.Name = 'MI Plot'; 
f.Color = 'white'; 
pause(1); 
set(gcf, 'Position', [0 0 1500, 1500]); % Set size
set(gcf, 'renderer', 'painters');

t = tiledlayout(3, 4);
for i = 1:3
    for j = 1:4
        nexttile;
        h = heatmap(squeeze(MIM_all((i-1)*4+j,:,:)), ...
            'XLabel', 'Variables', 'YLabel', 'Variables', ...
            'Title', ['Mutual Information', listStimuli{(i-1)*4+j}]);
                % Set the colormap to 'jet'
        colormap(jet);

        % Customize the axis labels
        h.XDisplayLabels = string(channels);
        h.YDisplayLabels = string(channels);
        h.ColorLimits = [0.2 1.4];
    end
end

%% Tester
% Define parameters
numFrequencies = 22;
numSamples = 30000;
samplingFrequency = 250;
time = (0:numSamples-1) / samplingFrequency;

% Initialize the array
signalArray = zeros(numFrequencies, numSamples);

% Generate sinusoidal signals and fill the array
for freq = 1:numFrequencies
    frequency = freq; % Hz, you can modify this as needed
    amplitude = 1; % You can adjust the amplitude if needed
    phase = 0; % Phase offset in radians, modify as needed
    
    % Generate sinusoidal signal
    sinusoidalSignal = amplitude * sin(2 * pi * frequency * time + phase);
    
    % Store the signal in the array
    signalArray(freq, :) = sinusoidalSignal;
end

%% Functions
function MIM = MI_matrix(EEG)
    
    channels = {'Fp1';'Fp2';'F3'; 'F4';'C3';'C4';'P3'; ...
                    'P4';'O1';'O2';'F7';'F8';'T7';'T8';'P7'; ...
                    'P8';'Fz';'Cz';'Pz';'AFz';'CPz'; 'POz'};
    data = mean(EEG.data,3);  
    MIM = nan(22,22);

    d1 = 0;
    for idx_1 = 1:22
        if ismember(channels(idx_1),{EEG.chanlocs.labels})
            d1 = 1+d1;
            d2 = 0;
            for idx_2 = 1:22
                if ismember(channels(idx_2),{EEG.chanlocs.labels})
                    d2 = 1+d2;
                    MIM(idx_1,idx_2) = mutual_information(data(d1,:),data(d2,:));   
                end
            end
        end
    end

end
