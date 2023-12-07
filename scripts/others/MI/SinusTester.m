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

output = MI_matrix_tester(signalArray);

figure()
h = heatmap(output, ...
'XLabel', 'Hz', 'YLabel', 'Hz', ...
'Title', 'Mutual Information of sinosoidal signals');
    % Set the colormap to 'jet'
colormap(jet);

% Customize the axis labels
h.XDisplayLabels = string(1:22);
h.YDisplayLabels = string(1:22);
h.ColorLimits = [0 3];

function MIM = MI_matrix_tester(tester)

    MIM = nan(22,22);

    d1 = 0;
    for idx_1 = 1:22
        d1 = 1+d1;
        d2 = 0;
        for idx_2 = 1:22
            d2 = 1+d2;
            MIM(idx_1,idx_2) = mutual_information(squeeze(tester(d1,:)'),squeeze(tester(d2,:)')...
                ,'freedmanDiaconisRule', true);   
        end
    end
end