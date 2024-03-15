% Level    Low Frequency (Hz)    High Frequency (Hz)
% D1        62.500000           125.000000
% D2        31.250000           62.500000
% D3        15.625000           31.250000
% D5        3.906250           7.812500
% A1        0 (DC)              3.906250
samplingRate = 200; % Replace with your actual EEG sampling rate
waveletName = 'bior1.3';
level = 8; % Number of levels in the wavelet decomposition

frequencyBands = zeros(level, 2); % Initialize matrix to hold frequency bands for each level

for i = 1:level
    frequencyBands(i, 1) = samplingRate / (2^(i+1)); % f_low
    frequencyBands(i, 2) = samplingRate / (2^i); % f_high
end

% Display the frequency ranges
disp('Level    Low Frequency (Hz)    High Frequency (Hz)');
for i = 1:level
    fprintf('D%d        %f           %f\n', i, frequencyBands(i, 1), frequencyBands(i, 2));
end

fprintf('A1        0 (DC)              %f\n', frequencyBands(i, 1));