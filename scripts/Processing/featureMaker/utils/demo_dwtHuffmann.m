% Generate a sample signal
x = sin(pi * (1:128) / 16) + 0.5 * randn(1, 128); % Sinusoidal signal with noise
epoch = 3
x = signal(:,epoch)
level = 4; % Level of decomposition
waveletName = 'bior3.3'; % Wavelet type

[C, L] = wavedec(x, level, waveletName); % Decomposition

subplot(6, 2, 1);
plot(x);

subplot(6, 2, 2);
plot(x);
title('Original Signal');
subplot(6, 2, 3);
plot(C);
subplot(6, 2, 4);
plot(C);
title('Original Signal');

for i = 1:level
    % Approximation coefficients at level i
    subplot(6, 2, 2+2 * i+1);
    Ai = appcoef(C, L, waveletName, i);
    plot(Ai);    
    xlim([-inf inf])

    title(['Approximation Coefficients at Level ', num2str(i)]);
    
    % Detail coefficients at level i
    subplot(6, 2, 2+ 2 * i + 2);
    Di = detcoef(C, L, i);

    plot(Di);
    xlim([-inf inf])
    title(['Detail Coefficients at Level ', num2str(i)]);
end


x_reconstructed = waverec(C, L, waveletName);
%%
% Assuming x and x_reconstructed are defined
energy_X = norm(x, 2)^2;            % Energy of the original signal
energy_Xr = norm(x_reconstructed, 2)^2;  % Energy of the reconstructed signal

% Calculate the percentage of energy retained in the reconstructed signal
Energy_E = 100 * (energy_Xr / energy_X);

% Check if the energy is greater than 99%
if Energy_E > 99
    disp('The reconstructed signal retains more than 99% of the original signal energy.');
else
    disp('The reconstructed signal retains less than 99% of the original signal energy.');
end

% Display the energy percentage
disp(['Energy of the reconstructed signal: ', num2str(Energy_E), '%']);

%%
% Assuming you have already computed the wavelet decomposition
D1 = detcoef(C, L, 1);
D2 = detcoef(C, L, 2);
D3 = detcoef(C, L, 3);
D4 = detcoef(C, L, 4);

% Compute the Median Absolute Deviation (MAD) of the finest scale detail coefficients
sigma_hat = median(abs(Di - median(D4))) / 0.6745;

% Calculate the threshold value alpha
N = length(D4); % Number of coefficients at the last level
alpha = sigma_hat * sqrt(2 * log(N));

% Threshold the detail coefficients at all levels
D1_thresh = D1 .* (abs(D1) >= alpha);
D2_thresh = D2 .* (abs(D2) >= alpha);
D3_thresh = D3 .* (abs(D3) >= alpha);
D4_thresh = D4 .* (abs(D4) >= alpha);

% Reconstruct the signal using the thresholded coefficients
% You would need to combine the thresholded detail coefficients
% with the approximation coefficients A4 for reconstruction
% This is not shown here as it requires the full decomposition vector
% and bookkeeping matrix L which were obtained during wavedec.
% The reconstruction would use waverec or idwt functions.

