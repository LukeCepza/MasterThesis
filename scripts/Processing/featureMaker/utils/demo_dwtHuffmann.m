% Generate a sample signal
x = sin(pi * (1:128) / 16) + 0.5 * randn(1, 128); % Sinusoidal signal with noise
epoch = 3;
x = EEG.data(1,:,epoch);
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

energy_X = norm(x, 2)^2;            
energy_Xr = norm(x_reconstructed, 2)^2;  

Energy_E = 100 * (energy_Xr / energy_X);

if Energy_E > 99
    disp('The reconstructed signal retains more than 99% of the original signal energy.');
else
    disp('The reconstructed signal retains less than 99% of the original signal energy.');
end

disp(['Energy of the reconstructed signal: ', num2str(Energy_E), '%']);

D1 = detcoef(C, L, 1);
D2 = detcoef(C, L, 2);
D3 = detcoef(C, L, 3);
D4 = detcoef(C, L, 4);

sigma_hat = median(abs(Di - median(D4))) / 0.6745;

N = length(D4); 
alpha = sigma_hat * sqrt(2 * log(N));

D1_thresh = D1 .* (abs(D1) >= alpha);
D2_thresh = D2 .* (abs(D2) >= alpha);
D3_thresh = D3 .* (abs(D3) >= alpha);
D4_thresh = D4 .* (abs(D4) >= alpha);



