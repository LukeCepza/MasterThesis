x = EEG.data(1:2000)
level = 4;
waveletName = 'bior3.5';
[C, L] = wavedec(x, level, waveletName);
D4 = detcoef(C, L, level);
sigma_hat = median(abs(D4 - median(D4))) / 0.6745;
N = length(D4); 
alpha = sigma_hat * sqrt(2 * log(N));

for i = 1:level
    first = sum(L(1:end-level+i-1)) + 1;
    last = first + L(end-level+i) - 1;
    
    % Extract detail coefficients for level i
    Di = C(first:last);
    
    % Threshold the detail coefficients
    Di_thresh = Di .* (abs(Di) >= alpha);
    
    % Ensure the thresholded coefficients have the correct size
    if length(Di_thresh) ~= (last - first + 1)
        error('Thresholded coefficients have a different number of elements than expected.');
    end
    
    % Replace the original coefficients with the thresholded ones in the vector
    C(first:last) = Di_thresh;
end

% Reconstruct the signal from the modified wavelet decomposition vector
x_reconstructed = waverec(C, L, waveletName);

% Plot the original and reconstructed signals
figure;
subplot(2, 1, 1);
plot(x);
title('Original Signal');

subplot(2, 1, 2);
plot(x_reconstructed);
title('Reconstructed Signal (with thresholding)');
