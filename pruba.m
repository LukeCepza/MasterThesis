
channels = {EEG.chanlocs.labels}';
idx = find(strcmp(channels, 'C3'));
signal = squeeze(EEG.data(idx,:,:));

level = 4;
waveletName = 'bior3.5';
for epoch = 1%:size(signal,2)
    [C, L] = wavedec(signal(:,epoch), level, waveletName);
    D4 = detcoef(C, L, level);
    sigma_hat = mean(abs(D4)) / 0.675;
    N = length(D4); 
    alpha = sigma_hat * sqrt(2 * log(N));
    
    Di_thresh = [];
    for i = 1:level
        Di = detcoef(C, L, i);
        Di_thresh = cat(1,Di_thresh,Di .* (abs(Di) >= alpha));
    end

D4 = C(89:89+88);
D1 = C(end-630:end);
sigma_hat = mean(abs(D1)) / 0.675;
N = 88; 
alpha = sigma_hat * sqrt(2 * log(N));
C(89:end) = C(89:end) .* (abs(C(89:end)) >= alpha);
%C = C.* (abs(C) <= alpha);
%C(89:end) = Di_thresh;   
Xr = waverec(C, L, waveletName);


plot(epoch);
energy = (100 * norm(Xr)^2 / norm(signal(:, epoch))^2);
disp('The reconstructed retains ' + string(energy));

% Plot the original and reconstructed signals
figure(8);
subplot(3, 1, 1);
plot(signal(:, epoch));
title('Original Signal');

subplot(3, 1, 2);
plot(Xr);
title('Reconstructed Signal (with thresholding)');

C_r = round(C);
subplot(3, 1, 3);
Xrr = waverec(C_r, L, waveletName);

plot(C_r);
title('Binarized');

C_r_u = unique(C_r)
C_r_p = zeros(size(C_r_u));

% Calculate the probability for each unique number
for i = 1:length(C_r_u)
    specificNumber = C_r_u(i);
    frequency = sum(C_r == specificNumber);
    C_r_p(i) = frequency / length(C_r);
end

% Display the probabilities
for i = 1:length(C_r_u)
    disp(['The probability of ' num2str(C_r_u(i)) ' appearing in the list is ' num2str(C_r_p(i))]);
end

[dict,avglen] = huffmandict(C_r_u,C_r_p)
C_coded = huffmanenco(C_r,dict)

% Calculate the compression ratio
original_size = numel(C); % Size of the original signal coefficients
compressed_size = numel(C_coded); % Size of the compressed signal coefficients
CR = original_size / compressed_size;
F = (1 / CR) * 100;
disp(['Compression ratio: ' + string(CR)]);
disp(['Feature F (compression efficiency): ' + string(F) + '%']);
end



