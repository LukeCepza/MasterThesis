
channels = {'Fp1';'Fp2';'F3'; 'F4';'C3';'C4';'P3'; ...
'P4';'O1';'O2';'F7';'F8';'T7';'T8';'P7'; ...
'P8';'Fz';'Cz';'Pz';'AFz';'CPz'; 'POz'};
EEGchannels = {EEG.chanlocs.labels}';
epoch_len = size(squeeze(EEG.data(1,:,:)),2);
F_sub = zeros(length(EEGchannels),epoch_len);
for epoch = 1:epoch_len
    for chan = 1:length(EEGchannels)
        idx = find(strcmp(channels, EEGchannels{chan}));
        signal = squeeze(EEG.data(idx,:,:));
        level = 4;
        waveletName = 'bior3.5';
            [C, L] = wavedec(signal(:,epoch), level, waveletName);
            D4 = C(89:89+88);
            D2 = C(89+88:89+88+165);
            D3 = C(end-630-320:end-630);
            D1 = C(end-630:end);
            sigma_hat = median(abs(D3)) / 0.675;
            N = 88; 
            alpha = sigma_hat * sqrt(2 * log(N));
            C(89:end) = C(89:end) .* (abs(C(89:end)) >= alpha);           

            Xr = waverec(C, L, waveletName);
            energy = (100 * norm(Xr)^2 / norm(signal(:, epoch))^2);
            disp('The reconstructed retains ' + string(energy));
            
            C = round(C*100)/100;
            C_u = unique(C);
            C_p = zeros(size(C_u));

        for i = 1:length(C_u)
            specificNumber = C_u(i);
            frequency = sum(C == specificNumber);
            C_p(i) = frequency / length(C);
        end
           
        [dict,avglen] = huffmandict(C_u,C_p);
        C_coded = huffmanenco(C,dict);

            C_r = round(C);  
            C_r_u = unique(C_r);
            C_r_p = zeros(size(C_r_u));

        for i = 1:length(C_r_u)
            specificNumber = C_r_u(i);
            frequency = sum(C_r == specificNumber);
            C_r_p(i) = frequency / length(C_r);
        end
           
        [dict,avglen] = huffmandict(C_r_u,C_r_p);
        C_r_coded = huffmanenco(C_r,dict);
        
        original_size = numel(C_coded); 
        compressed_size = numel(C_r_coded); 
        CR = original_size / compressed_size;
        F = (1 / CR) * 100;
        F_sub(chan,epoch) = F;
    end
end
