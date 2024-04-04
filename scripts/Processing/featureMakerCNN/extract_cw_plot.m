function EEG = extract_cw_plot(EEG, ch)
   std_val = 3;
   plot_freq = 250;

   epoch_len = size(squeeze(EEG.data(1,:,:)),2);
   x_size = 2*plot_freq*5;
   [~, f] = cwt(EEG.data(ch,:,1),'amor',EEG.srate);
   f = f(f <= 60);

   cw_plot = zeros(epoch_len,length(f), x_size);

    for epoch = 1:epoch_len
        [cw, ~] = cwt(EEG.data(ch,:,epoch),'amor',EEG.srate);
        cw = cw(1:length(f), :);
        cw_ds = zeros(length(f), x_size);

        for i = 1:length(f)
            cw_ds(i, :) = resample(cw(i, :), x_size, 1250);
        end
        cw_plot_temp = real(cw_ds);
        doubleArray = cw_plot_temp; 

        meanVal = mean(doubleArray(:));
        stdDev = std(doubleArray(:));
        
        normalizedArray = (doubleArray - meanVal) / (stdDev);
        
        normalizedArray(normalizedArray < -std_val * stdDev) = -std_val * stdDev;
        normalizedArray(normalizedArray > std_val * stdDev) = std_val * stdDev;  
        normalizedArray = normalizedArray + std_val*stdDev; 
        scaledArray = normalizedArray * 65535 / (std_val*stdDev*2);
        cw_plot_temp = uint16(scaledArray);
        cw_plot(epoch,:,:) = cw_plot_temp;
    end
    EEG.cw_plot = cw_plot;
end
