function covmat = avg_covar(EEG)
    channels = {'Fp1';'Fp2';'F3'; 'F4';'C3';'C4';'P3'; ...
                'P4';'O1';'O2';'F7';'F8';'T7';'T8';'P7'; ...
                'P8';'Fz';'Cz';'Pz';'AFz';'CPz'; 'POz'};

    covmat = nan(22,22);
    chans_idx = ismember(channels,{EEG.chanlocs.labels}); % Find indexes of available channels
    
    covmat(chans_idx,chans_idx) = 0; % Find indexes of available channels
  
    seg = mean(EEG.data,3);  
    seg = seg - mean(seg,2); % mean centered
    covmat(chans_idx,chans_idx) =  seg*seg'/(size(seg,2)-1); % cpvariance calculationn 

end