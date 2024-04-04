function EEG = dwtEnergy_tf(EEG)
    channels = {'Fp1';'Fp2';'F3'; 'F4';'C3';'C4';'P3'; ...
    'P4';'O1';'O2';'F7';'F8';'T7';'T8';'P7'; ...
    'P8';'Fz';'Cz';'Pz';'AFz';'CPz'; 'POz'};
    EEGchannels = {EEG.chanlocs.labels}';
    epoch_len = size(squeeze(EEG.data(1,:,:)),2);
    level = 5;
    waveletName = 'bior3.5';
    fs = 250;
    F_sub = nan(epoch_len,length(EEGchannels)*6*5);
    E = zeros(6,1);
    E_t = zeros(6*5,1);
    for epoch = 1:epoch_len
        for chan = 1:length(channels)
            %try
                idx = find(strcmp(channels, EEGchannels{chan}));
                signal = squeeze(EEG.data(idx,:,:));
                signal = signal(:,epoch);
                for t = 1:1:5
                    signal_t = signal(fs*(t-1)+1:fs*t);
                    [C, L] = wavedec(signal_t, level, waveletName);
                    Lidx = [1;cumsum(L)];
                    for Eidx = 1:6
                        E(Eidx) = sum(abs(C(Lidx(Eidx):Lidx(1+Eidx))).^2);
                    end
                    E_t((-5:0)+6*t*1) = E; 
                end
                E_tt = sum(E_t);
                E_t_rel = E_t./E_tt;
                F_sub(epoch,(-29:0)+5*6*chan) = E_t_rel;
            %catch
            %end
        end  
    end
    EEG.dwt_feats = F_sub;
end