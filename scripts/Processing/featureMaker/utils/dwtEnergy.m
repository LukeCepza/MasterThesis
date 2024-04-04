function EEG = dwtEnergy(EEG)
    channels = {'Fp1';'Fp2';'F3'; 'F4';'C3';'C4';'P3'; ...
    'P4';'O1';'O2';'F7';'F8';'T7';'T8';'P7'; ...
    'P8';'Fz';'Cz';'Pz';'AFz';'CPz'; 'POz'};
    EEGchannels = {EEG.chanlocs.labels}';
    epoch_len = size(squeeze(EEG.data(1,:,:)),2);
    
    F_sub = nan(epoch_len,length(channels)*6);
    for epoch = 1:epoch_len
        for chan = 1:length(channels)
            try
                idx = find(strcmp(channels, EEGchannels{chan}));
                signal = squeeze(EEG.data(idx,:,:));
                level = 5;
                waveletName = 'bior3.5';
                [C, L] = wavedec(signal(:,epoch), level, waveletName);
                Lidx = [1;cumsum(L)];
                E = zeros(6,1);
                for Eidx = 1:6
                    E(Eidx) = sum(abs(C(Lidx(Eidx):Lidx(1+Eidx))).^2);
                end
                Et = sum(E);
                E_rel = E./Et;
    
                F_sub(epoch,(-5:0)+6*chan) = E_rel;
            catch
    
            end 
        end  
    end
    EEG.dwt_feats = F_sub;
end