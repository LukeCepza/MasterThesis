%% Functions
function MIM = MI_matrix(EEG)
    
    channels = {'Fp1';'Fp2';'F3'; 'F4';'C3';'C4';'P3'; ...
                    'P4';'O1';'O2';'F7';'F8';'T7';'T8';'P7'; ...
                    'P8';'Fz';'Cz';'Pz';'AFz';'CPz'; 'POz'};
    data = mean(EEG.data,3);  
    MIM = nan(22,22);

    d1 = 0;
    for idx_1 = 1:22
        if ismember(channels(idx_1),{EEG.chanlocs.labels})
            d1 = 1+d1;
            d2 = 0;
            for idx_2 = 1:22
                if ismember(channels(idx_2),{EEG.chanlocs.labels})
                    d2 = 1+d2;
                    MIM(idx_1,idx_2) = mutual_information(data(d1,:),data(d2,:));   
                end
            end
        end
    end

end
