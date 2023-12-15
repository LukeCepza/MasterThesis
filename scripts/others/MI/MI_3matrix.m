function MIM_triple = MI_3matrix(EEG1,EEG2,EEG3)

   channels = {'Fp1';'Fp2';'F3'; 'F4';'C3';'C4';'P3'; ...
                    'P4';'O1';'O2';'F7';'F8';'T7';'T8';'P7'; ...
                    'P8';'Fz';'Cz';'Pz';'AFz';'CPz'; 'POz'};
    MIM_triple = nan(3,22,22);

    A = squeeze(reshape(EEG1.data,size(EEG1.data,1),[],1));  
    B = squeeze(reshape(EEG2.data,size(EEG2.data,1),[],1));  
    C = squeeze(reshape(EEG3.data,size(EEG3.data,1),[],1));  

    minlength = min([size(A,2),size(B,2),size(C,2)]);

    d1 = 0;
    % Test A vs B.
    for idx_1 = 1:22
        if ismember(channels(idx_1),{EEG1.chanlocs.labels})
            d1 = 1+d1;
            d2 = 0;
            for idx_2 = 1:22
                if ismember(channels(idx_2),{EEG2.chanlocs.labels})
                    d2 = 1+d2;
                    MIM_triple(1,idx_1,idx_2) = mutual_information(A(d1,1:minlength),B(d2,1:minlength),'numBins',13,'freedmanDiaconisRule', false);                         
                end
            end
        end
    end

    % Test B vs C.
    d1 = 0;
    for idx_1 = 1:22
        if ismember(channels(idx_1),{EEG2.chanlocs.labels})
            d1 = 1+d1;
            d2 = 0;
            for idx_2 = 1:22
                if ismember(channels(idx_2),{EEG3.chanlocs.labels})
                    d2 = 1+d2;
                    MIM_triple(2,idx_1,idx_2) = mutual_information(B(d1,1:minlength),C(d2,1:minlength),'numBins',13,'freedmanDiaconisRule', false);  
                
                end
            end
        end
    end

    % Test C vs. A
    d1 = 0;
    for idx_1 = 1:22
        if ismember(channels(idx_1),{EEG3.chanlocs.labels})
            d1 = 1+d1;
            d2 = 0;
            for idx_2 = 1:22
                if ismember(channels(idx_2),{EEG1.chanlocs.labels})
                    d2 = 1+d2;
                    MIM_triple(3,idx_1,idx_2) = mutual_information(C(d1,1:minlength),A(d2,1:minlength),'numBins',13,'freedmanDiaconisRule', false);   
                end
            end
        end
    end

end