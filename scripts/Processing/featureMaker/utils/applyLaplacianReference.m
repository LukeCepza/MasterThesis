function laplacianData = applyLaplacianReference(data, channel)

    neighbors = {
        'Fp1', {'AFz', 'F3', 'F7'};
        'Fp2', {'AFz', 'F4', 'F8'};
        'F3', {'Fp1', 'Fz', 'C3', 'F7'};
        'F4', {'Fp2', 'Fz', 'C4', 'F8'};
        'C3', {'F3', 'Cz', 'P3', 'T7'};
        'C4', {'F4', 'Cz', 'P4', 'T8'};
        'P3', {'C3', 'Pz', 'O1', 'P7'};
        'P4', {'C4', 'Pz', 'O2', 'P8'};
        'O1', {'P3', 'POz', 'P7'};
        'O2', {'P4', 'POz', 'P8'};
        'F7', {'Fp1', 'F3', 'T7'};
        'F8', {'Fp2', 'F4', 'T8'};
        'T7', {'F7', 'C3', 'P7'};
        'T8', {'F8', 'C4', 'P8'};
        'P7', {'T7', 'P3', 'O1'};
        'P8', {'T8', 'P4', 'O2'};
        'Fz', {'Fp1', 'Fp2', 'F3', 'F4'};
        'Cz', {'Fz', 'C3', 'C4', 'CPz'};
        'Pz', {'Cz', 'P3', 'P4', 'POz'};
        'AFz', {'Fp1', 'Fp2', 'Fz'};
        'CPz', {'C3', 'Cz', 'C4'};
        'POz', {'Pz', 'O1', 'O2'};
        };

    laplacianData = zeros(size(data));
    
    for i = 1:length(channel)
        idx = find(strcmp(neighbors(:,1), channel{i}));
        
        if ~isempty(idx)
            neighborChannels = neighbors{idx, 2};
            neighborIdx = find(ismember(channel, neighborChannels));
            
            if ~isempty(neighborIdx)
                laplacianData(i, :) = data(i, :) - mean(data(neighborIdx, :), 1);
            else
                laplacianData(i, :) = data(i, :);
            end
        else
            laplacianData(i, :) = data(i, :);
        end
    end
    laplacianData = laplacianData(1:length(channel),:);
end

