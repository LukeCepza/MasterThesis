% My issue is that some dipoles are not present in the data, so i remove in
% the data those. Original script looped through all of the dipoles.

% Load study and run

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Obtain all dipole xyz coordinates as a list. %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function plot_MakotoClusterOptimization(ALLEEG)
    dipXyz = [];
    for subjIdx = 1:length(ALLEEG)
     
        % Obtain xyz, dip moment, maxProj channel xyz.   
        xyz = zeros(length(ALLEEG(subjIdx).dipfit.model),3);
        for modelIdx = 1:length(ALLEEG(subjIdx).dipfit.model)
     
            % Choose the larger dipole if symmetrical.
            currentXyz = ALLEEG(subjIdx).dipfit.model(modelIdx).posxyz
            currentMom = ALLEEG(subjIdx).dipfit.model(modelIdx).momxyz % nAmm.
            if size(currentMom,1) == 2
                [~,largerOneIdx] = max([norm(currentMom(1,:)) norm(currentMom(2,:))]);
                currentXyz = ALLEEG(subjIdx).dipfit.model(modelIdx).posxyz(largerOneIdx,:);
            end
            
            try
                xyz(modelIdx,:) = currentXyz
            catch
                disp('No data')
            end
    
        end
        dipXyz = [dipXyz; xyz];
    end
     dipXyz = dipXyz(any(dipXyz,2),:); % remove zero rows
    
     %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% Optimize the number of clusters between the range 5-15. %%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Reduce data dimension using PCA. Use Matlab function evalclusters().
    
    kmeansClusterIdxMatrix = zeros(size(dipXyz,1),11);
    meanWithinClusterDistance = nan(11+4,11);
    for clustIdx = 1:20
        [IDX, ~, SUMD] = kmeans(dipXyz, clustIdx+4, 'emptyaction', 'singleton', 'maxiter', 10000, 'replicate', 100);
        kmeansClusterIdxMatrix(:,clustIdx)            = IDX;
        numIcEntries = hist(IDX, 1:clustIdx+4);
        meanWithinClusterDistance(1:clustIdx+4, clustIdx) = SUMD./numIcEntries';
    end
     
    eva1 = evalclusters(dipXyz, kmeansClusterIdxMatrix, 'CalinskiHarabasz');
    eva2 = evalclusters(dipXyz, kmeansClusterIdxMatrix, 'Silhouette');
    eva3 = evalclusters(dipXyz, kmeansClusterIdxMatrix, 'DaviesBouldin');
    
    %%   
    subplot(2,2,1)
    boxplot(meanWithinClusterDistance);axis square;
    set(gca, 'xticklabel', 5:24)
    xlabel('Number of clusters')
    ylabel('Mean distance to cluster centroid')
    subplot(2,2,2)
    plot(eva1); title('CalinskiHarabasz');axis square;
    subplot(2,2,3)
    plot(eva2); title('Silhouette');axis square;
    subplot(2,2,4)
    plot(eva3); title('DaviesBouldin');axis square;
    
end