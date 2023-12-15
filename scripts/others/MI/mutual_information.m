function MI = mutual_information(X,Y,varargin)
% as described in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3037822/
% high MI means there is a relationship
    p = inputParser;

    addRequired(p, 'X', @isnumeric);
    addRequired(p, 'Y', @isnumeric);

    addParameter(p, 'numBins', 15, @isnumeric)
    addParameter(p, 'freedmanDiaconisRule', true, @islogical)
    parse(p, X,Y,varargin{:});
    
    numBinsX = p.Results.numBins;
    numBinsY = numBinsX;
    do_freedmanDiaconisRule = p.Results.freedmanDiaconisRule;

    if do_freedmanDiaconisRule 
        % Use Freedman-Diaconis rule to determine the number of bins
        numBinsX = freedmanDiaconisRule(X);
        numBinsY = freedmanDiaconisRule(Y);
    end

    % Estimate PDF
    P_X = histcounts(X, numBinsX, 'Normalization', 'probability');
    P_Y = histcounts(Y, numBinsY, 'Normalization', 'probability');
    P_XY = histcounts2(X, Y, [numBinsX, numBinsY], 'Normalization', 'probability');

    %Estimate Shannon entropy
    H_X = -sum(P_X .* log2(P_X), 'omitnan');
    H_Y = -sum(P_Y .* log2(P_Y), 'omitnan');
    H_XY = -sum(P_XY(:) .* log2(P_XY(:)), 'omitnan');
    
    % Calculate mutual information
    MI = H_X + H_Y - H_XY;
end

