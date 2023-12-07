function num_bins = freedmanDiaconisRule(signas)
    % based on https://stats.stackexchange.com/questions/798/calculating-optimal-number-of-bins-in-a-histogram
    q75 = prctile(signas, 75);
    q25 = prctile(signas, 25);
    iqr_value = q75 - q25;

    bin_width = 2 * iqr_value * (numel(signas)^(-1/3));

    num_bins = range(signas) / bin_width;

    num_bins = ceil(num_bins);
end