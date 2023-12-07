function plotERSPFull_TL(tstimul, ERSP, fig_n, varargin)
    ts = linspace(-400,2440,200);
    freq = linspace(3,120,100);

    p = inputParser;

    % Define the mandatory argument
    addRequired(p, 'tstimul', @ischar);
    addRequired(p, 'ERSP', @isnumeric);
    addRequired(p, 'fig_n', @isnumeric);
    addParameter(p, 'xlim', [min(ts), max(ts)], @(x) isnumeric(x) && numel(x) == 2);
    addParameter(p, 'ylim', [min(freq),60], @(x) isnumeric(x) && numel(x) == 2);
    addParameter(p, 'clim', [0, 1], @(x) isnumeric(x) && numel(x) == 2);
    addParameter(p, 'ts', linspace(-400,2440,200), @isnumeric);
    addParameter(p, 'freq', linspace(3,120,100), @isnumeric);

    % Parse the input arguments
    parse(p, tstimul, ERSP, fig_n, varargin{:});

   % Access the parsed values
    tstimul = p.Results.tstimul;
    ERSP = p.Results.ERSP;
    fig_n = p.Results.fig_n;
    xaxeslims = p.Results.xlim;
    yaxeslims = p.Results.ylim;
    caxeslims = p.Results.clim;
    ts = p.Results.ts;
    freq = p.Results.freq;
%%
    f = figure(fig_n); 
    f.Name = 'ERSP Plot'; 
    f.Color = 'white'; 
    pause(1); 
    set(gcf, 'Position', [0 0 1500, 700]); % Set size
    set(gcf, 'renderer', 'painters');

    Labels = {'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', ...
        'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'AFz', 'CPz', 'POz'};
    Labels = convertCharsToStrings(Labels); % Get Labels

    % Matrix of Mbrain Cap chan locations for subplot locations
    subplotchloc = ["FP1", "2"; "FP2", "4"; "F3", "7"; "F4", "9"; "C3", "12"; ...
        "C4", "14"; "P3", "22"; "P4", "24"; "O1", "27"; "O2", "29"; "F7", "6"; ...
        "F8", "10"; "T7", "11"; "T8", "15"; "P7", "21"; "P8", "25"; "Fz", "8"; ...
        "Cz", "13"; "Pz", "23"; "AFz", "3"; "CPz", "18"; "POz", "28"];

    tiledlayout(6, 5, 'TileSpacing', 'compact', 'Padding', 'compact');
    
    ch = 0;
    for chpltloc = Labels % Plot per channel
        ch = ch + 1;
        Gav = squeeze(ERSP(ch, :, :));

        locplt = str2double(subplotchloc(find(chpltloc == subplotchloc), 2));
        nexttile(locplt);

        % Plot ERSP
        contourf(ts, freq, Gav, 'LineStyle', 'none');
        %set(gca, 'YDir', 'normal'); % Ensure frequency axis is oriented correctly
        %set(gca, 'Xtick', [-200, 0, 400], 'Ytick', [1, 100, 200]);
        set(gca, 'FontUnits', 'points', 'FontName', 'Sans', 'FontSize', 10);
        xlim(xaxeslims)
        ylim(yaxeslims) %        ylim([min(freq),max(freq)])
        clim(caxeslims)
        title(chpltloc);
        xline(0,'--r' );
        yline(10,'--b' );
        yline(21,'--b' );
    end

    % Add a common colorbar
    colorbar('Location', 'southoutside');

    % Add a global title
    sgtitle([tstimul, ' ERSP']);
end
