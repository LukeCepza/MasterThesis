function plotERPFull_TL(ERP_list, ts, varargin)

    p = inputParser;

        % Define the mandatory argument
    addRequired(p, 'ERP_list', @isnumeric);
    addRequired(p, 'ts', @isnumeric);

    addParameter(p, 'ylimits', [-3 3], @isnumeric)
    addParameter(p, 'tstimul', '', @ischar)
    addParameter(p, 'figColor', [1 1 1], @isnumeric)
    addParameter(p, 'LineW', 0.9, @isnumeric)
    addParameter(p, 'fig_n', 99, @isnumeric)

    % Parse the input arguments
    parse(p, ERP_list, ts, varargin{:});

   % Access the parsed values
    ERP_list = p.Results.ERP_list;
    ts= p.Results.ts;
    ylimits = p.Results.ylimits;
    tstimul = p.Results.tstimul;
    figColor = p.Results.figColor;
    LineW = p.Results.LineW;
    fig_n = p.Results.fig_n;
    
    f = figure(fig_n); 
    f.Name = '22 Chan Plot ERP'; 
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

    % Create a 6x5 tiled layout
    tiledlayout(6, 5, 'TileSpacing', 'compact', 'Padding', 'compact');

    ch = 0;
    for chpltloc = Labels % Plot per channel
        ch = ch + 1;
        Gav = ERP_list(ch, :);

        locplt = str2double(subplotchloc(find(chpltloc == subplotchloc), 2));
        nexttile(locplt);
        
        % Plot ERP
        hold on;
        plot(ts, Gav, 'LineWidth', LineW, 'Color', figColor);
        set(gca, 'Xtick', [-200, 0, 400], 'Ytick', [-4, 0, 6]);
        set(gca, 'FontUnits', 'points', 'FontName', 'Sans', 'FontSize', 10);
        xlim([-1000 2000])
        ylim(ylimits)
        line([0 0], [0 2], 'Color', [0.1 0.1 0.1], 'LineWidth', 0.8);
        yline(0, 'LineWidth', 1, 'Color', [0.1 0.1 0.1], 'Alpha', 0.4);

        ax = gca;
        ax.XAxisLocation = 'origin';
        ax.YAxisLocation = 'origin';
        set(gca, 'XColor', 'none', 'YColor', 'none', 'TickDir', 'in');
    end
    % Add a global title
    sgtitle([tstimul, ' ERP']);
end


