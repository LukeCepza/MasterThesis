function plot_MIacross(MI, varargin)

    p = inputParser;

    % Define the mandatory argument
    addParameter(p, 'ColorLimits', [0 3] , @isnumeric);
    addParameter(p, 'fig_n', 11,@isnumeric);
    addParameter(p, 'Title', 'plot', @ischar);

    % Parse the input arguments
    parse(p, varargin{:});

   % Access the parsed values
    ColorLimits = p.Results.ColorLimits;
    fig_n = p.Results.fig_n;
    Title = p.Results.Title


    channels = {'Fp1';'Fp2';'F3'; 'F4';'C3';'C4';'P3'; ...
                    'P4';'O1';'O2';'F7';'F8';'T7';'T8';'P7'; ...
                    'P8';'Fz';'Cz';'Pz';'AFz';'CPz'; 'POz'};

    Dropped = mean(MI,[1,3,4],'omitnan') > 0;
    MIM_all = squeeze(mean(MI(:,Dropped,:,:),2,"omitnan"));
    
    llim = min(MIM_all, [],'all'); rlim = max(MIM_all,[], 'all');
    ColorLimits = [llim, rlim];

    f = figure(fig_n); 
    f.Name = 'MI several Plot'; 
    f.Color = 'white'; 
    pause(1); 
    set(gcf, 'Position', [0 0 1500, 500]); % Set size
    set(gcf, 'renderer', 'painters');
    
    tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
        
    %Air vs Vib
    nexttile;
    h = heatmap(squeeze(MIM_all(1, :, :)), ...
        'XLabel', 'Air', 'YLabel', 'Vib', ...
        'Title', 'Mutual Information Air vs Vib');
    
    colormap(jet);
    
    h.XDisplayLabels = string(channels);
    h.YDisplayLabels = string(channels);
    h.ColorLimits = ColorLimits;
    %Vib vs Car
    nexttile;
    h = heatmap(squeeze(MIM_all(2, :, :)), ...
        'XLabel', 'Vib', 'YLabel', 'Car', ...
        'Title', 'Mutual Information Vib vs Car');
    
    colormap(jet);
    
    h.XDisplayLabels = string(channels);
    h.YDisplayLabels = string(channels);
    h.ColorLimits = ColorLimits;
    
    %Car vs Air
    nexttile;
    h = heatmap(squeeze(MIM_all(3, :, :)), ...
        'XLabel', 'Car', 'YLabel', 'Air', ...
        'Title', 'Mutual Information Car vs Air');
    
    colormap(jet);
    
    h.XDisplayLabels = string(channels);
    h.YDisplayLabels = string(channels);
    h.ColorLimits = ColorLimits;
    sgtitle(Title)
end