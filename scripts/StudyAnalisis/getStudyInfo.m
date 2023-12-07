function StudyInfo = getStudyInfo(tstimul,type_of_pp) 
% This function returns a cell of folders containing the .set data for
% given condition
    cells = cell(1,35);
    idx = 0;
    for i = 13:50
        id_str = sprintf('ID%02d', i);    
        file = fullfile('D:\shared_git\MaestriaThesis\data',id_str , type_of_pp, tstimul, ...
            [id_str, '_' ,type_of_pp, '_e',tstimul,'.set']);
        if exist(file, 'file') == 2 
            idx = idx + 1;
            cells{idx} = {'index',idx,'load',file,'subject',id_str,...
                'condition',tstimul,'session',1,'group','1'};
            disp("attaching to list " + id_str);
        else
            disp("Skipping " + file + " - file does not exist.");
        end
    end
    StudyInfo = cells(~cellfun('isempty', cells));%%
end