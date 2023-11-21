addpath("D:\NYNGroup\eeglab2023.1\");
clear;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CONFIGURATION VARIABLES
type_of_pp  = 'pp02';
dataPath    = 'D:\shared_git\MaestriaThesis\data';
ppnum = type_of_pp;
id_str = 'aaaa';
do_dipolefit = false;
% structure of epochNames pair is {label, foldername, save name concatenate}
epochNamesPairs = {
    {'33028'  '33029'  '33030'  '33031'}, 'Air',['_' , type_of_pp , '_eAir'];
    {'33028'},'Air1',['_' , type_of_pp , '_eAir1'];
    {'33029'},'Air2',['_' , type_of_pp , '_eAir2'];
    {'33030'},'Air3',['_' , type_of_pp , '_eAir3'];
    {'33031'},'Air4',['_' , type_of_pp , '_eAir4'];
    {'33024'  '33025'  '33026'  '33027'}, 'Car',['_' , type_of_pp , '_eCar'];
    {'33024'}, 'Car1',['_' , type_of_pp , '_eCar1'];
    {'33025'}, 'Car2',['_' , type_of_pp , '_eCar2'];
    {'33026'}, 'Car3',['_' , type_of_pp , '_eCar3'];
    {'33027'}, 'Car4',['_' , type_of_pp , '_eCar4'];
    {'33032'  '33033'  '33034'  '33035'}, 'Vib',['_' , type_of_pp , '_eVib'];
    {'33032'}, 'Vib1',['_' , type_of_pp , '_eVib1'];
    {'33033'}, 'Vib2',['_' , type_of_pp , '_eVib2'];
    {'33034'}, 'Vib3',['_' , type_of_pp , '_eVib3'];
    {'33035'}, 'Vib4',['_' , type_of_pp , '_eVib4']};
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Extract Epochs
% Name is {label, Folder, set name}
[~,~,~] = eeglab;
for i = 1
    id_str = sprintf('ID%02d', i);    
    nameInE = fullfile(dataPath, id_str, type_of_pp, ['E_', id_str, '_' , type_of_pp , '.set']);

    if exist(nameInE, 'file') == 2 
        % input file exists
        mkdir(fullfile(dataPath, id_str, type_of_pp));
        disp("Epoching " + id_str);
       % EEG = pop_loadset('filename',['E_', id_str,'_pp.set' ],'filepath',fullfile(dataPath,id_str));
       
        extract_epochs(ppnum,                                   ...
                       epochNamesPairs,                         ...
                       dataPath,                                ...
                       id_str,                                  ...
                       'do_dipolefit', do_dipolefit);
    else
        % file doesn't exist
        disp("Skipping " + nameInE + " - file does not exist.");
    end
end
