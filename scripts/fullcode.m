%% Steps
addpath("D:\NYNGroup\eeglab2023.1\")
addpath('D:\shared_git\MaestriaThesis\scripts\Preprocessing')
addpath('D:\shared_git\MaestriaThesis\scripts\describe')
addpath('D:\shared_git\MaestriaThesis\scripts\epoch_extraction')
addpath('D:\shared_git\MaestriaThesis\scripts\StudyAnalisis')
addpath('D:\shared_git\MaestriaThesis\scripts\ERDS')
addpath('D:\shared_git\MaestriaThesis\scripts\others\MI')
addpath('D:\shared_git\MaestriaThesis\scripts\others\Covar')
%% First, Preprocess data which could be
%   1. call_preproEEG.m pp.set 
addpath('D:\shared_git\MaestriaThesis\scripts\Preprocessing')
run('call_preproEEG')
% Preprocessing done:
% pp01 - Classic method, epochs not renamed
% pp02 - Classic method, epochs renamed to perception
% pp03 - Classic method, for dipole fitting. ICA components are not rejected.
% pp04 - Classic method, epochs not renamed reref M1M2 then global
% pp05 - Classic method, epochs not renamed reref M1M2 then laplacian
%%  2. call_summarize.m which creates a summary of the retained information
%Creates a summary of the preprocessed data and stores it in a table
addpath('D:\shared_git\MaestriaThesis\scripts\describe')
run('call_summarize')
%%  3. after preprocessing pp signals
addpath('D:\shared_git\MaestriaThesis\scripts\epoch_extraction')
% Second, Epoch extraction creates the folders and extracts the epochs to
% those folders.
%   1. call_extract_epochs.m which extracts epochs for pp.set (normal) and \
% pe_pp.set (perceived epoch)
run('call_extract_epochs')
%% 4. exportAllTimeERP.m Extract time based ERPs characteristics.
addpath('D:\shared_git\MaestriaThesis\scripts\epoch_extraction')
run('exportAllTimeERP')
%% 5. ERDS maps
addpath('D:\shared_git\MaestriaThesis\scripts\ERDS')
run('call_ERDS.m')
%% Mutual information 
addpath('D:\shared_git\MaestriaThesis\scripts\others\MI')
run('call_MI.m')
run("call_across_condition_MI.m")
%% Covar
addpath('D:\shared_git\MaestriaThesis\scripts\others\Covar')
run('D:\shared_git\MaestriaThesis\scripts\others\Covar')
%% Study analisis
% Medir amplitud y latencia
addpath('D:\shared_git\MaestriaThesis\scripts\StudyAnalisis')
run('StudyEEGLABAnalisisERP.m')