%% Steps
addpath("D:\NYNGroup\eeglab2023.1\")
addpath('D:\shared_git\MaestriaThesis\scripts\epoch_extraction')
addpath('D:\shared_git\MaestriaThesis\scripts\StudyAnalisis')
addpath('D:\shared_git\MaestriaThesis\scripts\Preprocessing')
addpath('D:\shared_git\MaestriaThesis\scripts\describe')
addpath('D:\shared_git\MaestriaThesis\scripts\StudyAnalisis')
%% First, Preprocess data which could be
%   1. call_preproEEG.m pp.set 
run('call_preproEEG')
% Preprocessing done:
% pp01 - Classic method, epochs not renamed
% pp02 - Classic method, epochs renamed to perception
% pp03 - Classic method, for dipole fitting. ICA components are not rejected.
% pp04 - Classic method, epochs not renamed reref M1M2 then global
% pp05 - Classic method, epochs not renamed reref M1M2 then laplacian
%%  2. call_summarize.m which creates a summary of the retained information
run('call_summarize')
%%  3.   after preprocessing pp signals
% Second, Epoch extraction creates the folders and extracts the epochs to
% those folders.
%   1. call_extract_epochs.m which extracts epochs for pp.set (normal) and \
% pe_pp.set (perceived epoch)
run('call_extract_epochs')
%% 4. exportAllTimeERP.m Extract time based ERPs characteristics.
run('exportAllTimeERP')

