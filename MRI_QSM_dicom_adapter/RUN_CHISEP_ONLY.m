function RUN_CHISEP_ONLY(whichSubject)
% RUN_CHISEP_ONLY.m
% Adapter-folder wrapper: run susceptibility separation only from existing WH-QSM outputs.
if nargin < 1
    whichSubject = 'all';
end
adapterDir = fileparts(mfilename('fullpath'));
repoRoot = fileparts(adapterDir);
addpath(fullfile(repoRoot, 'modules'), '-begin');
addpath(adapterDir, '-begin');
addpath(fullfile(repoRoot, 'Utils_self'), '-begin');
run_chisep_only_impl(whichSubject);
end
