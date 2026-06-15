function RUN_CHISEP_ONLY(whichSubject)
% RUN_CHISEP_ONLY.m
% Root-level wrapper: run susceptibility separation only from existing WH-QSM outputs.
if nargin < 1
    whichSubject = 'all';
end
repoRoot = fileparts(mfilename('fullpath'));
addpath(fullfile(repoRoot, 'modules'), '-begin');
addpath(fullfile(repoRoot, 'MRI_QSM_dicom_adapter'), '-begin');
addpath(fullfile(repoRoot, 'Utils_self'), '-begin');
run_chisep_only_impl(whichSubject);
end
