function RUN_REALDATA_WHQSM_ONECLICK()
% RUN_REALDATA_WHQSM_ONECLICK.m
% Root-level wrapper for the fixed-path real-data WH-QSM pipeline.
%
% Usage from repository root:
%   RUN_REALDATA_WHQSM_ONECLICK

repoRoot = fileparts(mfilename('fullpath'));
adapterDir = fullfile(repoRoot, 'MRI_QSM_dicom_adapter');
if exist(adapterDir, 'dir') ~= 7
    error('Cannot find MRI_QSM_dicom_adapter under repo root: %s', repoRoot);
end
addpath(adapterDir, '-begin');
addpath(fullfile(repoRoot, 'modules'), '-begin');
addpath(fullfile(repoRoot, 'Utils_self'), '-begin');
feval('RUN_WHQSM_ONECLICK');
end
