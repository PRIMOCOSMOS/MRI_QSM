function run_chisep_only_impl(whichSubject)
% RUN_CHISEP_ONLY.m
% ============================================================================
% Run only the susceptibility source separation stage using existing WH-QSM
% outputs. This avoids re-running DICOM loading and WH-QSM.
%
% Usage from project root:
%   RUN_CHISEP_ONLY              % process normal_* and elderly_* if present
%   RUN_CHISEP_ONLY('normal')
%   RUN_CHISEP_ONLY('elderly')
%   RUN_CHISEP_ONLY('D:\...\normal_swi_subj1')
% ============================================================================

if nargin < 1, whichSubject = 'all'; end

repoRoot = fileparts(fileparts(mfilename('fullpath')));
adapterDir = fullfile(repoRoot, 'MRI_QSM_dicom_adapter');
addpath(repoRoot, '-begin');
addpath(adapterDir, '-begin');
addpath(fullfile(repoRoot, 'modules'), '-begin');
addpath(fullfile(repoRoot, 'Utils_self'), '-begin');

P = whqsm_local_paths();
addpath(P.projectRoot, '-begin');
addpath(P.adapterDir, '-begin');
addpath(P.modulesDir, '-begin');
addpath(P.utilsDir, '-begin');
if isfield(P,'sepiaRoot') && exist(P.sepiaRoot,'dir') == 7
    addpath(P.sepiaRoot, '-begin'); addpath(genpath(P.sepiaRoot), '-begin');
end
if isfield(P,'mediRoot') && exist(P.mediRoot,'dir') == 7
    addpath(P.mediRoot, '-begin'); addpath(genpath(P.mediRoot), '-begin');
end
if isfield(P,'chiSepRoot') && exist(P.chiSepRoot,'dir') == 7
    addpath(P.chiSepRoot, '-begin'); addpath(genpath(P.chiSepRoot), '-begin');
end
% Put project modules first again so importONNXNetwork shim wins.
addpath(P.modulesDir, '-begin');
clear functions; clear mex; rehash;

outRoot = fullfile(P.dataRoot, '_qsm_comparison_results');
if exist(outRoot, 'dir') ~= 7
    error('QSM comparison output root not found: %s. Run WH-QSM first.', outRoot);
end

subjectDirs = resolve_subject_dirs(outRoot, whichSubject);
if isempty(subjectDirs)
    error('No matching subject result directories found under %s', outRoot);
end

fprintf('\n============================================================\n');
fprintf(' CHI-SEPARATION ONLY ENTRY\n');
fprintf('============================================================\n');
fprintf('Subjects: %d\n', numel(subjectDirs));
fprintf('chiSepRoot: %s\n', P.chiSepRoot);
fprintf('adapter   : %s\n', P.chiSepAdapterFunction);

for i = 1:numel(subjectDirs)
    subDir = subjectDirs{i};
    fprintf('\n------------------------------------------------------------\n');
    fprintf('Subject result dir: %s\n', subDir);
    [data, chi, cfg, label] = load_existing_whqsm_subject(subDir, P);
    cfg = configure_sep_cfg(cfg, P, subDir);
    sep_results = mod_susceptibility_separation(data, chi, cfg); %#ok<NASGU>
    save(fullfile(subDir, ['chisep_only_' label '.mat']), 'sep_results', 'cfg', '-v7.3');
    fprintf('Saved chi-separation-only result marker: %s\n', fullfile(subDir, ['chisep_only_' label '.mat']));
end

fprintf('\n✅ CHI-SEPARATION ONLY completed.\n');
end

%% =========================================================================
function subjectDirs = resolve_subject_dirs(outRoot, whichSubject)
whichSubject = char(whichSubject);
if exist(whichSubject, 'dir') == 7
    subjectDirs = {whichSubject};
    return;
end
allDirs = [dir(fullfile(outRoot, 'normal_*')); dir(fullfile(outRoot, 'elderly_*'))];
allDirs = allDirs([allDirs.isdir]);
subjectDirs = {};
for i = 1:numel(allDirs)
    name = allDirs(i).name;
    if strcmpi(whichSubject, 'all') || contains(lower(name), lower(whichSubject))
        subjectDirs{end+1} = fullfile(allDirs(i).folder, name); %#ok<AGROW>
    end
end
end

function [data, chi, cfg, label] = load_existing_whqsm_subject(subDir, P)
[~, name] = fileparts(subDir);
if startsWith(lower(name), 'normal_')
    label = 'normal';
elseif startsWith(lower(name), 'elderly_')
    label = 'elderly';
else
    label = 'subject';
end

completeFile = fullfile(subDir, ['whqsm_' label '_complete.mat']);
if exist(completeFile, 'file') ~= 2
    % fallback: find any complete mat
    d = dir(fullfile(subDir, 'whqsm_*_complete.mat'));
    if isempty(d)
        error('No whqsm_*_complete.mat found in %s', subDir);
    end
    completeFile = fullfile(d(1).folder, d(1).name);
end
S = load(completeFile);
if ~isfield(S, 'data') || ~isfield(S, 'chi')
    error('Complete file lacks data/chi: %s', completeFile);
end
data = S.data;
chi = S.chi;
if isfield(S, 'cfg')
    cfg = S.cfg;
else
    cfg = struct();
end

% If old complete file lacks R2star_Hz, try qsm2016_format data_full.
if ~isfield(data, 'R2star_Hz') || isempty(data.R2star_Hz)
    df = fullfile(subDir, 'qsm2016_format', 'data_full.mat');
    if exist(df, 'file') == 2
        T = load(df);
        if isfield(T, 'data') && isfield(T.data, 'R2star_Hz')
            data.R2star_Hz = T.data.R2star_Hz;
            data.R2star_s0 = T.data.R2star_s0;
            data.R2star_fit_residual = T.data.R2star_fit_residual;
        end
    end
end
if ~isfield(data, 'R2star_Hz') || isempty(data.R2star_Hz)
    error(['Existing WH-QSM data do not contain R2star_Hz. ' ...
           'Run the updated DICOM loader/WH-QSM once, or provide R2star_Hz.mat. Subject: %s'], subDir);
end
end

function cfg = configure_sep_cfg(cfg, P, subDir)
if ~isfield(cfg, 'resultDir') || isempty(cfg.resultDir)
    cfg.resultDir = fullfile(subDir, 'results');
end
if ~exist(cfg.resultDir, 'dir'), mkdir(cfg.resultDir); end
cfg.sep.enable = true;
cfg.sep.method = P.suscepSepMethod;
cfg.sep.chiSepRoot = P.chiSepRoot;
cfg.sep.adapter_function = P.chiSepAdapterFunction;
cfg.sep.allow_exploratory_fallback = P.allowExploratorySeparationFallback;
cfg.sep.r2star_to_chi_abs_HzPerPpm = P.r2starToChiAbsHzPerPpm;
cfg.sep.snu_local_field_mode = P.snuLocalFieldMode;
cfg.sep.snu_resgen = P.snuResgen;
cfg.sep.snu_HaveR2Prime = P.snuHaveR2Prime;
cfg.sep.snu_is_scaling = P.snuIsScaling;
cfg.sep.snu_scaling_factor = P.snuScalingFactor;
cfg.sep.snu_interp_method = P.snuInterpMethod;
cfg.sep.snu_sinc_window_size = P.snuSincWindowSize;
cfg.sep.snu_sinc_window_type = P.snuSincWindowType;
cfg.sep.snu_Dr = P.snuDr;
end
