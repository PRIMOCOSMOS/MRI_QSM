function run_whqsm_comparison(data_root, mri_qsm_root, varargin)
% run_whqsm_comparison.m (v4 - WH-QSM-only real subject pipeline)
% ============================================================================
% Complete two-subject real-data WH-QSM pipeline:
%   1) discover NORMAL + ELDERLY subjects from DICOM metadata
%   2) for each subject: DICOM phase/magnitude -> multi-echo field map
%   3) call lower-level SEPIA QSMMacroIOWrapper + FANSI weak-harmonic QSM
%   4) save individual WH-QSM results and non-registered QC summaries
%
% This v4 intentionally does NOT run TKD/CFL2/iLSQR/MEDI/xQSM on the real
% subjects. Those belong to the old algorithm-test pipeline and are not part
% of the real-data WH-QSM analysis requested here.
%
% Usage:
%   run_whqsm_comparison()
%   run_whqsm_comparison(data_root)
%   run_whqsm_comparison(data_root, mri_qsm_root)
%   run_whqsm_comparison(..., 'sepia_root', 'D:\path\sepia')
%   run_whqsm_comparison(..., 'keep_sepia_work', true)
% ============================================================================

%% Defaults
if nargin < 1 || isempty(data_root)
    data_root = 'D:\MRI_PRO\MRILAB_X\20170327_qsm2016_recon_challenge\data_course';
end
if nargin < 2 || isempty(mri_qsm_root)
    this_dir = fileparts(mfilename('fullpath'));
    mri_qsm_root = fileparts(this_dir);
end

data_root = char(data_root);
mri_qsm_root = char(mri_qsm_root);

p = inputParser;
addParameter(p, 'sepia_root', '', @(x) ischar(x) || isstring(x));
addParameter(p, 'keep_sepia_work', false, @islogical);
addParameter(p, 'mask_method', 'auto', @(x) ischar(x) || isstring(x));
addParameter(p, 'mask_erode_mm', 1.5, @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x >= 0);
addParameter(p, 'mask_threshold_factor', 0.12, @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x > 0 && x < 1);
addParameter(p, 'bet_fractional_threshold', 0.50, @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x > 0 && x < 1);
addParameter(p, 'bet_vertical_gradient', 0.0, @(x) isnumeric(x) && isscalar(x) && isfinite(x));
addParameter(p, 'run_susceptibility_separation', true, @islogical);
addParameter(p, 'chi_sep_root', '', @(x) ischar(x) || isstring(x));
addParameter(p, 'chi_sep_adapter_function', '', @(x) ischar(x) || isstring(x));
addParameter(p, 'suscep_sep_method', 'auto', @(x) ischar(x) || isstring(x));
addParameter(p, 'allow_exploratory_separation_fallback', false, @islogical);
addParameter(p, 'r2star_to_chi_abs_HzPerPpm', 137.0, @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x > 0);
addParameter(p, 'snu_local_field_mode', 'forward_from_whqsm', @(x) ischar(x) || isstring(x));
addParameter(p, 'snu_resgen', false, @islogical);
addParameter(p, 'snu_HaveR2Prime', false, @islogical);
addParameter(p, 'snu_is_scaling', false, @islogical);
addParameter(p, 'snu_scaling_factor', 0.19, @(x) isnumeric(x) && isscalar(x) && isfinite(x));
addParameter(p, 'snu_interp_method', 'sinc', @(x) ischar(x) || isstring(x));
addParameter(p, 'snu_sinc_window_size', 15, @(x) isnumeric(x) && isscalar(x) && isfinite(x));
addParameter(p, 'snu_sinc_window_type', 'hann', @(x) ischar(x) || isstring(x));
addParameter(p, 'snu_Dr', 114, @(x) isnumeric(x) && isscalar(x) && isfinite(x));
addParameter(p, 'whqsm_maxiter', 200, @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x > 0);
addParameter(p, 'whqsm_tol', 1e-5, @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x > 0);
addParameter(p, 'whqsm_lambda', 5e-4, @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x > 0);
addParameter(p, 'whqsm_beta', 150, @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x > 0);
addParameter(p, 'whqsm_muh', 5, @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x > 0);
parse(p, varargin{:});
user_sepia_root = char(p.Results.sepia_root);
keep_sepia_work = p.Results.keep_sepia_work;
mask_method = char(p.Results.mask_method);
mask_erode_mm = p.Results.mask_erode_mm;
mask_threshold_factor = p.Results.mask_threshold_factor;
bet_fractional_threshold = p.Results.bet_fractional_threshold;
bet_vertical_gradient = p.Results.bet_vertical_gradient;
run_susceptibility_separation = p.Results.run_susceptibility_separation;
chi_sep_root = char(p.Results.chi_sep_root);
chi_sep_adapter_function = char(p.Results.chi_sep_adapter_function);
suscep_sep_method = char(p.Results.suscep_sep_method);
allow_exploratory_separation_fallback = p.Results.allow_exploratory_separation_fallback;
r2star_to_chi_abs_HzPerPpm = p.Results.r2star_to_chi_abs_HzPerPpm;
snu_local_field_mode = char(p.Results.snu_local_field_mode);
snu_resgen = p.Results.snu_resgen;
snu_HaveR2Prime = p.Results.snu_HaveR2Prime;
snu_is_scaling = p.Results.snu_is_scaling;
snu_scaling_factor = p.Results.snu_scaling_factor;
snu_interp_method = char(p.Results.snu_interp_method);
snu_sinc_window_size = p.Results.snu_sinc_window_size;
snu_sinc_window_type = char(p.Results.snu_sinc_window_type);
snu_Dr = p.Results.snu_Dr;
whqsm_maxiter = p.Results.whqsm_maxiter;
whqsm_tol = p.Results.whqsm_tol;
whqsm_lambda = p.Results.whqsm_lambda;
whqsm_beta = p.Results.whqsm_beta;
whqsm_muh = p.Results.whqsm_muh;

fprintf('\n');
fprintf('╔══════════════════════════════════════════════════════════════╗\n');
fprintf('║  MRI_QSM real-data Pipeline v4: WH-QSM ONLY                 ║\n');
fprintf('║  DICOM multi-echo field fit → SEPIA/FANSI weak harmonic     ║\n');
fprintf('╚══════════════════════════════════════════════════════════════╝\n\n');

fprintf('[Init] Paths\n');
fprintf('  data_root    : %s\n', data_root);
fprintf('  MRI_QSM root : %s\n', mri_qsm_root);

this_dir = fileparts(mfilename('fullpath'));
addpath(genpath(this_dir));
addpath(mri_qsm_root);
addpath(fullfile(mri_qsm_root, 'modules'));
addpath(fullfile(mri_qsm_root, 'Utils_self'));
fprintf('  adapter      : %s\n', this_dir);

%% Preflight checks
fprintf('\n[Init] WH-QSM preflight checks\n');
if exist(data_root, 'dir') ~= 7
    error('Data root does not exist: %s', data_root);
end
fprintf('  ✅ data_root exists\n');

required_files = { ...
    fullfile(this_dir, 'dicom_loader_subject.m'), ...
    fullfile(this_dir, 'discover_subjects.m'), ...
    fullfile(this_dir, 'compare_subjects.m'), ...
    fullfile(mri_qsm_root, 'modules', 'mod_whqsm_reconstruction.m')};
for k = 1:numel(required_files)
    if exist(required_files{k}, 'file') ~= 2
        error('Required WH-QSM pipeline file missing: %s', required_files{k});
    end
end
fprintf('  ✅ WH-QSM-only modules exist\n');

if exist('dicominfo', 'file') ~= 2 || exist('dicomread', 'file') ~= 2
    error('dicominfo/dicomread not available. MATLAB Image Processing Toolbox is required.');
end
if exist('niftiwrite', 'file') ~= 2 || exist('niftiread', 'file') ~= 2
    error('niftiwrite/niftiread not available. MATLAB Image Processing Toolbox is required for SEPIA file I/O.');
end
fprintf('  ✅ DICOM and NIfTI I/O functions available\n');

sepiaRoot = resolve_sepia_root(user_sepia_root, mri_qsm_root);
if isempty(sepiaRoot)
    error(['SEPIA with QSMMacroIOWrapper was not found. WH-QSM-only pipeline cannot run without SEPIA.' newline ...
           'Pass explicit path: run_whqsm_comparison(..., ''sepia_root'', ''D:\path\sepia'')']);
end
fprintf('  ✅ SEPIA root: %s\n', sepiaRoot);

%% Step 1: discover subjects
fprintf('\n');
subjects = discover_subjects(data_root);
if isempty(subjects)
    error('No subjects discovered under data_root.');
end

n_normal  = sum(strcmp({subjects.group}, 'NORMAL'));
n_elderly = sum(strcmp({subjects.group}, 'ELDERLY'));
if n_normal == 0 || n_elderly == 0
    error(['Need at least one NORMAL and one ELDERLY subject for this two-subject WH-QSM run.' newline ...
           'Current counts: NORMAL=%d, ELDERLY=%d. Run inspect_subjects(data_root) or adjust folder names/metadata.'], ...
           n_normal, n_elderly);
end

idx_normal  = find(strcmp({subjects.group}, 'NORMAL'),  1, 'first');
idx_elderly = find(strcmp({subjects.group}, 'ELDERLY'), 1, 'first');
selected.normal = subjects(idx_normal);
selected.elderly = subjects(idx_elderly);

fprintf('\nSelected subjects:\n');
fprintf('  NORMAL  : %s (%s)\n', selected.normal.name, selected.normal.path);
fprintf('  ELDERLY : %s (%s)\n', selected.elderly.name, selected.elderly.path);
if n_normal > 1 || n_elderly > 1
    fprintf('  Note: multiple subjects were found; this pipeline processes the first NORMAL and first ELDERLY only.\n');
end

%% Output root
output_root = fullfile(data_root, '_qsm_comparison_results');
if ~exist(output_root, 'dir')
    mkdir(output_root);
end

%% Step 2: process each subject with WH-QSM only
results = struct();
labels = {'normal', 'elderly'};
for li = 1:numel(labels)
    grp = labels{li};
    sub = selected.(grp);

    fprintf('\n');
    fprintf('╔══════════════════════════════════════════════════════════════╗\n');
    fprintf('║  WH-QSM subject: %s (%s)\n', sub.name, upper(grp));
    fprintf('╚══════════════════════════════════════════════════════════════╝\n\n');

    sub_output = fullfile(output_root, [grp '_' sub.name]);
    if ~exist(sub_output, 'dir')
        mkdir(sub_output);
    end

    try
        % 2a. DICOM -> field map / mask / metadata
        data = dicom_loader_subject(sub, fullfile(sub_output, 'qsm2016_format'), ...
            'mask_method', mask_method, ...
            'mask_erode_mm', mask_erode_mm, ...
            'mask_threshold_factor', mask_threshold_factor, ...
            'bet_fractional_threshold', bet_fractional_threshold, ...
            'bet_vertical_gradient', bet_vertical_gradient);

        % 2b. Build minimal WH-QSM config
        cfg = build_whqsm_cfg(sub_output, mri_qsm_root, sepiaRoot, keep_sepia_work);
        cfg.whqsm.maxiter = whqsm_maxiter;
        cfg.whqsm.tol = whqsm_tol;
        cfg.whqsm.lambda = whqsm_lambda;
        cfg.whqsm.beta = whqsm_beta;
        cfg.whqsm.muh = whqsm_muh;
        cfg.sep.enable = run_susceptibility_separation;
        cfg.sep.method = suscep_sep_method;
        cfg.sep.chiSepRoot = chi_sep_root;
        cfg.sep.adapter_function = chi_sep_adapter_function;
        cfg.sep.allow_exploratory_fallback = allow_exploratory_separation_fallback;
        cfg.sep.r2star_to_chi_abs_HzPerPpm = r2star_to_chi_abs_HzPerPpm;
        cfg.sep.snu_local_field_mode = snu_local_field_mode;
        cfg.sep.snu_resgen = snu_resgen;
        cfg.sep.snu_HaveR2Prime = snu_HaveR2Prime;
        cfg.sep.snu_is_scaling = snu_is_scaling;
        cfg.sep.snu_scaling_factor = snu_scaling_factor;
        cfg.sep.snu_interp_method = snu_interp_method;
        cfg.sep.snu_sinc_window_size = snu_sinc_window_size;
        cfg.sep.snu_sinc_window_type = snu_sinc_window_type;
        cfg.sep.snu_Dr = snu_Dr;

        % 2c. Call lower-level SEPIA/FANSI weak-harmonic interface only
        [chi, whqsm_info] = mod_whqsm_reconstruction(data, cfg);
        chi(~data.Mask) = 0;

        % 2d. Susceptibility source separation add-on (toolbox first)
        sep_results = [];
        if cfg.sep.enable
            try
                sep_results = mod_susceptibility_separation(data, chi, cfg);
            catch ME_sep
                warning('Susceptibility separation failed/skipped: %s', ME_sep.message);
                sep_results = struct('status','failed','message',ME_sep.message);
            end
        end

        % 2d. Save subject result
        results.(grp) = struct( ...
            'name', sub.name, ...
            'group', upper(grp), ...
            'path', sub.path, ...
            'chi', chi, ...
            'mask', data.Mask, ...
            'spatial_res', data.spatial_res, ...
            'fieldmap_Hz', data.fieldmap_Hz, ...
            'local_field_ppm', data.local_field_ppm, ...
            'magn', data.magn, ...
            't1', data.mp_rage, ...
            'echo_times_ms', data.echo_times_ms, ...
            'delta_TE_sec', data.delta_TE, ...
            'B0', data.B0, ...
            'B0_dir', data.B0_dir, ...
            'phase_fit_method', data.phase_fit_method, ...
            'qsm_method', 'WH-QSM_SEPIA_FANSI', ...
            'whqsm_info', whqsm_info, ...
            'susceptibility_separation', sep_results);

        save(fullfile(sub_output, ['chi_' grp '.mat']), 'chi', 'whqsm_info', 'sep_results', '-v7.3');
        save(fullfile(sub_output, ['whqsm_' grp '_complete.mat']), 'data', 'chi', 'whqsm_info', 'sep_results', 'cfg', '-v7.3');
        fprintf('  💾 Subject WH-QSM saved: %s\n', sub_output);

    catch ME
        fprintf('\n  ❌ Subject %s failed: %s\n', sub.name, ME.message);
        if ~isempty(ME.stack)
            fprintf('     at %s:%d\n', ME.stack(1).name, ME.stack(1).line);
        end
        results.(grp) = [];
    end
end

%% Step 3: non-registered QC comparison / summaries
if isempty(results.normal) || isempty(results.elderly)
    save(fullfile(output_root, 'partial_results_failed.mat'), 'results', '-v7.3');
    error('At least one subject failed; comparison summaries were not generated. Partial results saved in %s', output_root);
end

fprintf('\n');
fprintf('╔══════════════════════════════════════════════════════════════╗\n');
fprintf('║  QC summary: NORMAL vs ELDERLY WH-QSM                       ║\n');
fprintf('║  No voxel-wise subtraction is performed without registration ║\n');
fprintf('╚══════════════════════════════════════════════════════════════╝\n\n');

compare_dir = fullfile(output_root, 'comparison');
compare_subjects(results.normal, results.elderly, compare_dir);

save(fullfile(output_root, 'all_whqsm_results.mat'), 'results', '-v7.3');

fprintf('\n');
fprintf('╔══════════════════════════════════════════════════════════════╗\n');
fprintf('║  ✅ WH-QSM-only real-data pipeline completed                 ║\n');
fprintf('║  Output root: %s\n', output_root);
fprintf('║  Main outputs:\n');
fprintf('║    - normal_*/chi_normal.mat\n');
fprintf('║    - elderly_*/chi_elderly.mat\n');
fprintf('║    - *_/whqsm_*_complete.mat\n');
fprintf('║    - comparison/compare_3view.png\n');
fprintf('║    - comparison/compare_histogram.png\n');
fprintf('║    - comparison/subject_summary.csv\n');
fprintf('╚══════════════════════════════════════════════════════════════╝\n\n');
end

%% =========================================================================
function sepiaRoot = resolve_sepia_root(userPath, mri_qsm_root)
sepiaRoot = '';
candidates = {};
if ~isempty(userPath)
    candidates{end+1} = userPath;
end
candidates = [candidates, { ...
    fullfile(mri_qsm_root, 'sepia'), ...
    'D:\MRI_PRO\MRILAB_X\sepia', ...
    'C:\MRILAB_X\sepia', ...
    'C:\sepia', ...
    '/opt/sepia', ...
    '/usr/local/sepia'}];

for k = 1:numel(candidates)
    c = candidates{k};
    if isempty(c), continue; end
    if exist(c, 'dir') == 7
        addpath(c);
        addpath(genpath(c));
        if exist('sepia_addpath', 'file') == 2
            try
                sepia_addpath;
            catch
            end
        end
        if exist('QSMMacroIOWrapper', 'file') == 2
            sepiaRoot = c;
            return;
        end
    end
end
end

%% =========================================================================
function cfg = build_whqsm_cfg(sub_output, mri_qsm_root, sepiaRoot, keep_sepia_work)
cfg = struct();
cfg.rootDir = sub_output;
cfg.dataDir = fullfile(sub_output, 'qsm2016_format');
cfg.outDir = sub_output;
cfg.figDir = fullfile(sub_output, 'figures');
cfg.resultDir = fullfile(sub_output, 'results');
cfg.mri_qsm_root = mri_qsm_root;
cfg.sepiaRoot = sepiaRoot;

for d = {cfg.outDir, cfg.figDir, cfg.resultDir}
    if ~exist(d{1}, 'dir'), mkdir(d{1}); end
end

% Validated WH-QSM/FANSI parameters. Adjust here only if the validated WH-QSM
% protocol changes.
cfg.whqsm.method = 'FANSI';
cfg.whqsm.isWeakHarmonic = true;
cfg.whqsm.reference_tissue = 'None';
cfg.whqsm.constraint = 'TV';
cfg.whqsm.lambda = 5e-4;
cfg.whqsm.tol = 1e-4;
cfg.whqsm.maxiter = 100;
cfg.whqsm.alpha1 = 5e-4;
cfg.whqsm.mu1 = 5e-5;
cfg.whqsm.mu = 5e-5;
cfg.whqsm.mu2 = 1.0;
cfg.whqsm.solver = 'Nonlinear';
cfg.whqsm.gradient_mode = 'none';
cfg.whqsm.beta = 150;
cfg.whqsm.muh = 5;
cfg.whqsm.isGPU = false;
cfg.whqsm.keep_work_dir = keep_sepia_work;

% Legacy flags kept explicit to prevent accidental algorithm-test modules from
% being called in this real-data pipeline.
cfg.deeplearning.enable = false;
cfg.run_only = 'WH-QSM';
end
