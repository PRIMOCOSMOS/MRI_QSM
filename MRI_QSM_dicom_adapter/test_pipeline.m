function test_pipeline(data_root, mri_qsm_root, varargin)
% test_pipeline.m (v4 - WH-QSM-only preflight / loader test)
% ============================================================================
% Lightweight test for the real-data WH-QSM pipeline. It does NOT run TKD,
% CFL2, iLSQR, MEDI, or xQSM. By default it tests:
%   1) path / dependency availability
%   2) subject discovery
%   3) DICOM loader, including two-echo field fitting
%
% Optional:
%   test_pipeline(..., 'run_whqsm', true, 'sepia_root', 'D:\path\sepia')
% ============================================================================

if nargin < 1 || isempty(data_root)
    data_root = 'D:\MRI_PRO\MRILAB_X\20170327_qsm2016_recon_challenge\data_course';
end
if nargin < 2 || isempty(mri_qsm_root)
    mri_qsm_root = fileparts(fileparts(mfilename('fullpath')));
end

p = inputParser;
addParameter(p, 'run_whqsm', false, @islogical);
addParameter(p, 'sepia_root', '', @(x) ischar(x) || isstring(x));
parse(p, varargin{:});
run_whqsm = p.Results.run_whqsm;
sepia_root = char(p.Results.sepia_root);

fprintf('\n');
fprintf('╔══════════════════════════════════════════════════════════════╗\n');
fprintf('║  WH-QSM-only pipeline test v4                               ║\n');
fprintf('╚══════════════════════════════════════════════════════════════╝\n\n');

this_dir = fileparts(mfilename('fullpath'));
addpath(genpath(this_dir));
addpath(mri_qsm_root);
addpath(fullfile(mri_qsm_root, 'modules'));
addpath(fullfile(mri_qsm_root, 'Utils_self'));

fprintf('[Test 1] Dependencies...\n');
needed = {'dicominfo','dicomread','niftiwrite','niftiread','mod_whqsm_reconstruction'};
for k = 1:numel(needed)
    if exist(needed{k}, 'file') == 2
        fprintf('  ✅ %s\n', needed{k});
    else
        fprintf('  ❌ %s\n', needed{k});
    end
end

if exist(data_root, 'dir') ~= 7
    fprintf('  ❌ data_root does not exist: %s\n', data_root);
    return;
end

fprintf('\n[Test 2] Subject discovery...\n');
subjects = discover_subjects(data_root);
if isempty(subjects)
    fprintf('  ❌ No subjects found.\n');
    return;
end

idx = find(strcmp({subjects.group}, 'NORMAL') | strcmp({subjects.group}, 'ELDERLY'), 1, 'first');
if isempty(idx), idx = 1; end
sub = subjects(idx);
fprintf('  Testing loader with subject: %s (%s)\n', sub.name, sub.group);

fprintf('\n[Test 3] DICOM loader / echo fitting...\n');
test_output = fullfile(data_root, '_whqsm_loader_test_output');
try
    data = dicom_loader_subject(sub, fullfile(test_output, 'qsm2016_format'));
    fprintf('  ✅ Loader succeeded.\n');
    fprintf('  Matrix       : %s\n', mat2str(data.N));
    fprintf('  Voxel size   : %s mm\n', mat2str(data.spatial_res, 6));
    fprintf('  B0           : %.4g T\n', data.B0);
    fprintf('  Echo times   : %s ms\n', mat2str(data.echo_times_ms, 6));
    fprintf('  delta_TE     : %.6g ms\n', data.delta_TE * 1000);
    fprintf('  Fit method   : %s\n', data.phase_fit_method);
    valsHz = data.fieldmap_Hz(data.Mask);
    valsPpm = data.local_field_ppm(data.Mask);
    fprintf('  Field Hz     : [%.5g, %.5g], std=%.5g\n', min(valsHz), max(valsHz), std(valsHz));
    fprintf('  Field ppm    : [%.5g, %.5g], std=%.5g\n', min(valsPpm), max(valsPpm), std(valsPpm));
catch ME
    fprintf('  ❌ Loader failed: %s\n', ME.message);
    if ~isempty(ME.stack), fprintf('     at %s:%d\n', ME.stack(1).name, ME.stack(1).line); end
    return;
end

if run_whqsm
    fprintf('\n[Test 4] Optional WH-QSM smoke run...\n');
    if isempty(sepia_root)
        fprintf('  ❌ sepia_root is required for optional WH-QSM test.\n');
        return;
    end
    cfg = struct();
    cfg.resultDir = fullfile(test_output, 'results');
    if ~exist(cfg.resultDir, 'dir'), mkdir(cfg.resultDir); end
    cfg.sepiaRoot = sepia_root;
    cfg.whqsm.keep_work_dir = true;
    try
        [chi, info] = mod_whqsm_reconstruction(data, cfg); %#ok<ASGLU>
        fprintf('  ✅ WH-QSM smoke run succeeded.\n');
    catch ME
        fprintf('  ❌ WH-QSM smoke run failed: %s\n', ME.message);
        return;
    end
else
    fprintf('\n[Test 4] Optional WH-QSM smoke run skipped. Use ''run_whqsm'', true to enable.\n');
end

fprintf('\n✅ WH-QSM-only pipeline test completed. Test output: %s\n\n', test_output);
end
