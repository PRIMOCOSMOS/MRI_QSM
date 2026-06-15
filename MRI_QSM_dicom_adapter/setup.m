% setup.m (v4 - WH-QSM-only real-data pipeline)
% ============================================================================
% Configure MATLAB path for the DICOM real-subject WH-QSM-only pipeline.
% Usage:
%   cd MRI_QSM/MRI_QSM_dicom_adapter
%   setup
%   run_whqsm_comparison(...)
% ============================================================================

fprintf('\n');
fprintf('╔══════════════════════════════════════════════════════════════╗\n');
fprintf('║  MRI_QSM dicom_adapter setup v4: WH-QSM ONLY                ║\n');
fprintf('╚══════════════════════════════════════════════════════════════╝\n\n');

this_dir = fileparts(mfilename('fullpath'));
mri_qsm_root = fileparts(this_dir);

addpath(genpath(this_dir));
addpath(mri_qsm_root);
addpath(fullfile(mri_qsm_root, 'modules'));
addpath(fullfile(mri_qsm_root, 'Utils_self'));

fprintf('✅ adapter      : %s\n', this_dir);
fprintf('✅ MRI_QSM root : %s\n', mri_qsm_root);

fprintf('\n[1/3] Checking WH-QSM-only project files...\n');
required = { ...
    fullfile(this_dir, 'run_whqsm_comparison.m'), ...
    fullfile(this_dir, 'dicom_loader_subject.m'), ...
    fullfile(this_dir, 'discover_subjects.m'), ...
    fullfile(this_dir, 'compare_subjects.m'), ...
    fullfile(mri_qsm_root, 'modules', 'mod_whqsm_reconstruction.m')};
for k = 1:numel(required)
    if exist(required{k}, 'file') == 2
        fprintf('  ✅ %s\n', required{k});
    else
        fprintf('  ❌ %s\n', required{k});
    end
end

fprintf('\n[2/3] Checking MATLAB I/O functions...\n');
check_func('dicominfo');
check_func('dicomread');
check_func('niftiwrite');
check_func('niftiread');
check_func('imfill');
check_func('bwconncomp');

fprintf('\n[3/3] Checking SEPIA candidates...\n');
sepia_candidates = { ...
    fullfile(mri_qsm_root, 'sepia'), ...
    'D:\MRI_PRO\MRILAB_X\sepia', ...
    'C:\MRILAB_X\sepia', ...
    'C:\sepia', ...
    '/opt/sepia', ...
    '/usr/local/sepia'};
sepia_found = false;
for k = 1:numel(sepia_candidates)
    c = sepia_candidates{k};
    if exist(c, 'dir') == 7
        addpath(c); addpath(genpath(c));
        if exist('sepia_addpath', 'file') == 2
            try sepia_addpath; catch; end %#ok<CTCH>
        end
        if exist('QSMMacroIOWrapper', 'file') == 2
            fprintf('  ✅ SEPIA/QSMMacroIOWrapper: %s\n', c);
            sepia_found = true;
            break;
        else
            fprintf('  ⚠️ SEPIA-like folder but QSMMacroIOWrapper missing: %s\n', c);
        end
    end
end
if ~sepia_found
    fprintf('  ❌ SEPIA/QSMMacroIOWrapper not found. WH-QSM cannot run until SEPIA is installed or passed via sepia_root.\n');
end

fprintf('\n');
fprintf('╔══════════════════════════════════════════════════════════════╗\n');
fprintf('║  Setup complete                                              ║\n');
fprintf('║  Next: run_whqsm_comparison(data_root, mri_qsm_root, ...     ║\n');
fprintf('║          ''sepia_root'', ''D:\\path\\sepia'')                  ║\n');
fprintf('╚══════════════════════════════════════════════════════════════╝\n\n');

function check_func(name)
if exist(name, 'file') == 2 || exist(name, 'builtin') == 5
    fprintf('  ✅ %s\n', name);
else
    fprintf('  ❌ %s\n', name);
end
end
