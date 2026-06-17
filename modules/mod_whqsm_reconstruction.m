function [chi, info] = mod_whqsm_reconstruction(data, cfg)
% mod_whqsm_reconstruction.m
% ============================================================================
% WH-QSM-only reconstruction module for real subject DICOM data.
%
% This module intentionally bypasses the old algorithm-comparison pipeline
% (TKD/CFL2/iLSQR/MEDI/xQSM). It calls the validated lower-level SEPIA
% QSMMacroIOWrapper + FANSI weak-harmonic interface directly.
%
% Required data fields:
%   data.Mask / data.msk          brain mask
%   data.spatial_res             [dx dy dz] in mm
%   data.B0 or data.b0           scanner field strength, Tesla
%   one of:
%       data.fieldmap_Hz         field map in Hz, recommended
%       data.local_field_ppm     field map in ppm
%       data.phs_tissue          compatibility ppm field
%   optional:
%       data.magn                magnitude image
%       data.echo_times_sec      vector of echo times in seconds
%       data.delta_TE            echo spacing in seconds
%
% Required cfg fields:
%   cfg.sepiaRoot
%   cfg.resultDir
%   cfg.whqsm.*                  optional FANSI/WH-QSM parameters
% ============================================================================

chi = [];
info = struct();

%% ------------------------------------------------------------------------
% Validate data / cfg
% -------------------------------------------------------------------------
if nargin < 2
    error('mod_whqsm_reconstruction requires data and cfg.');
end

if ~isfield(cfg, 'resultDir') || isempty(cfg.resultDir)
    error('cfg.resultDir is required.');
end
if ~exist(cfg.resultDir, 'dir')
    mkdir(cfg.resultDir);
end

if ~isfield(cfg, 'sepiaRoot') || isempty(cfg.sepiaRoot)
    error('cfg.sepiaRoot is required for WH-QSM.');
end
sepiaRoot = char(cfg.sepiaRoot);
if exist(sepiaRoot, 'dir') ~= 7
    error('SEPIA root does not exist: %s', sepiaRoot);
end

Mask = get_mask(data);
N = size(Mask);
voxel_size = get_voxel_size(data);
B0 = get_scalar_field(data, {'B0','b0','FieldStrength','MagneticFieldStrength'}, 3);
B0_dir = get_vector_field(data, {'B0_dir','b0dir'}, [0 0 1]);
B0_dir = B0_dir ./ max(norm(B0_dir), eps);

[localField_Hz, localField_ppm, field_source] = get_fieldmap(data, Mask, B0);
localField_Hz(~Mask) = 0;
localField_ppm(~Mask) = 0;

% -------------------------------------------------------------------------
% 背景场去除(BFR): QSM 共识规定的反演前必需步骤。
% - 真实 DICOM 数据: data.fieldmap_Hz 是【总场】, 必须 BFR 转成局部场。
% - Challenge 数据: 已是 phs_tissue(局部场), 不需要也不应再做(默认关闭)。
% 由 cfg.whqsm.do_bfr 控制(真实数据 pipeline 自动开启)。不改动反演本身。
% -------------------------------------------------------------------------
do_bfr = get_cfg_bool(cfg, {'whqsm','do_bfr'}, false);
MaskStrict = Mask;
MaskFilled = Mask;
if do_bfr
    if exist('mod_field_preprocess','file') == 2
        try
            [localField_Hz, prepInfo] = mod_field_preprocess(data, Mask, cfg);
            if isfield(prepInfo,'mask_two_pass_strict') && isequal(size(prepInfo.mask_two_pass_strict), size(Mask))
                MaskStrict = logical(prepInfo.mask_two_pass_strict);
            elseif isfield(prepInfo,'mask_for_qsm') && isequal(size(prepInfo.mask_for_qsm), size(Mask))
                MaskStrict = logical(prepInfo.mask_for_qsm);
            end
            if isfield(prepInfo,'mask_two_pass_filled') && isequal(size(prepInfo.mask_two_pass_filled), size(Mask))
                MaskFilled = logical(prepInfo.mask_two_pass_filled);
            elseif isfield(prepInfo,'mask_after_bfr') && isequal(size(prepInfo.mask_after_bfr), size(Mask))
                MaskFilled = logical(prepInfo.mask_after_bfr);
            end
            localField_Hz(~MaskFilled) = 0;
            localField_ppm = localField_Hz ./ (resolve_gyro() * B0);
            localField_ppm(~MaskFilled) = 0;
            field_source = sprintf('%s -> BFR(%s)', field_source, prepInfo.bfr_method);
            info.field_preprocess = prepInfo; %#ok<STRNU>
        catch ME_bfr
            error(['背景场去除(BFR)失败: %s' newline ...
                   '请确认 MEDI/SEPIA 在 path 上(P.mediRoot/P.sepiaRoot)。' newline ...
                   '若你的输入已是局部场(如 Challenge phs_tissue), 请设 cfg.whqsm.do_bfr=false。'], ME_bfr.message);
        end
    else
        warning('cfg.whqsm.do_bfr=true 但找不到 mod_field_preprocess.m, 跳过 BFR。');
    end
end

% Default single-pass path uses the broad/fill mask to avoid holes in final
% real-data outputs. The strict mask is only used when two-pass is enabled.
useTwoPass = get_cfg_bool(cfg, {'whqsm','use_two_pass_qsm'}, false);
if useTwoPass && nnz(MaskStrict) < nnz(MaskFilled)
    Mask = MaskStrict;
else
    Mask = MaskFilled;
end
localField_Hz_single = localField_Hz; localField_Hz_single(~Mask) = 0;
localField_ppm_single = localField_ppm; localField_ppm_single(~Mask) = 0;

if ~is_valid_volume(localField_Hz_single, Mask)
    error('WH-QSM input field map is invalid or nearly all zero. Source=%s', field_source);
end

TE_sec = get_echo_times_sec(data);
delta_TE_sec = get_delta_te_sec(data, TE_sec);

fprintf('\n============================================================\n');
fprintf(' WH-QSM reconstruction only (SEPIA/FANSI weak harmonic)\n');
fprintf('============================================================\n');
fprintf('Input field source : %s\n', field_source);
fprintf('Matrix size        : [%d %d %d]\n', N(1), N(2), N(3));
fprintf('Voxel size         : [%.6g %.6g %.6g] mm\n', voxel_size(1), voxel_size(2), voxel_size(3));
fprintf('B0                 : %.6g T\n', B0);
fprintf('B0 direction       : [%.6g %.6g %.6g]\n', B0_dir(1), B0_dir(2), B0_dir(3));
if isempty(TE_sec)
    fprintf('Echo times         : <not available>\n');
else
    fprintf('Echo times         : %s ms\n', mat2str(TE_sec(:).' * 1000, 6));
    fprintf('delta_TE           : %.6g ms\n', delta_TE_sec * 1000);
end
print_stats('Input field Hz (strict mask)', localField_Hz_single, Mask);
print_stats('Input field ppm (strict mask)', localField_ppm_single, Mask);
if nnz(MaskFilled) > nnz(Mask)
    print_stats('Input field Hz (filled mask)', localField_Hz, MaskFilled);
end

%% ------------------------------------------------------------------------
% 优先委托给【Challenge 已验证】的 inversion_whqsm_stable(mod_dipole_inversion.m)。
% 在 real-data 中可选 two-pass：
%   pass-1 用严格/有洞 mask 重建可靠与细微来源；
%   pass-2 用填洞/更宽 mask 重建强源与 less-reliable 区域；
%   最终用 pass-2 只填补 pass-1 的缺失区。
% 这不替代 WH-QSM，而是把 WH-QSM 作为单次反演器调用两次，是一种 masking /
% artefact-reduction strategy（QSMxT 思路），与 weak-harmonic 正则互补。
% -------------------------------------------------------------------------
useValidated = get_cfg_bool(cfg, {'whqsm','use_validated_inversion'}, true);
useTwoPass = get_cfg_bool(cfg, {'whqsm','use_two_pass_qsm'}, false);
if useValidated && exist('inversion_whqsm_stable','file')==2
    try
        if useTwoPass && nnz(MaskFilled) > nnz(MaskStrict)
            fprintf('\nWH-QSM: 启用 two-pass WH-QSM (strict + filled masks).\n');
            [chi_strict, ok1, msg1] = run_validated_whqsm_once(localField_ppm_single, data, MaskStrict, voxel_size);
            [chi_filled, ok2, msg2] = run_validated_whqsm_once(localField_ppm, data, MaskFilled, voxel_size);
            if ok1 && ok2
                chi = chi_strict;
                fillRegion = MaskFilled & ~MaskStrict;
                chi(fillRegion) = chi_filled(fillRegion);
                finalMask = MaskFilled;
                chi(~finalMask) = 0;
                validate_qsm_or_error(chi, finalMask);
                info.inversion_path = 'two-pass inversion_whqsm_stable (validated strict+filled)';
                info.field_source = field_source;
                info.B0 = B0; info.B0_dir = B0_dir; info.voxel_size = voxel_size;
                info.matrix_size = N; info.final_mask = finalMask; info.finished_at = datestr(now,31);
                info.two_pass = struct('enabled', true, 'strict_mask_voxels', nnz(MaskStrict), ...
                    'filled_mask_voxels', nnz(MaskFilled), 'filled_only_voxels', nnz(fillRegion), ...
                    'strict_msg', msg1, 'filled_msg', msg2);
                print_stats('WH-QSM chi ppm (two-pass final)', chi, finalMask);
                outMat = fullfile(cfg.resultDir, 'whqsm_result.mat');
                Mask = finalMask; %#ok<NASGU>
                save(outMat, 'chi', 'chi_strict', 'chi_filled', 'info', 'localField_Hz', 'localField_ppm', 'Mask', 'MaskStrict', 'MaskFilled', '-v7.3');
                try
                    niftiwrite(single(chi_strict), fullfile(cfg.resultDir, 'WHQSM_chi_strict.nii'));
                    niftiwrite(single(chi_filled), fullfile(cfg.resultDir, 'WHQSM_chi_filled.nii'));
                    niftiwrite(uint8(MaskStrict), fullfile(cfg.resultDir, 'WHQSM_mask_strict.nii'));
                    niftiwrite(uint8(MaskFilled), fullfile(cfg.resultDir, 'WHQSM_mask_filled.nii'));
                catch
                end
                return;
            else
                warning('two-pass validated WH-QSM 未完全成功，回退单次 strict pass. strict=%s | filled=%s', msg1, msg2);
            end
        end

        data_for_inv = data;
        data_for_inv.Mask = Mask;
        if ~isfield(data_for_inv,'magn') || isempty(data_for_inv.magn)
            data_for_inv.magn = get_magnitude(data, Mask);
        end
        fprintf('\nWH-QSM: 委托 Challenge 已验证的 inversion_whqsm_stable (ppm 输入)。\n');
        chi = inversion_whqsm_stable(localField_ppm_single, data_for_inv, voxel_size);
        chi = double(squeeze(chi));
        if ndims(chi) > 3, chi = chi(:,:,:,1); end
        if isequal(size(chi), N) && is_valid_volume(chi, Mask)
            chi(~Mask) = 0;
            validate_qsm_or_error(chi, Mask);
            info.inversion_path = 'inversion_whqsm_stable (validated, shared with Challenge)';
            info.field_source = field_source;
            info.B0 = B0; info.B0_dir = B0_dir; info.voxel_size = voxel_size;
            info.matrix_size = N; info.final_mask = Mask; info.finished_at = datestr(now,31);
            print_stats('WH-QSM chi ppm', chi, Mask);
            outMat = fullfile(cfg.resultDir, 'whqsm_result.mat');
            save(outMat, 'chi', 'info', 'localField_Hz_single', 'localField_ppm_single', 'Mask', '-v7.3');
            return;
        else
            warning('inversion_whqsm_stable 输出无效/尺寸不符, 回退到内置 SEPIA 调用。');
        end
    catch ME_val
        warning('委托 inversion_whqsm_stable 失败, 回退内置实现: %s', ME_val.message);
    end
end

%% ------------------------------------------------------------------------
% Add SEPIA path and validate lower-level interface
% -------------------------------------------------------------------------
addpath(sepiaRoot);
addpath(genpath(sepiaRoot));

if exist('sepia_addpath', 'file') == 2
    try
        sepia_addpath;
        fprintf('SEPIA sepia_addpath called successfully.\n');
    catch ME
        warning('sepia_addpath failed, continuing with genpath: %s', ME.message);
    end
end

if exist('QSMMacroIOWrapper', 'file') ~= 2
    error('QSMMacroIOWrapper not found after adding SEPIA path: %s', sepiaRoot);
end
if exist('niftiwrite', 'file') ~= 2
    error('niftiwrite is not available. Image Processing Toolbox is required.');
end
if exist('niftiread', 'file') ~= 2
    error('niftiread is not available. Image Processing Toolbox is required.');
end

gyro = resolve_gyro();
CF = B0 * gyro;  % MHz; 1 ppm = CF Hz because CF is in MHz

%% ------------------------------------------------------------------------
% Build FANSI / weak-harmonic parameters
% -------------------------------------------------------------------------
algorParam = build_fansi_whqsm_params(cfg);

%% ------------------------------------------------------------------------
% Prepare SEPIA file interface
% -------------------------------------------------------------------------
workDir = make_work_dir(cfg.resultDir);
if ~exist(workDir, 'dir')
    mkdir(workDir);
end

info.workDir = workDir;
info.sepiaRoot = sepiaRoot;
info.field_source = field_source;
info.B0 = B0;
info.B0_dir = B0_dir;
info.gyro_MHz_per_T = gyro;
info.CF_MHz = CF;
info.echo_times_sec = TE_sec;
info.delta_TE_sec = delta_TE_sec;
info.voxel_size = voxel_size;
info.matrix_size = N;
info.started_at = datestr(now, 31);

localFieldFile  = fullfile(workDir, 'Sepia_localfield_Hz.nii');
maskFile        = fullfile(workDir, 'Sepia_mask.nii');
magFile         = fullfile(workDir, 'Sepia_magnitude.nii');
headerFile      = fullfile(workDir, 'Sepia_header.mat');
output_basename = fullfile(workDir, 'Sepia_WHQSM');

try
    niftiwrite(single(localField_Hz), localFieldFile);
    niftiwrite(uint8(Mask), maskFile);

    mag = get_magnitude(data, Mask);
    niftiwrite(single(mag), magFile);

    header = struct();
    header.matrixSize  = N;
    header.matrix_size = N;
    header.voxelSize   = voxel_size;
    header.voxel_size  = voxel_size;
    header.b0dir       = B0_dir;
    header.B0_dir      = B0_dir;
    header.b0          = B0;
    header.B0          = B0;
    header.TE          = TE_sec;
    header.delta_TE    = delta_TE_sec;
    header.CF          = CF;
    header.units       = 'local field in Hz';

    % SEPIA versions differ in the exact variable names expected by the
    % wrapper, so save redundant aliases intentionally.
    matrix_size = N; %#ok<NASGU>
    matrixSize  = N; %#ok<NASGU>
    voxelSize   = voxel_size; %#ok<NASGU>
    B0_dir      = B0_dir; %#ok<NASGU>
    b0dir       = B0_dir; %#ok<NASGU>
    b0          = B0; %#ok<NASGU>
    TE          = TE_sec; %#ok<NASGU>
    delta_TE    = delta_TE_sec; %#ok<NASGU>

    save(headerFile, ...
        'header', 'matrix_size', 'matrixSize', ...
        'voxel_size', 'voxelSize', ...
        'B0_dir', 'b0dir', 'B0', 'b0', ...
        'TE', 'delta_TE', 'CF');

    input = struct();
    input(1).name = localFieldFile;
    input(2).name = magFile;
    input(3).name = '';
    input(4).name = headerFile;
    mask_filename = maskFile;

    fprintf('\nCalling lower-level SEPIA QSMMacroIOWrapper...\n');
    fprintf('  local field : %s\n', localFieldFile);
    fprintf('  mask        : %s\n', maskFile);
    fprintf('  magnitude   : %s\n', magFile);
    fprintf('  header      : %s\n', headerFile);
    fprintf('  output base : %s\n', output_basename);

    tStart = tic;
    QSMMacroIOWrapper(input, output_basename, mask_filename, algorParam);
    info.runtime_sec = toc(tStart);

    qsmPath = resolve_sepia_qsm_output(output_basename, workDir);
    if isempty(qsmPath)
        niiList = list_nifti_files(workDir);
        error(['SEPIA completed but QSM output could not be located.' newline ...
               'WorkDir: %s' newline 'NIfTI files: %s'], workDir, strjoin(niiList, ', '));
    end

    fprintf('Reading SEPIA WH-QSM output: %s\n', qsmPath);
    chi = double(squeeze(niftiread(qsmPath)));

    if ndims(chi) > 3
        chi = chi(:,:,:,1);
    end
    if ~isequal(size(chi), N)
        error('WH-QSM output size mismatch. Expected [%d %d %d], got %s.', ...
            N(1), N(2), N(3), mat2str(size(chi)));
    end

    chi(~Mask) = 0;
    validate_qsm_or_error(chi, Mask);

    info.qsmPath = qsmPath;
    info.finished_at = datestr(now, 31);
    info.algorParam = algorParam;
    info.header = header;
    info.final_mask = Mask;

    print_stats('WH-QSM chi ppm', chi, Mask);

    outMat = fullfile(cfg.resultDir, 'whqsm_result.mat');
    save(outMat, 'chi', 'info', 'algorParam', 'header', ...
        'localField_Hz', 'localField_ppm', 'Mask', '-v7.3');
    fprintf('Saved WH-QSM MAT: %s\n', outMat);

    try
        outNii = fullfile(cfg.resultDir, 'WHQSM_chi.nii');
        niftiwrite(single(chi), outNii);
        fprintf('Saved WH-QSM NIfTI: %s\n', outNii);
        info.savedNifti = outNii;
    catch ME
        warning('Could not save WH-QSM NIfTI copy: %s', ME.message);
    end

    keepWorkDir = get_cfg_bool(cfg, {'whqsm','keep_work_dir'}, false);
    if ~keepWorkDir
        try
            rmdir(workDir, 's');
            info.workDir_removed = true;
        catch ME
            warning('Could not remove SEPIA workDir %s: %s', workDir, ME.message);
            info.workDir_removed = false;
        end
    else
        info.workDir_removed = false;
        fprintf('Keeping SEPIA workDir for QC: %s\n', workDir);
    end

catch ME
    info.failed_at = datestr(now, 31);
    info.error_message = ME.message;
    try
        save(fullfile(cfg.resultDir, 'whqsm_failed_debug.mat'), 'info', 'algorParam', '-v7.3');
    catch
    end
    fprintf('\nWH-QSM failed. SEPIA workDir retained for debugging: %s\n', workDir);
    rethrow(ME);
end

fprintf('WH-QSM completed in %.2f sec.\n', info.runtime_sec);
fprintf('============================================================\n\n');

end

%% =========================================================================
function [chi_out, ok, msg] = run_validated_whqsm_once(localField_ppm_in, data, MaskIn, voxel_size)
chi_out = [];
ok = false;
msg = 'not_run';
try
    data_for_inv = data;
    data_for_inv.Mask = logical(MaskIn);
    if ~isfield(data_for_inv,'magn') || isempty(data_for_inv.magn)
        data_for_inv.magn = get_magnitude(data, data_for_inv.Mask);
    end
    chi_out = inversion_whqsm_stable(localField_ppm_in, data_for_inv, voxel_size);
    chi_out = double(squeeze(chi_out));
    if ndims(chi_out) > 3, chi_out = chi_out(:,:,:,1); end
    chi_out(~data_for_inv.Mask) = 0;
    if is_valid_volume(chi_out, data_for_inv.Mask)
        ok = true;
        msg = 'ok';
    else
        msg = 'invalid_output';
    end
catch ME
    msg = ME.message;
end
end

%% =========================================================================
% Data access helpers
% =========================================================================
function Mask = get_mask(data)
if isfield(data, 'Mask') && ~isempty(data.Mask)
    Mask = logical(data.Mask);
elseif isfield(data, 'msk') && ~isempty(data.msk)
    Mask = logical(data.msk);
else
    error('data.Mask or data.msk is required.');
end
if ndims(Mask) ~= 3 || nnz(Mask) == 0
    error('Mask must be a non-empty 3D logical volume.');
end
end

function voxel_size = get_voxel_size(data)
if ~isfield(data, 'spatial_res') || numel(data.spatial_res) < 3
    error('data.spatial_res [dx dy dz] is required.');
end
voxel_size = double(data.spatial_res(:).');
voxel_size = voxel_size(1:3);
if any(~isfinite(voxel_size)) || any(voxel_size <= 0)
    error('Invalid voxel size: %s', mat2str(voxel_size));
end
end

function val = get_scalar_field(s, names, default)
val = default;
for i = 1:numel(names)
    name = names{i};
    if isfield(s, name) && ~isempty(s.(name)) && isnumeric(s.(name))
        tmp = double(s.(name));
        tmp = tmp(1);
        if isfinite(tmp) && tmp > 0
            val = tmp;
            return;
        end
    end
end
end

function vec = get_vector_field(s, names, default)
vec = default;
for i = 1:numel(names)
    name = names{i};
    if isfield(s, name) && ~isempty(s.(name)) && isnumeric(s.(name)) && numel(s.(name)) >= 3
        tmp = double(s.(name)(1:3));
        if all(isfinite(tmp)) && norm(tmp) > 0
            vec = tmp(:).';
            return;
        end
    end
end
vec = double(vec(:).');
end

function [fieldHz, fieldPpm, source] = get_fieldmap(data, Mask, B0)
gyro = 42.57747892;
if isfield(data, 'fieldmap_Hz') && ~isempty(data.fieldmap_Hz)
    fieldHz = double(data.fieldmap_Hz);
    fieldPpm = fieldHz ./ (gyro * B0);
    source = 'data.fieldmap_Hz from DICOM multi-echo phase fit';
elseif isfield(data, 'local_field_ppm') && ~isempty(data.local_field_ppm)
    fieldPpm = double(data.local_field_ppm);
    fieldHz = fieldPpm .* (gyro * B0);
    source = 'data.local_field_ppm';
elseif isfield(data, 'phs_tissue') && ~isempty(data.phs_tissue)
    fieldPpm = double(data.phs_tissue);
    fieldHz = fieldPpm .* (gyro * B0);
    source = 'data.phs_tissue compatibility ppm field';
else
    error('No usable field map found: need fieldmap_Hz, local_field_ppm, or phs_tissue.');
end
if ~isequal(size(fieldHz), size(Mask))
    error('Field map size %s does not match Mask size %s.', mat2str(size(fieldHz)), mat2str(size(Mask)));
end
fieldHz(~isfinite(fieldHz)) = 0;
fieldPpm(~isfinite(fieldPpm)) = 0;
end

function TE_sec = get_echo_times_sec(data)
TE_sec = [];
if isfield(data, 'echo_times_sec') && ~isempty(data.echo_times_sec)
    TE_sec = double(data.echo_times_sec(:).');
elseif isfield(data, 'TE') && ~isempty(data.TE)
    TE_sec = double(data.TE(:).');
elseif isfield(data, 'EchoTime') && ~isempty(data.EchoTime)
    TE_sec = double(data.EchoTime(:).') / 1000;
end
TE_sec = TE_sec(isfinite(TE_sec) & TE_sec > 0);
TE_sec = unique(TE_sec);
end

function delta_TE_sec = get_delta_te_sec(data, TE_sec)
delta_TE_sec = [];
if isfield(data, 'delta_TE') && ~isempty(data.delta_TE)
    delta_TE_sec = double(data.delta_TE(1));
elseif isfield(data, 'delta_TE_sec') && ~isempty(data.delta_TE_sec)
    delta_TE_sec = double(data.delta_TE_sec(1));
end
if isempty(delta_TE_sec) || ~isfinite(delta_TE_sec) || delta_TE_sec <= 0
    if numel(TE_sec) >= 2
        d = diff(sort(TE_sec));
        d = d(isfinite(d) & d > 0);
        if ~isempty(d)
            delta_TE_sec = median(d);
        end
    elseif numel(TE_sec) == 1
        delta_TE_sec = TE_sec(1);
    else
        delta_TE_sec = 0.025;
        warning('No TE/delta_TE found; using fallback delta_TE=25 ms only for SEPIA header.');
    end
end
end

function mag = get_magnitude(data, Mask)
if isfield(data, 'magn') && ~isempty(data.magn) && isequal(size(data.magn), size(Mask))
    mag = double(data.magn);
elseif isfield(data, 'magn_raw') && ~isempty(data.magn_raw) && isequal(size(data.magn_raw), size(Mask))
    mag = double(data.magn_raw);
else
    mag = double(Mask);
end
mag(~isfinite(mag)) = 0;
mag(~Mask) = 0;
end

%% =========================================================================
% SEPIA / FANSI helpers
% =========================================================================
function gyro = resolve_gyro()
% Use MHz/T. Some SEPIA installations define gyro. Accept only plausible
% MHz/T values; otherwise fallback to the IUPAC proton value.
fallback = 42.57747892;
gyro = fallback;
if exist('sepia_universal_variables', 'file') == 2
    try
        sepia_universal_variables;
        if exist('gyro', 'var') && isnumeric(gyro) && isfinite(gyro) && gyro > 1 && gyro < 100
            return;
        else
            gyro = fallback;
        end
    catch
        gyro = fallback;
    end
end
end

function algorParam = build_fansi_whqsm_params(cfg)
algorParam = struct();
algorParam.general.isBET = 0;
algorParam.general.isInvert = 0;

algorParam.qsm.reference_tissue = get_cfg_value(cfg, {'whqsm','reference_tissue'}, 'None');
algorParam.qsm.method = 'FANSI';
algorParam.qsm.tol = get_cfg_value(cfg, {'whqsm','tol'}, 1e-4);
algorParam.qsm.maxiter = get_cfg_value(cfg, {'whqsm','maxiter'}, 100);
algorParam.qsm.lambda = get_cfg_value(cfg, {'whqsm','lambda'}, 5e-4);
algorParam.qsm.alpha1 = get_cfg_value(cfg, {'whqsm','alpha1'}, 5e-4);
algorParam.qsm.mu1 = get_cfg_value(cfg, {'whqsm','mu1'}, 5e-5);
algorParam.qsm.mu = get_cfg_value(cfg, {'whqsm','mu'}, 5e-5);
algorParam.qsm.mu2 = get_cfg_value(cfg, {'whqsm','mu2'}, 1.0);
algorParam.qsm.solver = get_cfg_value(cfg, {'whqsm','solver'}, 'Nonlinear');
algorParam.qsm.constraint = get_cfg_value(cfg, {'whqsm','constraint'}, 'TV');
algorParam.qsm.gradient_mode = get_cfg_value(cfg, {'whqsm','gradient_mode'}, 'none');
algorParam.qsm.isWeakHarmonic = true;
algorParam.qsm.beta = get_cfg_value(cfg, {'whqsm','beta'}, 150);
algorParam.qsm.muh = get_cfg_value(cfg, {'whqsm','muh'}, 5);
algorParam.qsm.isGPU = get_cfg_value(cfg, {'whqsm','isGPU'}, false);

fprintf('WH-QSM / FANSI parameters:\n');
fprintf('  method          : %s\n', algorParam.qsm.method);
fprintf('  isWeakHarmonic  : %d\n', algorParam.qsm.isWeakHarmonic);
fprintf('  constraint      : %s\n', algorParam.qsm.constraint);
fprintf('  lambda          : %.6g\n', algorParam.qsm.lambda);
fprintf('  maxiter         : %d\n', algorParam.qsm.maxiter);
fprintf('  tol             : %.6g\n', algorParam.qsm.tol);
fprintf('  beta            : %.6g\n', algorParam.qsm.beta);
end

function v = get_cfg_value(cfg, pathCells, default)
v = default;
try
    s = cfg;
    for i = 1:numel(pathCells)
        if isfield(s, pathCells{i})
            s = s.(pathCells{i});
        else
            return;
        end
    end
    if ~isempty(s)
        v = s;
    end
catch
    v = default;
end
end

function tf = get_cfg_bool(cfg, pathCells, default)
v = get_cfg_value(cfg, pathCells, default);
tf = logical(v);
end

function workDir = make_work_dir(resultDir)
stamp = datestr(now, 'yyyymmdd_HHMMSS');
workDir = fullfile(resultDir, sprintf('sepia_whqsm_work_%s_%06d', stamp, randi(999999)));
end

function qsmPath = resolve_sepia_qsm_output(output_basename, workDir)
qsmPath = '';
[outDir, base, ~] = fileparts(output_basename);
if nargin < 2 || isempty(workDir)
    workDir = outDir;
end

cands = { ...
    fullfile(outDir, [base '_QSM.nii.gz']), ...
    fullfile(outDir, [base '_QSM.nii']), ...
    fullfile(outDir, [base '_qsm.nii.gz']), ...
    fullfile(outDir, [base '_qsm.nii']), ...
    fullfile(outDir, [base '_Chi.nii.gz']), ...
    fullfile(outDir, [base '_Chi.nii']), ...
    fullfile(outDir, [base '_chi.nii.gz']), ...
    fullfile(outDir, [base '_chi.nii']), ...
    fullfile(outDir, [base '_Chimap.nii.gz']), ...
    fullfile(outDir, [base '_Chimap.nii']), ...
    fullfile(outDir, [base '_chimap.nii.gz']), ...
    fullfile(outDir, [base '_chimap.nii'])};

for i = 1:numel(cands)
    if exist(cands{i}, 'file') == 2
        qsmPath = cands{i};
        return;
    end
end

files = [dir(fullfile(workDir, '*.nii')); dir(fullfile(workDir, '*.nii.gz'))];
if isempty(files)
    return;
end

bestScore = -Inf;
bestPath = '';
for i = 1:numel(files)
    f = files(i);
    lname = lower(f.name);
    score = 0;
    if contains(lname, 'chimap'), score = score + 20; end
    if contains(lname, 'qsm'),    score = score + 15; end
    if contains(lname, 'chi'),    score = score + 10; end
    if contains(lower(fullfile(f.folder, f.name)), lower(base)), score = score + 5; end
    if contains(lname, 'localfield') || contains(lname, 'mask') || contains(lname, 'mag')
        score = score - 50;
    end
    if score > bestScore
        bestScore = score;
        bestPath = fullfile(f.folder, f.name);
    end
end
if bestScore > 0
    qsmPath = bestPath;
end
end

function names = list_nifti_files(rootDir)
names = {};
try
    d1 = dir(fullfile(rootDir, '*.nii'));
    d2 = dir(fullfile(rootDir, '*.nii.gz'));
    names = [{d1.name}, {d2.name}];
catch
end
if isempty(names)
    names = {'<none>'};
end
end

%% =========================================================================
% Validation / logging helpers
% =========================================================================
function tf = is_valid_volume(vol, Mask)
if isempty(vol) || ~isequal(size(vol), size(Mask))
    tf = false;
    return;
end
v = double(vol(Mask));
v = v(isfinite(v));
tf = ~isempty(v) && any(abs(v) > 1e-12) && std(v) > 1e-12;
end

function validate_qsm_or_error(chi, Mask)
if ~is_valid_volume(chi, Mask)
    error('WH-QSM output is invalid or nearly all zero.');
end
v = double(chi(Mask));
v = v(isfinite(v));
p999 = prctile(abs(v), 99.9);
s = std(v);
if p999 > 10 || s > 5
    error('WH-QSM output is numerically implausible: p99.9(abs)=%.4g ppm, std=%.4g ppm.', p999, s);
elseif p999 > 2 || s > 1
    warning('WH-QSM output is large for brain QSM: p99.9(abs)=%.4g ppm, std=%.4g ppm. Please QC.', p999, s);
end
end

function print_stats(name, vol, Mask)
v = double(vol(Mask));
v = v(isfinite(v));
if isempty(v)
    fprintf('%s: empty / non-finite\n', name);
    return;
end
fprintf('%s: min=%.6g, p1=%.6g, median=%.6g, p99=%.6g, max=%.6g, std=%.6g\n', ...
    name, min(v), prctile(v,1), median(v), prctile(v,99), max(v), std(v));
end

