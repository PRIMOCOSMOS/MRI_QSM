function data = dicom_loader_subject(subject, output_data_dir, varargin)
% dicom_loader_subject.m  (v4 - WH-QSM real-data loader)
% ============================================================================
% Load one Siemens SWI DICOM subject and prepare the minimum reliable input
% needed by WH-QSM:
%   - magnitude image and brain mask
%   - multi-echo phase -> field map in Hz via echo-time fitting
%   - ppm compatibility field for older code paths
%   - actual DICOM echo times / delta_TE / B0 passed downstream
%
% Key v4 changes:
%   1) Two-echo phase is no longer reduced to "last echo".
%      We convert Siemens phase units to radians for each echo, unwrap along
%      echo dimension, and fit phase(TE) slope to obtain fieldmap_Hz.
%   2) If only one echo is present, fallback is explicit and documented:
%      fieldmap_Hz = phase_rad / (2*pi*TE).
%   3) EchoTime, delta_TE, B0, voxel size, and phase-conversion metadata are
%      stored in data and later written into the SEPIA header.
%   4) qsm2016_format output now saves individual variables plus data_full.mat
%      instead of a misleading phs_tissue.mat containing only variable "data".
% ============================================================================

if nargin < 2 || isempty(output_data_dir)
    output_data_dir = fullfile(subject.path, '_qsm2016_format');
end

p = inputParser;
addParameter(p, 'mask_method', 'auto', @(x) ischar(x) || isstring(x));
addParameter(p, 'mask_erode_mm', 1.5, @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x >= 0);
addParameter(p, 'mask_threshold_factor', 0.12, @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x > 0 && x < 1);
addParameter(p, 'bet_fractional_threshold', 0.50, @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x > 0 && x < 1);
addParameter(p, 'bet_vertical_gradient', 0.0, @(x) isnumeric(x) && isscalar(x) && isfinite(x));
parse(p, varargin{:});
mask_method = char(p.Results.mask_method);
mask_erode_mm = p.Results.mask_erode_mm;
mask_threshold_factor = p.Results.mask_threshold_factor;
bet_fractional_threshold = p.Results.bet_fractional_threshold;
bet_vertical_gradient = p.Results.bet_vertical_gradient;

if ~exist(output_data_dir, 'dir')
    mkdir(output_data_dir);
end

fprintf('\n');
fprintf('============================================================\n');
fprintf(' DICOM loader v4 - subject: %s (%s)\n', subject.name, upper(subject.group));
fprintf(' Path: %s\n', subject.path);
fprintf(' Mask method: %s, erosion: %.3g mm, threshold factor: %.3g\n', mask_method, mask_erode_mm, mask_threshold_factor);
fprintf('============================================================\n\n');

%% ------------------------------------------------------------------------
% Step 1: discover DICOM files
% -------------------------------------------------------------------------
fprintf('[1/7] Discovering DICOM files...\n');
file_list = discover_all_dicom(subject.path);
fprintf('  -> %d DICOM files found.\n', numel(file_list));
if isempty(file_list)
    error('No DICOM files found under subject path: %s', subject.path);
end

%% ------------------------------------------------------------------------
% Step 2: group by SeriesInstanceUID
% -------------------------------------------------------------------------
fprintf('\n[2/7] Grouping by SeriesInstanceUID...\n');
series_map = classify_series_simple(file_list);
fprintf('  -> %d series found.\n', series_map.Count);
if series_map.Count == 0
    error('No readable DICOM series found.');
end

%% ------------------------------------------------------------------------
% Step 3: identify T1 / magnitude / phase series
% -------------------------------------------------------------------------
fprintf('\n[3/7] Identifying T1 / Magnitude / Phase series...\n');
[t1_series, mag_series, phase_series] = find_key_series(series_map);
if isempty(phase_series)
    error('No PHASE series found. Need Siemens phase DICOM for WH-QSM.');
end
if isempty(mag_series)
    error('No MAGNITUDE series found. Need magnitude DICOM for mask and SEPIA input.');
end

B0 = detect_B0(phase_series.info, mag_series.info, get_info_or_empty(t1_series));
B0_dir = detect_B0_dir(phase_series.info);
fprintf('  Phase     : Ser#%s, %d files, %s\n', safe_series_no(phase_series.info), numel(phase_series.file_paths), get_desc(phase_series.info));
fprintf('  Magnitude : Ser#%s, %d files, %s\n', safe_series_no(mag_series.info), numel(mag_series.file_paths), get_desc(mag_series.info));
if ~isempty(t1_series)
    fprintf('  T1        : Ser#%s, %d files, %s\n', safe_series_no(t1_series.info), numel(t1_series.file_paths), get_desc(t1_series.info));
else
    fprintf('  T1        : not found (WH-QSM does not require it)\n');
end
fprintf('  B0        : %.4g T\n', B0);
fprintf('  B0 dir    : [%.4g %.4g %.4g]\n', B0_dir(1), B0_dir(2), B0_dir(3));

%% ------------------------------------------------------------------------
% Step 4: load magnitude
% -------------------------------------------------------------------------
fprintf('\n[4/7] Loading magnitude...\n');
[magn_vol, mag_meta] = load_magnitude_volume(mag_series);
fprintf('  Magnitude size: %s\n', mat2str(size(magn_vol)));
fprintf('  Magnitude echo times: %s ms\n', mat2str(mag_meta.echo_times_ms, 6));
if isfield(mag_meta, 'r2star_Hz') && any(mag_meta.r2star_Hz(:) > 0)
    tmp_r2 = mag_meta.r2star_Hz(isfinite(mag_meta.r2star_Hz) & mag_meta.r2star_Hz > 0);
    fprintf('  R2* map from magnitude: median=%.4g Hz, p95=%.4g Hz\n', median(tmp_r2), prctile(tmp_r2,95));
end

%% ------------------------------------------------------------------------
% Step 5: load phase and fit field map
% -------------------------------------------------------------------------
fprintf('\n[5/7] Loading phase and fitting field map...\n');
% 传入逐回波 magnitude 做 SNR(magnitude^2) 加权拟合: 降低长 TE 低 SNR 回波
% (信号衰减到~T2*, 噪声大) 对深部核团场估计的污染。文献标准做法。
mag4d_for_fit = [];
if isfield(mag_meta,'mag4d') && ~isempty(mag_meta.mag4d)
    mag4d_for_fit = mag_meta.mag4d;
end
[fieldmap_Hz, fieldmap_ppm, phase_meta] = load_phase_fieldmap(phase_series, B0, mag4d_for_fit);
fprintf('  Phase array size       : %s\n', mat2str(size(phase_meta.phase_rad_4d)));
fprintf('  Phase echo times       : %s ms\n', mat2str(phase_meta.echo_times_ms, 6));
fprintf('  Field fitting method   : %s\n', phase_meta.fit_method);
fprintf('  Fieldmap Hz range      : [%.6g, %.6g] Hz\n', min(fieldmap_Hz(:)), max(fieldmap_Hz(:)));
fprintf('  Fieldmap ppm range     : [%.6g, %.6g] ppm\n', min(fieldmap_ppm(:)), max(fieldmap_ppm(:)));

%% ------------------------------------------------------------------------
% Step 6: load T1 if available
% -------------------------------------------------------------------------
fprintf('\n[6/7] Loading T1 / structural image...\n');
if ~isempty(t1_series)
    try
        t1_vol = load_t1_volume(t1_series);
        fprintf('  T1 size: %s\n', mat2str(size(t1_vol)));
    catch ME
        warning('T1 loading failed; using zeros. Reason: %s', ME.message);
        t1_vol = zeros(size(magn_vol), 'double');
    end
else
    t1_vol = zeros(size(magn_vol), 'double');
end

%% ------------------------------------------------------------------------
% Step 7: build mask, assemble data, save variables
% -------------------------------------------------------------------------
fprintf('\n[7/7] Building mask and assembling data...\n');

if ~isequal(size(fieldmap_Hz), size(magn_vol))
    error('Phase fieldmap size %s does not match magnitude size %s. Resampling is intentionally not done in the WH-QSM loader.', ...
        mat2str(size(fieldmap_Hz)), mat2str(size(magn_vol)));
end

if ~isequal(size(t1_vol), size(magn_vol))
    fprintf('  T1 size %s -> nearest-neighbour resize to %s for storage only.\n', mat2str(size(t1_vol)), mat2str(size(magn_vol)));
    t1_vol = resize_volume_nn(t1_vol, size(magn_vol));
end

spatial_res = get_spatial_res(phase_series.info);
if any(~isfinite(spatial_res)) || any(spatial_res <= 0)
    spatial_res = get_spatial_res(mag_series.info);
end
if any(~isfinite(spatial_res)) || any(spatial_res <= 0)
    error('Could not determine valid voxel size from DICOM metadata.');
end
fprintf('  Voxel size: [%.6g %.6g %.6g] mm\n', spatial_res(1), spatial_res(2), spatial_res(3));

mask = generate_brain_mask(magn_vol, spatial_res, mask_erode_mm, mask_threshold_factor, ...
    mask_method, bet_fractional_threshold, bet_vertical_gradient);
voxel_volume_ml = prod(spatial_res) / 1000;
fprintf('  Brain mask: %d voxels (%.2f%% of volume, %.1f mL)\n', ...
    nnz(mask), 100*nnz(mask)/numel(mask), nnz(mask)*voxel_volume_ml);

fieldmap_Hz(~mask) = 0;
fieldmap_ppm(~mask) = 0;
phase_last = phase_meta.phase_unwrapped_4d(:,:,:,end);
phase_last(~mask) = 0;

TE_sec = phase_meta.echo_times_ms(:).' / 1000;
delta_TE = compute_delta_te(TE_sec);

data = struct();
data.fieldmap_Hz       = double(fieldmap_Hz);
data.local_field_ppm   = double(fieldmap_ppm);
data.R2star_Hz         = double(mag_meta.r2star_Hz);
data.R2star_s0         = double(mag_meta.s0);
data.R2star_fit_residual = double(mag_meta.r2_fit_residual);
data.phs_tissue        = double(fieldmap_ppm);      % compatibility: ppm field used by older modules
% phs_unwrap/phs_wrap are kept for compatibility and QC only. For multi-echo
% data, they refer to the last echo after echo-dimension unwrap.
data.phs_unwrap        = double(phase_last);
data.phs_wrap          = wrap_to_pi(double(phase_meta.phase_rad_4d(:,:,:,end)));
data.phase_rad_4d      = double(phase_meta.phase_rad_4d);
data.phase_scaled_4d   = double(phase_meta.phase_scaled_4d);
data.phase_fit_method  = phase_meta.fit_method;
data.phase_fit_residual_rad = phase_meta.fit_residual_rad;
data.mask_method       = mask_method;
data.mask_erode_mm     = mask_erode_mm;
data.mask_threshold_factor = mask_threshold_factor;
data.bet_fractional_threshold = bet_fractional_threshold;
data.bet_vertical_gradient = bet_vertical_gradient;
data.msk               = logical(mask);
data.Mask              = logical(mask);
data.magn              = double(magn_vol);
data.magn_raw          = double(magn_vol);
data.mp_rage           = double(t1_vol);
data.chi_33            = zeros(size(mask), 'double');
data.chi_cosmos        = zeros(size(mask), 'double');
data.evaluation_mask   = double(mask);
data.spatial_res       = double(spatial_res);
data.N                 = size(mask);
data.EchoTime          = phase_meta.echo_times_ms(:).';      % ms, vector
data.echo_times_ms     = phase_meta.echo_times_ms(:).';
data.echo_times_sec    = TE_sec;
data.TE                = TE_sec;
data.delta_TE          = delta_TE;
data.delta_TE_sec      = delta_TE;
data.B0                = B0;
data.b0                = B0;
data.FieldStrength     = B0;
data.MagneticFieldStrength = B0;
data.B0_dir            = B0_dir;
data.b0dir             = B0_dir;
data.Manufacturer      = char(safe_field_str(phase_series.info, 'Manufacturer', ''));
data.patient_group     = subject.group;
data.subject_name      = subject.name;
data.phase_conversion  = phase_meta.phase_conversion;
data.phase_series_desc = get_desc(phase_series.info);
data.mag_series_desc   = get_desc(mag_series.info);
data.mag_echo_times_ms = mag_meta.echo_times_ms;

save_subject_variables(output_data_dir, data);

fprintf('\nDICOM loading complete. WH-QSM input fieldmap: %s, TE=%s ms, delta_TE=%.6g ms.\n', ...
    data.phase_fit_method, mat2str(data.echo_times_ms, 6), data.delta_TE * 1000);
end

%% =========================================================================
% DICOM discovery / classification
% =========================================================================
function file_list = discover_all_dicom(root_dir)
file_list = {};
seen = containers.Map('KeyType', 'char', 'ValueType', 'logical');
try
    all_files = dir(fullfile(root_dir, '**', '*'));
catch
    all_files = dir(root_dir);
end
all_files = all_files(~[all_files.isdir]);
for k = 1:numel(all_files)
    fp = fullfile(all_files(k).folder, all_files(k).name);
    if isKey(seen, fp), continue; end
    [~, ~, ext] = fileparts(all_files(k).name);
    is_dcm = any(strcmpi(ext, {'.dcm', '.dicom', '.ima', '.001', '.img'}));
    if ~is_dcm
        is_dcm = check_magic(fp);
    end
    if is_dcm
        file_list{end+1} = fp; %#ok<AGROW>
        seen(fp) = true;
    end
end
end

function tf = check_magic(fp)
tf = false;
fid = fopen(fp, 'r');
if fid == -1, return; end
try
    fseek(fid, 128, 'bof');
    magic = fread(fid, 4, 'uint8=>char')';
    tf = strcmp(magic, 'DICM');
catch
    tf = false;
end
fclose(fid);
end

function series_map = classify_series_simple(file_list)
series_map = containers.Map('KeyType', 'char', 'ValueType', 'any');
for k = 1:numel(file_list)
    fp = file_list{k};
    try
        info = dicominfo(fp);
        uid = safe_field_str(info, 'SeriesInstanceUID', '');
        if isempty(uid)
            uid = sprintf('__NO_UID_%06d', k);
        end
        if isKey(series_map, uid)
            s = series_map(uid);
            s.file_paths{end+1} = fp;
            series_map(uid) = s;
        else
            s = struct();
            s.file_paths = {fp};
            s.info = info;
            s.uid = uid;
            series_map(uid) = s;
        end
    catch ME
        fprintf('  Skipping unreadable DICOM %s: %s\n', fp, ME.message);
    end
end
end

function [t1_series, mag_series, phase_series] = find_key_series(series_map)
% Robust series selection for WH-QSM.
%
% Important failure mode fixed here:
%   T1/MPRAGE DICOM ImageType often contains "M" because it is a magnitude
%   image. It must NOT be used as the SWI magnitude paired with phase.
%
% Selection order:
%   1) find phase candidates
%   2) combine/select phase series
%   3) choose magnitude candidate that is geometry-compatible with phase
%   4) keep T1 separately as optional structural image

t1_series = [];
mag_series = [];
phase_series = [];

t1_best = -Inf;
phase_candidates = {};
mag_candidates = {};
keys = series_map.keys;

for k = 1:numel(keys)
    s = series_map(keys{k});
    info = s.info;
    sd = lower(safe_field_str(info, 'SeriesDescription', ''));
    pn = lower(safe_field_str(info, 'ProtocolName', ''));
    it = lower(image_type_to_string(safe_field_any(info, 'ImageType', '')));
    folder_label = lower(get_series_folder_label(s.file_paths));
    nfiles = numel(s.file_paths);
    text_all = [sd ' ' pn ' ' it ' ' folder_label];

    % Folder layout observed in the real dataset:
    %   8_t1_mprage_sag_p2_iso  -> structural T1, never WH-QSM magnitude
    %   14_Mag_Images / 15_Mag_Images -> raw SWI magnitude candidates
    %   16_Pha_Images -> raw SWI phase
    %   17/18_mIP_Images(SW), 19/20_SWI_Images -> postprocessed, reject
    is_structural = contains_any([sd ' ' pn ' ' folder_label], {'mprage','mp-rage','t1'});
    is_projection = contains_any(text_all, {'mip','minip','mnip'});
    is_swi_post   = contains_any([sd ' ' folder_label], {'swi_images','swi images','swi_image','swi post','swi_post'});
    is_raw_mag_folder = contains_any(folder_label, {'mag_images','mag images'});
    is_raw_pha_folder = contains_any(folder_label, {'pha_images','pha images','phase_images','phase images'});

    % T1 candidate
    t1_score = 0;
    if is_structural, t1_score = t1_score + 30; end
    if contains_any(folder_label, {'8_t1','t1_mprage'}), t1_score = t1_score + 20; end
    t1_score = t1_score + min(nfiles/100, 5);

    % Phase candidate. The observed folder 16_Pha_Images is a strong prior.
    phase_score = 0;
    if is_raw_pha_folder, phase_score = phase_score + 80; end
    if contains_any(it, {'\p\','phase'}), phase_score = phase_score + 40; end
    if contains_any(sd, {'phase','pha','ph_','_ph'}), phase_score = phase_score + 10; end
    if is_projection || is_swi_post, phase_score = phase_score - 40; end
    if is_structural, phase_score = phase_score - 80; end
    phase_score = phase_score + min(nfiles/100, 5);

    % SWI magnitude candidate. The observed 14/15_Mag_Images folders are
    % strong priors; T1 and postprocessed SWI/mIP are explicitly rejected.
    mag_score = -Inf;
    if ~is_structural && ~is_projection && ~is_swi_post
        mag_score = 0;
        if is_raw_mag_folder, mag_score = mag_score + 90; end
        if contains_any(it, {'\m\','magnitude'}), mag_score = mag_score + 35; end
        if contains_any(sd, {'mag','magnitude','mag_images'}), mag_score = mag_score + 25; end
        if contains_any(text_all, {'norm'}), mag_score = mag_score - 5; end
        mag_score = mag_score + min(nfiles/100, 5);
    end

    s.score_phase = phase_score;
    s.score_mag = mag_score;
    s.score_t1 = t1_score;

    if t1_score > t1_best
        t1_best = t1_score;
        t1_series = s;
    end
    if phase_score >= 20
        phase_candidates{end+1} = s; %#ok<AGROW>
    end
    if isfinite(mag_score) && mag_score >= 20
        mag_candidates{end+1} = s; %#ok<AGROW>
    end
end

phase_series = select_phase_series(phase_candidates);
mag_series = select_magnitude_series(mag_candidates, phase_series);

if t1_best < 15
    t1_series = [];
end
end

function phase_series = select_phase_series(phase_candidates)
phase_series = [];
if isempty(phase_candidates)
    return;
end
scores = cellfun(@(s) s.score_phase, phase_candidates);
ser_nums = cellfun(@(s) safe_dicom_num(s.info, 'SeriesNumber', 9999), phase_candidates);
[~, ord] = sortrows([-scores(:), ser_nums(:)], [1 2]);
phase_candidates = phase_candidates(ord);

% Use the best phase candidate as geometry reference. If phase echoes are
% split across compatible SeriesInstanceUIDs, combine compatible candidates.
phase_series = phase_candidates{1};
phase_series.file_paths = {};
phase_series.combined_uids = {};
ref = phase_candidates{1};
for i = 1:numel(phase_candidates)
    s = phase_candidates{i};
    if is_compatible_series(ref.info, s.info)
        phase_series.file_paths = [phase_series.file_paths, s.file_paths]; %#ok<AGROW>
        if isfield(s, 'uid'), phase_series.combined_uids{end+1} = s.uid; end %#ok<AGROW>
    else
        fprintf('  Skipping PHASE candidate Ser#%s as geometry differs from selected phase series.\n', safe_series_no(s.info));
    end
end
fprintf('  Selected PHASE series: Ser#%s, %d compatible UID(s), %d files.\n', ...
    safe_series_no(phase_series.info), max(1,numel(phase_series.combined_uids)), numel(phase_series.file_paths));
end

function mag_series = select_magnitude_series(mag_candidates, phase_series)
mag_series = [];
if isempty(mag_candidates)
    return;
end

% The WH-QSM magnitude must match the phase geometry. This prevents selecting
% T1/MPRAGE as magnitude (e.g. [256 256 192]) when phase is SWI geometry
% (e.g. [336 384 104]).
compatible = false(1, numel(mag_candidates));
for i = 1:numel(mag_candidates)
    if ~isempty(phase_series)
        compatible(i) = is_compatible_series(phase_series.info, mag_candidates{i}.info);
    else
        compatible(i) = true;
    end
end

if any(compatible)
    cand = mag_candidates(compatible);
else
    fprintf('  WARNING: no MAGNITUDE candidate matches phase geometry; using highest-score magnitude candidate.\n');
    cand = mag_candidates;
end

scores = cellfun(@(s) s.score_mag, cand);
ser_nums = cellfun(@(s) safe_dicom_num(s.info, 'SeriesNumber', 9999), cand);
[~, ord] = sortrows([-scores(:), ser_nums(:)], [1 2]);
cand = cand(ord);
mag_series = cand{1};

% Do not blindly merge all magnitude-like series: SWI postprocessed images can
% share geometry but are not raw echoes. The selected best series is enough for
% mask and SEPIA magnitude input; if it contains multiple echoes, build_4d_volume
% will average them.
fprintf('  Selected MAGNITUDE series: Ser#%s, score %.2f, %d files.\n', ...
    safe_series_no(mag_series.info), mag_series.score_mag, numel(mag_series.file_paths));

if ~isempty(phase_series) && ~is_compatible_series(phase_series.info, mag_series.info)
    fprintf('  WARNING: selected magnitude geometry still differs from phase; downstream size check may fail.\n');
end
end

function combined = combine_compatible_series(candidates, label)
combined = [];
if isempty(candidates)
    return;
end

% Sort candidates by SeriesNumber for deterministic echo ordering when echoes
% are stored as separate SeriesInstanceUIDs.
ser_nums = zeros(1, numel(candidates));
for i = 1:numel(candidates)
    ser_nums(i) = safe_dicom_num(candidates{i}.info, 'SeriesNumber', i);
end
[~, ord] = sort(ser_nums);
candidates = candidates(ord);

ref = candidates{1};
combined = ref;
combined.file_paths = {};
combined.combined_uids = {};

for i = 1:numel(candidates)
    s = candidates{i};
    if is_compatible_series(ref.info, s.info)
        combined.file_paths = [combined.file_paths, s.file_paths]; %#ok<AGROW>
        if isfield(s, 'uid'), combined.combined_uids{end+1} = s.uid; end %#ok<AGROW>
    else
        fprintf('  Skipping %s candidate Ser#%s as geometry differs from selected series.\n', ...
            label, safe_series_no(s.info));
    end
end

if isempty(combined.file_paths)
    combined = [];
else
    fprintf('  Combined %s series: %d compatible UID(s), %d files.\n', ...
        label, max(1, numel(combined.combined_uids)), numel(combined.file_paths));
end
end

function tf = is_compatible_series(info_ref, info)
rows_ref = safe_dicom_num(info_ref, 'Rows', NaN);
cols_ref = safe_dicom_num(info_ref, 'Columns', NaN);
rows = safe_dicom_num(info, 'Rows', NaN);
cols = safe_dicom_num(info, 'Columns', NaN);
if rows_ref ~= rows || cols_ref ~= cols
    tf = false;
    return;
end
sr_ref = get_spatial_res(info_ref);
sr = get_spatial_res(info);
if all(isfinite(sr_ref)) && all(isfinite(sr))
    tf = max(abs(sr_ref - sr)) < 1e-3;
else
    tf = true;
end
end

%% =========================================================================
% Volume loaders
% =========================================================================
function [vol, meta] = load_magnitude_volume(series)
[vol4d, echo_times_ms, echo_numbers] = build_4d_volume(series.file_paths);
if isempty(vol4d)
    error('Magnitude volume is empty.');
end
vol = mean(vol4d, 4);
[r2star_Hz, s0, r2_fit_residual] = compute_r2star_from_magnitude(vol4d, echo_times_ms);
meta = struct('echo_times_ms', echo_times_ms, ...
    'echo_numbers', echo_numbers, ...
    'n_echoes', size(vol4d,4), ...
    'r2star_Hz', r2star_Hz, ...
    's0', s0, ...
    'r2_fit_residual', r2_fit_residual, ...
    'mag4d', vol4d);   % 逐回波 magnitude, 供相位拟合做 SNR 加权(magnitude^2)
end

function [r2star_Hz, s0, residual] = compute_r2star_from_magnitude(mag4d, echo_times_ms)
% Compute R2* from multi-echo magnitude by log-linear fitting:
%   S(TE) = S0 * exp(-R2* TE)
% This feature map is required by susceptibility source separation methods.
TE_sec = double(echo_times_ms(:).') / 1000;
mag4d = double(mag4d);
mag4d(~isfinite(mag4d)) = 0;
if numel(TE_sec) < 2 || size(mag4d,4) < 2 || any(~isfinite(TE_sec))
    r2star_Hz = zeros(size(mag4d,1), size(mag4d,2), size(mag4d,3));
    s0 = mag4d(:,:,:,1);
    residual = zeros(size(s0));
    return;
end
[TE_sec, ord] = sort(TE_sec);
mag4d = mag4d(:,:,:,ord);
logS = log(max(mag4d, eps));
t = reshape(TE_sec, [1 1 1 numel(TE_sec)]);
t0 = mean(TE_sec);
tc = t - t0;
denom = sum((TE_sec - t0).^2);
logS_mean = mean(logS, 4);
slope = sum((logS - logS_mean) .* tc, 4) ./ max(denom, eps);
intercept = logS_mean - slope .* t0;
r2star_Hz = max(-slope, 0);
s0 = exp(intercept);
pred = intercept + slope .* t;
residual = sqrt(mean((logS - pred).^2, 4));
r2star_Hz(~isfinite(r2star_Hz)) = 0;
s0(~isfinite(s0)) = 0;
residual(~isfinite(residual)) = 0;
end

function [fieldmap_Hz, fieldmap_ppm, meta] = load_phase_fieldmap(series, B0, mag4d)
if nargin < 3, mag4d = []; end
[phase_scaled_4d, echo_times_ms, echo_numbers] = build_4d_volume(series.file_paths);
if isempty(phase_scaled_4d)
    error('Phase volume is empty.');
end

[phase_rad_4d, conv] = convert_phase_to_rad(phase_scaled_4d, series.info);
TE_sec = double(echo_times_ms(:).') / 1000;
if any(~isfinite(TE_sec)) || any(TE_sec <= 0)
    error('Invalid or missing EchoTime in phase DICOM series: %s ms', mat2str(echo_times_ms));
end

[TE_sec, order] = sort(TE_sec);
phase_rad_4d = phase_rad_4d(:,:,:,order);
phase_scaled_4d = phase_scaled_4d(:,:,:,order);
echo_times_ms = echo_times_ms(order);
echo_numbers = echo_numbers(order);

nEcho = numel(TE_sec);
% Do NOT call MATLAB/third-party unwrap() here. After SEPIA/FANSI is added to
% the path, some toolboxes may shadow MATLAB's unwrap with a different
% signature, which caused the second subject to fail with "too many input
% arguments". Echo-dimension unwrapping is simple and deterministic, so use a
% local implementation.
phase_unwrapped_4d = unwrap_echo_phase_local(phase_rad_4d);

% ============================================================================
% 多回波场图估计 (按文献最佳实践排序选择, 优先调用工具箱成熟算法)
% 依据: Mancini et al. MRM 2022 "Multi-echo QSM: how to combine echoes" —
%   推荐【非线性复数拟合(NLFit)】在 Laplacian 处理前合并多回波, 深部灰质/静脉
%   精度最高、噪声传播最低; 优于 SNR加权平均 > 简单平均 > 无权线性拟合。
%   工具箱实现: MEDI 的 Fit_ppm_complex (Cornell QSM, 文献 NLFit 所用)。
% 优先级: (1) MEDI Fit_ppm_complex (NLFit) -> (2) magnitude^2 加权线性 -> (3) 无权线性
% ============================================================================
fieldmap_Hz = [];
residual = [];
fit_method = '';

% ---- (1) 优先: MEDI Fit_ppm_complex 非线性复数拟合 ----
% 仅在【等回波间隔】时使用(Fit_ppm_complex 假设等间隔, 返回相位/echo)。
% 结果会做合理性校验(与加权线性拟合量级比对), 不合理则丢弃回退, 确保安全。
dTE_all = diff(sort(TE_sec));
equalSpacing = (numel(dTE_all)>=1) && (max(dTE_all)-min(dTE_all) < 0.1*median(dTE_all));
if nEcho >= 2 && equalSpacing && ~isempty(mag4d) ...
        && isequal(size(mag4d), size(phase_rad_4d)) ...
        && exist('Fit_ppm_complex', 'file') == 2
    try
        cplx = double(mag4d) .* exp(1i * double(phase_rad_4d));
        % MEDI 接口: [p1, dp1, relres, p0] = Fit_ppm_complex(M)
        % p1 = 每回波相位增量(rad/echo)。用 nargout 容错不同版本。
        try
            [p1, ~, relres] = Fit_ppm_complex(cplx);
        catch
            p1 = Fit_ppm_complex(cplx); relres = [];
        end
        dTE = median(dTE_all);
        if ~(isfinite(dTE) && dTE > 0), dTE = TE_sec(2)-TE_sec(1); end
        cand_Hz = double(p1) ./ (2*pi*dTE);   % rad/echo -> Hz
        cand_Hz(~isfinite(cand_Hz)) = 0;
        % 合理性校验: 脑内 std 应在生理范围 (0.5~20 Hz), 否则视为接口/尺度不符
        v = cand_Hz(abs(cand_Hz)<200);
        if ~isempty(v) && std(v(:))>0.3 && std(v(:))<30
            fieldmap_Hz = cand_Hz;
            if ~isempty(relres), residual = double(relres); else, residual = zeros(size(cand_Hz)); end
            fit_method = sprintf('MEDI_Fit_ppm_complex_NLFit_%decho', nEcho);
            fprintf('  [场图] 使用 MEDI Fit_ppm_complex 非线性复数拟合 (文献推荐).\n');
        else
            fprintf('  [场图] Fit_ppm_complex 输出量级异常(std=%.3g), 回退加权线性拟合.\n', std(v(:)));
            fieldmap_Hz = [];
        end
    catch ME_nlfit
        fprintf('  [场图] Fit_ppm_complex 失败(%s), 回退加权线性拟合.\n', ME_nlfit.message);
        fieldmap_Hz = [];
    end
elseif nEcho >= 2 && ~equalSpacing
    fprintf('  [场图] 回波非等间隔, 跳过 Fit_ppm_complex, 用加权线性拟合.\n');
end

if isempty(fieldmap_Hz) && nEcho >= 2 && numel(unique(TE_sec)) >= 2
    t = reshape(TE_sec, [1 1 1 nEcho]);
    % --- (2)/(3) magnitude^2 加权线性拟合 (NLFit 不可用时的回退) ---
    % 权重 w = magnitude^2 (近似 SNR^2)。长 TE 低 SNR 回波权重自动变小,
    % 减少其对深部核团(T2*短)场估计的噪声污染。无 magnitude 时退化为等权。
    useW = ~isempty(mag4d) && isequal(size(mag4d), size(phase_unwrapped_4d));
    if useW
        w = double(mag4d).^2;
        w(~isfinite(w)) = 0;
        sw = sum(w, 4); sw(sw<=0) = eps;
        tw = sum(w .* t, 4) ./ sw;                 % 加权 TE 均值(逐体素)
        pw = sum(w .* phase_unwrapped_4d, 4) ./ sw; % 加权相位均值
        num = sum(w .* (phase_unwrapped_4d - pw) .* (t - tw), 4);
        den = sum(w .* (t - tw).^2, 4); den(den<=0) = eps;
        slope_rad_per_sec = num ./ den;
        intercept = pw - slope_rad_per_sec .* tw;
        pred = intercept + slope_rad_per_sec .* t;
        residual = sqrt(sum(w.*(phase_unwrapped_4d - pred).^2,4) ./ sw);
        fit_method = sprintf('multi_echo_MAGWEIGHTED_phase_fit_%decho', nEcho);
    else
        t0 = mean(TE_sec);
        tc = t - t0;
        denom = sum((TE_sec - t0).^2);
        phase_mean = mean(phase_unwrapped_4d, 4);
        slope_rad_per_sec = sum((phase_unwrapped_4d - phase_mean) .* tc, 4) ./ max(denom, eps);
        intercept = phase_mean - slope_rad_per_sec .* t0;
        pred = intercept + slope_rad_per_sec .* t;
        residual = sqrt(mean((phase_unwrapped_4d - pred).^2, 4));
        fit_method = sprintf('multi_echo_linear_phase_fit_%decho', nEcho);
    end
    fieldmap_Hz = slope_rad_per_sec ./ (2*pi);
end

% 单回波(或多回波拟合都未产出)兜底
if isempty(fieldmap_Hz)
    fieldmap_Hz = phase_unwrapped_4d(:,:,:,end) ./ (2*pi*TE_sec(end));
    residual = zeros(size(fieldmap_Hz));
    fit_method = 'single_echo_phase_over_TE_fallback';
end

gyro_MHz_per_T = 42.57747892;
fieldmap_ppm = fieldmap_Hz ./ (gyro_MHz_per_T * B0);

meta = struct();
meta.phase_scaled_4d = phase_scaled_4d;
meta.phase_rad_4d = phase_rad_4d;
meta.phase_unwrapped_4d = phase_unwrapped_4d;
meta.echo_times_ms = echo_times_ms(:).';
meta.echo_numbers = echo_numbers(:).';
meta.fit_method = fit_method;
meta.fit_residual_rad = residual;
meta.phase_conversion = conv;
end

function ph_unwrapped = unwrap_echo_phase_local(ph)
% Local unwrap along 4th dimension. This avoids calling any path-dependent
% unwrap() implementation. Adjacent echo phase differences are mapped to
% (-pi, pi], then accumulated.
ph = double(ph);
ph_unwrapped = ph;
if ndims(ph) < 4 || size(ph,4) <= 1
    return;
end
for e = 2:size(ph,4)
    d = ph(:,:,:,e) - ph(:,:,:,e-1);
    d_wrapped = mod(d + pi, 2*pi) - pi;
    ph_unwrapped(:,:,:,e) = ph_unwrapped(:,:,:,e-1) + d_wrapped;
end
end

function vol = load_t1_volume(series)
[vol4d, ~, ~] = build_4d_volume(series.file_paths);
if isempty(vol4d)
    vol = [];
else
    vol = vol4d(:,:,:,1);
end
end

function [vol4d, echo_times_ms, echo_numbers] = build_4d_volume(files)
% Build [row, col, slice, echo] volume. Echo grouping prioritises EchoTime;
% slice ordering prioritises ImagePositionPatient/SliceLocation over InstanceNumber.

n = numel(files);
if n == 0
    vol4d = [];
    echo_times_ms = [];
    echo_numbers = [];
    return;
end

infos = cell(n,1);
echo_time = nan(n,1);
echo_num = nan(n,1);
inst_num = nan(n,1);
slice_pos = nan(n,1);

for k = 1:n
    infos{k} = dicominfo(files{k});
    echo_time(k) = safe_dicom_num(infos{k}, 'EchoTime', NaN);
    echo_num(k) = safe_dicom_num(infos{k}, 'EchoNumber', NaN);
    if isnan(echo_num(k)), echo_num(k) = safe_dicom_num(infos{k}, 'EchoNumbers', NaN); end
    inst_num(k) = safe_dicom_num(infos{k}, 'InstanceNumber', k);
    slice_pos(k) = get_slice_position(infos{k}, inst_num(k));
end

finite_te = echo_time(isfinite(echo_time));
if numel(unique(round(finite_te*1000)/1000)) >= 2
    echo_key = round(echo_time * 1000) / 1000;  % ms rounded to 1 us
elseif any(isfinite(echo_num))
    echo_key = echo_num;
else
    echo_key = ones(n,1);
end
if all(~isfinite(echo_key))
    echo_key = ones(n,1);
else
    echo_key(~isfinite(echo_key)) = 1;
end

unique_echo_keys = unique(echo_key(isfinite(echo_key)));
unique_echo_keys = sort(unique_echo_keys(:).');
n_echoes = numel(unique_echo_keys);
if n_echoes == 0
    unique_echo_keys = 1;
    n_echoes = 1;
end

sample = dicomread(files{1});
sz = size(sample);
if numel(sz) < 2
    error('DICOM pixel data is not 2D: %s', files{1});
end
rows = sz(1); cols = sz(2);

idx_by_echo = cell(1, n_echoes);
n_slices = 0;
for e = 1:n_echoes
    idx = find(echo_key == unique_echo_keys(e));
    if isempty(idx) && n_echoes == 1
        idx = 1:n;
    end
    if all(isfinite(slice_pos(idx)))
        [~, ord] = sort(slice_pos(idx), 'ascend');
    else
        [~, ord] = sort(inst_num(idx), 'ascend');
    end
    idx_by_echo{e} = idx(ord);
    n_slices = max(n_slices, numel(idx));
end

if n_slices == 0
    error('No slices found while building 4D volume.');
end

vol4d = zeros(rows, cols, n_slices, n_echoes, 'double');
echo_times_ms = nan(1, n_echoes);
echo_numbers = nan(1, n_echoes);

for e = 1:n_echoes
    idx = idx_by_echo{e};
    echo_times_ms(e) = nanmedian_local(echo_time(idx));
    echo_numbers(e) = nanmedian_local(echo_num(idx));
    if isnan(echo_numbers(e)), echo_numbers(e) = e; end
    if isnan(echo_times_ms(e))
        echo_times_ms(e) = safe_dicom_num(infos{idx(1)}, 'EchoTime', NaN);
    end
    if numel(idx) ~= n_slices
        warning('Echo %d has %d slices, while max slices is %d. Missing slices will be zero.', e, numel(idx), n_slices);
    end
    for s = 1:numel(idx)
        k = idx(s);
        Xraw = dicomread(files{k});
        X = double(Xraw);
        X = maybe_convert_signed(X, Xraw, infos{k});
        slope = safe_dicom_num(infos{k}, 'RescaleSlope', 1);
        intercept = safe_dicom_num(infos{k}, 'RescaleIntercept', 0);
        if ~isfinite(slope) || slope == 0, slope = 1; end
        if ~isfinite(intercept), intercept = 0; end
        X = X * slope + intercept;
        vol4d(:,:,s,e) = X;
    end
end
end

function X = maybe_convert_signed(X, Xraw, info)
pix_rep = safe_dicom_num(info, 'PixelRepresentation', 0);
bits = safe_dicom_num(info, 'BitsStored', 16);
if pix_rep == 1 && isa(Xraw, 'uint16') && isfinite(bits) && bits > 0 && bits < 16
    cutoff = 2^(bits-1);
    fullscale = 2^bits;
    idx = X >= cutoff;
    X(idx) = X(idx) - fullscale;
end
end

function val = nanmedian_local(x)
x = x(isfinite(x));
if isempty(x)
    val = NaN;
else
    val = median(x);
end
end

function [phase_rad, conv] = convert_phase_to_rad(phase_scaled, info)
maxAbs = max(abs(phase_scaled(:)));
bits = safe_dicom_num(info, 'BitsStored', 12);
if ~isfinite(bits) || bits <= 0
    bits = 12;
end

conv = struct();
conv.input_max_abs = maxAbs;
conv.bits_stored = bits;

if maxAbs <= pi + 0.2
    phase_rad = phase_scaled;
    conv.method = 'already_radians';
    conv.scale_to_rad = 1;
elseif maxAbs <= 2*pi + 0.5
    phase_rad = phase_scaled;
    conv.method = 'already_radians_wide_range';
    conv.scale_to_rad = 1;
else
    % Siemens common convention after RescaleSlope/Intercept:
    % scaled phase is approximately [-4096, 4094] for 12-bit data.
    denom = 2^bits;
    if denom < 1024 || denom > 65536
        denom = 4096;
    end
    phase_rad = phase_scaled * (pi / denom);
    conv.method = sprintf('siemens_internal_units_times_pi_over_%d', denom);
    conv.scale_to_rad = pi / denom;
end

fprintf('  Phase conversion: %s (scale %.9g), input max abs %.6g -> rad max abs %.6g\n', ...
    conv.method, conv.scale_to_rad, maxAbs, max(abs(phase_rad(:))));
end

%% =========================================================================
% Metadata helpers
% =========================================================================
function B0 = detect_B0(varargin)
B0 = NaN;
fields = {'MagneticFieldStrength','FieldStrength','ImagingFrequency'};
for k = 1:nargin
    info = varargin{k};
    if isempty(info), continue; end
    for f = 1:numel(fields)
        if isfield(info, fields{f}) && ~isempty(info.(fields{f}))
            v = scalarize(info.(fields{f}));
            if strcmp(fields{f}, 'ImagingFrequency')
                v = v / 42.57747892;  % MHz -> Tesla
            end
            if isfinite(v) && v > 0 && v < 20
                B0 = v;
                return;
            end
        end
    end
end
if isnan(B0)
    B0 = 3;
    warning('Could not detect B0 from DICOM; using fallback B0=3 T.');
end
end

function B0_dir = detect_B0_dir(info)
% Full oblique handling is outside this project. We keep [0 0 1] but store it
% explicitly so downstream SEPIA receives a defined direction.
B0_dir = [0 0 1];
if isfield(info, 'B0_dir') && isnumeric(info.B0_dir) && numel(info.B0_dir) >= 3
    tmp = double(info.B0_dir(1:3));
    if all(isfinite(tmp)) && norm(tmp) > 0
        B0_dir = tmp(:).' ./ norm(tmp);
    end
end
end

function spatial_res = get_spatial_res(info)
ps = safe_field_any(info, 'PixelSpacing', [NaN NaN]);
if isnumeric(ps) && numel(ps) >= 2
    dy = double(ps(1));
    dx = double(ps(2));
else
    dx = NaN; dy = NaN;
end
if isfield(info, 'SpacingBetweenSlices') && ~isempty(info.SpacingBetweenSlices)
    dz = scalarize(info.SpacingBetweenSlices);
else
    dz = safe_dicom_num(info, 'SliceThickness', NaN);
end
spatial_res = [dx dy dz];
end

function pos = get_slice_position(info, default)
pos = default;
if isfield(info, 'ImagePositionPatient') && isnumeric(info.ImagePositionPatient) && numel(info.ImagePositionPatient) >= 3
    pos = double(info.ImagePositionPatient(3));
elseif isfield(info, 'SliceLocation') && ~isempty(info.SliceLocation)
    pos = scalarize(info.SliceLocation);
end
if ~isfinite(pos)
    pos = default;
end
end

function v = safe_dicom_num(info, fname, default)
v = default;
if ~isfield(info, fname) || isempty(info.(fname)), return; end
v = scalarize(info.(fname));
if ~isfinite(v), v = default; end
end

function v = scalarize(x)
if isnumeric(x) && ~isempty(x)
    v = double(x(1));
elseif iscell(x) && ~isempty(x)
    v = scalarize(x{1});
elseif ischar(x) && ~isempty(x)
    v = str2double(x);
elseif isstring(x) && ~isempty(x)
    v = str2double(char(x(1)));
else
    v = NaN;
end
end

function val = safe_field_any(info, fname, default)
if isfield(info, fname) && ~isempty(info.(fname))
    val = info.(fname);
else
    val = default;
end
end

function v = safe_field_str(info, fname, default)
v = default;
if ~isfield(info, fname) || isempty(info.(fname)), return; end
val = info.(fname);
if isstruct(val)
    parts = {};
    fns = fieldnames(val);
    for i = 1:numel(fns)
        item = val.(fns{i});
        if ischar(item) && ~isempty(item)
            parts{end+1} = strtrim(item); %#ok<AGROW>
        elseif isstring(item) && strlength(item) > 0
            parts{end+1} = strtrim(char(item)); %#ok<AGROW>
        end
    end
    if ~isempty(parts), v = strjoin(parts, ' '); end
elseif ischar(val)
    v = strtrim(val);
elseif isstring(val)
    v = strtrim(char(val));
elseif iscell(val) && ~isempty(val)
    v = safe_field_str(struct('x', val{1}), 'x', default);
elseif isnumeric(val) && isscalar(val)
    v = num2str(val);
end
end

function s = image_type_to_string(v)
if ischar(v)
    s = v;
elseif isstring(v)
    s = char(join(v, '\'));
elseif iscell(v)
    parts = cell(size(v));
    for i = 1:numel(v)
        if ischar(v{i}), parts{i} = v{i}; else, parts{i} = ''; end
    end
    s = strjoin(parts, '\');
else
    s = '';
end
end

function tf = contains_any(str, patterns)
tf = false;
for i = 1:numel(patterns)
    if contains(str, patterns{i})
        tf = true;
        return;
    end
end
end

function out = get_info_or_empty(series)
if isempty(series), out = []; else, out = series.info; end
end

function label = get_series_folder_label(file_paths)
label = '';
if isempty(file_paths)
    return;
end
try
    folder = fileparts(file_paths{1});
    [~, label] = fileparts(folder);
catch
    label = '';
end
end

function s = safe_series_no(info)
v = safe_dicom_num(info, 'SeriesNumber', NaN);
if isnan(v), s = '?'; else, s = sprintf('%g', v); end
end

function s = get_desc(info)
s = safe_field_str(info, 'SeriesDescription', '');
if isempty(s), s = safe_field_str(info, 'ProtocolName', ''); end
if isempty(s), s = '<no description>'; end
end

%% =========================================================================
% Mask / saving helpers
% =========================================================================
function mask = generate_brain_mask(magn, spatial_res, erode_mm, thr_factor, mask_method, bet_f, bet_g)
% Generate an intracranial mask from SWI magnitude.
%
% Priority is to call mature toolbox brain extraction first (MEDI/SEPIA BET).
% If unavailable or invalid, fallback to a conservative magnitude head mask
% with physical erosion.

if nargin < 2 || isempty(spatial_res), spatial_res = [1 1 1]; end
if nargin < 3 || isempty(erode_mm), erode_mm = 1.5; end
if nargin < 4 || isempty(thr_factor), thr_factor = 0.12; end
if nargin < 5 || isempty(mask_method), mask_method = 'auto'; end
if nargin < 6 || isempty(bet_f), bet_f = 0.50; end
if nargin < 7 || isempty(bet_g), bet_g = 0.0; end

magn = double(magn);
magn(~isfinite(magn)) = 0;
v = magn(:);
v = v(isfinite(v));
if isempty(v) || max(v) <= 0
    error('Magnitude is empty; cannot create mask.');
end

mask_method = lower(char(mask_method));
mask = [];
used_method = '';

if any(strcmp(mask_method, {'auto','toolbox_bet','bet'}))
    [mask_bet, bet_info] = try_toolbox_bet_mask(magn, spatial_res, bet_f, bet_g);
    if ~isempty(mask_bet)
        mask = logical(mask_bet);
        used_method = bet_info;
        fprintf('  Brain extraction: %s\n', used_method);
    elseif strcmp(mask_method, 'toolbox_bet') || strcmp(mask_method, 'bet')
        error('Requested toolbox BET mask, but no usable BET implementation was found on MATLAB path.');
    else
        fprintf('  Toolbox BET unavailable/invalid; fallback to threshold+erosion mask.\n');
    end
end

if isempty(mask)
    thr = thr_factor * prctile(v, 99);
    head_mask = magn > thr;
    head_mask = imfill_safe(head_mask);
    head_mask = largest_component(head_mask);
    head_vox = nnz(head_mask);
    head_ml = head_vox * prod(spatial_res) / 1000;
    fprintf('  Initial head mask: %d voxels (%.1f mL), threshold=%.6g\n', head_vox, head_ml, thr);
    mask = head_mask;
    used_method = 'threshold_largest_component';
end

mask = imfill_safe(mask);
mask = largest_component(mask);

if erode_mm > 0
    mask = erode_mask_mm(mask, spatial_res, erode_mm);
    mask = imfill_safe(mask);
    mask = largest_component(mask);
    fprintf('  Final mask edge peel: %.3g mm.\n', erode_mm);
else
    fprintf('  Final mask edge peel disabled.\n');
end

mask = logical(mask);
if nnz(mask) == 0
    error('Generated brain mask is empty.');
end

mask_ml = nnz(mask) * prod(spatial_res) / 1000;
fprintf('  Final WH-QSM mask method: %s\n', used_method);
fprintf('  Final WH-QSM mask volume: %.1f mL\n', mask_ml);

if mask_ml > 1800
    warning('Mask volume is large (%.1f mL). Skull/scalp may still be included; consider stronger BET or increasing mask_erode_mm.', mask_ml);
elseif mask_ml < 600
    warning('Mask volume is small (%.1f mL). Cortex may be over-eroded; consider decreasing mask_erode_mm.', mask_ml);
end
end

function [mask, info] = try_toolbox_bet_mask(magn, spatial_res, bet_f, bet_g)
mask = [];
info = '';
if exist('BET', 'file') ~= 2
    return;
end

matrix_size = size(magn);
voxel_size = double(spatial_res(:).');
func = str2func('BET');
call_list = {
    {@() func(magn, matrix_size, voxel_size, bet_f, bet_g), 'BET(mag,matrix_size,voxel_size,f,g)'}, ...
    {@() func(magn, matrix_size, voxel_size, bet_f),        'BET(mag,matrix_size,voxel_size,f)'}, ...
    {@() func(magn, matrix_size, voxel_size),               'BET(mag,matrix_size,voxel_size)'}, ...
    {@() func(magn),                                       'BET(mag)'} ...
    };

for i = 1:numel(call_list)
    try
        out = call_list{i}{1}();
        candidate = parse_bet_output(out);
        if is_valid_mask_candidate(candidate, size(magn))
            mask = logical(candidate);
            mask = imfill_safe(mask);
            mask = largest_component(mask);
            info = call_list{i}{2};
            return;
        end
    catch
        % Try next known signature.
    end
end
end

function mask = parse_bet_output(out)
mask = [];
if isnumeric(out) || islogical(out)
    mask = out;
elseif isstruct(out)
    fields = {'Mask','mask','BrainMask','brain_mask','msk'};
    for i = 1:numel(fields)
        if isfield(out, fields{i})
            mask = out.(fields{i});
            return;
        end
    end
end
end

function tf = is_valid_mask_candidate(mask, target_size)
tf = false;
if isempty(mask) || ~isequal(size(mask), target_size)
    return;
end
mask = logical(mask);
frac = nnz(mask) / numel(mask);
% Loose validity range: avoid all-head/all-empty failures.
tf = frac > 0.05 && frac < 0.80;
end

function out = imfill_safe(mask)
try
    out = imfill(logical(mask), 'holes');
catch
    out = logical(mask);
end
end

function out = erode_mask_mm(mask, spatial_res, erode_mm)
spatial_res = double(spatial_res(:).');
rx = max(1, ceil(erode_mm / spatial_res(1)));
ry = max(1, ceil(erode_mm / spatial_res(2)));
rz = max(1, ceil(erode_mm / spatial_res(3)));
[x, y, z] = ndgrid(-rx:rx, -ry:ry, -rz:rz);
se = (x*spatial_res(1)).^2 + (y*spatial_res(2)).^2 + (z*spatial_res(3)).^2 <= erode_mm^2;
out = imerode(logical(mask), se);
end

function mask = largest_component(mask)
mask = logical(mask);
try
    CC = bwconncomp(mask, 6);
    if CC.NumObjects > 1
        sizes = cellfun(@numel, CC.PixelIdxList);
        [~, idx_max] = max(sizes);
        tmp = false(size(mask));
        tmp(CC.PixelIdxList{idx_max}) = true;
        mask = tmp;
    end
catch
    % Fallback: keep mask as-is if bwconncomp is unavailable for any reason.
end
end

function out = resize_volume_nn(vol, target_size)
vol = double(vol);
sz = size(vol);
if numel(sz) < 3, sz(3) = 1; end
idx = cell(1,3);
for d = 1:3
    if sz(d) == target_size(d)
        idx{d} = 1:sz(d);
    else
        idx{d} = max(1, min(sz(d), round(linspace(1, sz(d), target_size(d)))));
    end
end
out = vol(idx{1}, idx{2}, idx{3});
end

function p = wrap_to_pi(x)
p = mod(x + pi, 2*pi) - pi;
end

function delta_TE = compute_delta_te(TE_sec)
if numel(TE_sec) >= 2
    d = diff(sort(TE_sec));
    d = d(isfinite(d) & d > 0);
    if ~isempty(d)
        delta_TE = median(d);
        return;
    end
end
if numel(TE_sec) == 1
    delta_TE = TE_sec(1);
else
    delta_TE = 0.025;
end
end

function save_subject_variables(output_data_dir, data)
if ~exist(output_data_dir, 'dir'), mkdir(output_data_dir); end

phs_tissue = data.phs_tissue; %#ok<NASGU>
phs_unwrap = data.phs_unwrap; %#ok<NASGU>
phs_wrap = data.phs_wrap; %#ok<NASGU>
fieldmap_Hz = data.fieldmap_Hz; %#ok<NASGU>
local_field_ppm = data.local_field_ppm; %#ok<NASGU>
R2star_Hz = data.R2star_Hz; %#ok<NASGU>
R2star_s0 = data.R2star_s0; %#ok<NASGU>
R2star_fit_residual = data.R2star_fit_residual; %#ok<NASGU>
msk = data.msk; %#ok<NASGU>
Mask = data.Mask; %#ok<NASGU>
magn = data.magn; %#ok<NASGU>
magn_raw = data.magn_raw; %#ok<NASGU>
mp_rage = data.mp_rage; %#ok<NASGU>
chi_33 = data.chi_33; %#ok<NASGU>
chi_cosmos = data.chi_cosmos; %#ok<NASGU>
spatial_res = data.spatial_res; %#ok<NASGU>
evaluation_mask = data.evaluation_mask; %#ok<NASGU>
echo_times_ms = data.echo_times_ms; %#ok<NASGU>
echo_times_sec = data.echo_times_sec; %#ok<NASGU>
delta_TE = data.delta_TE; %#ok<NASGU>
B0 = data.B0; %#ok<NASGU>
B0_dir = data.B0_dir; %#ok<NASGU>
phase_fit_method = data.phase_fit_method; %#ok<NASGU>
phase_conversion = data.phase_conversion; %#ok<NASGU>

save(fullfile(output_data_dir, 'phs_tissue.mat'), 'phs_tissue');
save(fullfile(output_data_dir, 'phs_unwrap.mat'), 'phs_unwrap');
save(fullfile(output_data_dir, 'phs_wrap.mat'), 'phs_wrap');
try
    save(fullfile(output_data_dir, 'fieldmap_Hz.mat'), 'fieldmap_Hz', '-v7.3');
    save(fullfile(output_data_dir, 'local_field_ppm.mat'), 'local_field_ppm', '-v7.3');
    save(fullfile(output_data_dir, 'R2star_Hz.mat'), 'R2star_Hz', 'R2star_s0', 'R2star_fit_residual', '-v7.3');
catch
    save(fullfile(output_data_dir, 'fieldmap_Hz.mat'), 'fieldmap_Hz');
    save(fullfile(output_data_dir, 'local_field_ppm.mat'), 'local_field_ppm');
    save(fullfile(output_data_dir, 'R2star_Hz.mat'), 'R2star_Hz', 'R2star_s0', 'R2star_fit_residual');
end
save(fullfile(output_data_dir, 'msk.mat'), 'msk');
save(fullfile(output_data_dir, 'Mask.mat'), 'Mask');
save(fullfile(output_data_dir, 'magn.mat'), 'magn');
save(fullfile(output_data_dir, 'magn_raw.mat'), 'magn_raw');
save(fullfile(output_data_dir, 'mp_rage.mat'), 'mp_rage');
save(fullfile(output_data_dir, 'chi_33.mat'), 'chi_33');
save(fullfile(output_data_dir, 'chi_cosmos.mat'), 'chi_cosmos');
save(fullfile(output_data_dir, 'spatial_res.mat'), 'spatial_res');
save(fullfile(output_data_dir, 'evaluation_mask.mat'), 'evaluation_mask');
save(fullfile(output_data_dir, 'dicom_whqsm_metadata.mat'), ...
    'echo_times_ms', 'echo_times_sec', 'delta_TE', 'B0', 'B0_dir', ...
    'phase_fit_method', 'phase_conversion');
try
    save(fullfile(output_data_dir, 'data_full.mat'), 'data', '-v7.3');
catch
    save(fullfile(output_data_dir, 'data_full.mat'), 'data');
end
fprintf('  Saved WH-QSM input variables to: %s\n', output_data_dir);
end
