function data = dicom_loader_subject(subject, output_data_dir)
% dicom_loader_subject.m  (v3 - 防御性 + 调试版)
% ============================================================================
%   加载单个被试的 Siemens SWI DICOM 数据
%   每个步骤都有详细的进度输出和错误捕获
% ============================================================================

if nargin < 2 || isempty(output_data_dir)
    output_data_dir = fullfile(subject.path, '_qsm2016_format');
end

fprintf('\n');
fprintf('============================================================\n');
fprintf(' DICOM 加载器 v3 - 被试: %s (%s)\n', subject.name, upper(subject.group));
fprintf(' 路径: %s\n', subject.path);
fprintf('============================================================\n\n');

if ~exist(output_data_dir, 'dir'), mkdir(output_data_dir); end

% ====== Step 1: 扫描所有 DICOM ======
fprintf('[Step 1/6] 扫描 DICOM...\n');
try
    file_list = discover_all_dicom(subject.path);
    fprintf('  ✅ 发现 %d 个 DICOM\n', length(file_list));
catch ME
    fprintf('  ❌ 扫描失败: %s\n', ME.message);
    rethrow(ME);
end

% ====== Step 2: 按 UID 分组 ======
fprintf('\n[Step 2/6] 按 SeriesInstanceUID 分组...\n');
try
    series_map = classify_series_simple(file_list);
    fprintf('  ✅ 发现 %d 个不同序列\n', series_map.Count);
catch ME
    fprintf('  ❌ 分组失败: %s\n', ME.message);
    rethrow(ME);
end

% ====== Step 3: 识别 T1/Mag/Phase ======
fprintf('\n[Step 3/6] 识别 T1/Mag/Phase 序列...\n');
try
    [t1_info, mag_info, pha_info] = find_key_series(series_map);

    if isempty(pha_info)
        error('未找到 PHASE 序列!');
    end
    if isempty(mag_info)
        error('未找到 MAGNITUDE 序列!');
    end

    B0 = detect_B0(pha_info, mag_info, t1_info);
    fprintf('  ✅ T1=%s  Mag=%s  Phase=%s\n', ...
        iff(~isempty(t1_info), '✓', '✗'), ...
        iff(~isempty(mag_info), '✓', '✗'), ...
        iff(~isempty(pha_info), '✓', '✗'));
    fprintf('  ✅ 场强 B0 = %.2f T\n', B0);
catch ME
    fprintf('  ❌ 识别失败: %s\n', ME.message);
    rethrow(ME);
end

% ====== Step 4: 加载 Magnitude ======
fprintf('\n[Step 4/6] 加载 Magnitude...\n');
try
    [magn_vol, mag_info_used] = load_magnitude_volume(mag_info, subject.path);
    fprintf('  ✅ Magnitude 体积: %s\n', mat2str(size(magn_vol)));
catch ME
    fprintf('  ❌ Magnitude 加载失败: %s\n', ME.message);
    fprintf('     at %s:%d\n', ME.stack(1).name, ME.stack(1).line);
    rethrow(ME);
end

% ====== Step 5: 加载 Phase ======
fprintf('\n[Step 5/6] 加载 Phase...\n');
try
    [pha_vol_rad, pha_info_used] = load_phase_volume_rad(pha_info, subject.path);
    fprintf('  ✅ Phase 体积: %s\n', mat2str(size(pha_vol_rad)));
    fprintf('  ✅ 相位值域: [%.4f, %.4f] rad\n', min(pha_vol_rad(:)), max(pha_vol_rad(:)));
catch ME
    fprintf('  ❌ Phase 加载失败: %s\n', ME.message);
    fprintf('     at %s:%d\n', ME.stack(1).name, ME.stack(1).line);
    rethrow(ME);
end

% ====== rad → ppm 单位转换 ======
fprintf('\n[Step 5.5] rad → ppm 单位转换...\n');
try
    TE_sec = scalarize(pha_info_used.EchoTime) / 1000;
    gyro_MHz_per_T = 42.57747892;

    % 🔴 关键修正: Siemens 12-bit Phase DICOM 的缩放约定
    % 用户数据: RescaleSlope=2.0, RescaleIntercept=-4096
    % 应用后: phase_dicom ∈ [-4096, +4094] (Siemens 内部单位, 非弧度!)
    % 必须先 × (π/4096) 转成弧度, 再转 ppm
    %
    % 合并公式 (γ 用 MHz/T):
    %   ppm = phase_dicom × 1 / (4096 × 2 × γ_MHz × B0 × TE_sec)

    ppm_factor = 1 / (4096 * 2 * gyro_MHz_per_T * B0 * TE_sec);
    fprintf('  B0=%.2fT, TE=%.4fms, γ=%.4f MHz/T\n', B0, TE_sec*1000, gyro_MHz_per_T);
    fprintf('  ppm_factor = %.6e\n', ppm_factor);
    fprintf('  公式: ppm = phase_dicom × 1 / (4096 × 2 × γ × B0 × TE)\n');

    pha_vol_ppm = pha_vol_rad * ppm_factor;
    fprintf('  ✅ ppm 值域: [%.4f, %.4f] ppm\n', ...
        min(pha_vol_ppm(:)), max(pha_vol_ppm(:)));

    if abs(max(pha_vol_ppm(:))) > 5 || abs(min(pha_vol_ppm(:))) < -5
        warning('⚠️ ppm 值域异常 [%.3f, %.3f]，请检查 Phase Siemens 缩放', ...
            min(pha_vol_ppm(:)), max(pha_vol_ppm(:)));
    end
catch ME
    fprintf('  ❌ 转换失败: %s\n', ME.message);
    rethrow(ME);
end

% ====== Step 6: 加载 T1 ======
fprintf('\n[Step 6/6] 加载 T1...\n');
try
    if ~isempty(t1_info)
        t1_vol = load_t1_volume(t1_info, subject.path);
        fprintf('  ✅ T1 体积: %s\n', mat2str(size(t1_vol)));
    else
        fprintf('  ⚠️ 无 T1，使用占位\n');
        t1_vol = zeros(size(magn_vol), 'double');
    end
catch ME
    fprintf('  ⚠️ T1 加载失败: %s\n', ME.message);
    fprintf('     使用占位零矩阵\n');
    t1_vol = zeros(size(magn_vol), 'double');
end

% ====== 准备 11 个变量 ======
fprintf('\n[Final] 准备 QSM2016 变量...\n');
try
    spatial_res = [scalarize(pha_info_used.PixelSpacing(2)), ...
                   scalarize(pha_info_used.PixelSpacing(1)), ...
                   scalarize(pha_info_used.SliceThickness)];

    if ~isequal(size(t1_vol), size(magn_vol))
        fprintf('  T1 %s → resize 到 %s\n', mat2str(size(t1_vol)), mat2str(size(magn_vol)));
        t1_vol = resize_volume_nn(t1_vol, size(magn_vol));
    end

    mask = generate_brain_mask(magn_vol);
    fprintf('  ✅ Brain mask: %d voxels (%.2f%%)\n', ...
        nnz(mask), 100*nnz(mask)/numel(mask));

    % 组装 data
    data = struct();
    data.phs_tissue = double(pha_vol_ppm);
    data.phs_unwrap = double(pha_vol_rad);
    data.phs_wrap   = wrap_to_pi(pha_vol_rad);
    data.msk        = logical(mask);
    data.magn       = double(magn_vol);
    data.magn_raw   = double(magn_vol);
    data.mp_rage    = double(t1_vol);
    data.chi_33     = zeros(size(mask), 'double');
    data.chi_cosmos = zeros(size(mask), 'double');
    data.spatial_res = double(spatial_res);
    data.N          = size(mask);
    data.Mask       = logical(mask);
    data.evaluation_mask = double(mask);
    data.EchoTime       = scalarize(pha_info_used.EchoTime);
    data.TE             = scalarize(pha_info_used.EchoTime) / 1000;
    data.B0             = B0;
    data.b0             = B0;
    data.FieldStrength  = B0;
    data.Manufacturer   = char(safe_field_str(pha_info_used, 'Manufacturer', ''));
    data.patient_group  = subject.group;
    data.subject_name   = subject.name;
    data.ppm_factor     = ppm_factor;

    % mask 应用
    data.phs_tissue(~mask) = 0;
    data.phs_unwrap(~mask) = 0;

    fprintf('  ✅ data 结构体已组装\n');
catch ME
    fprintf('  ❌ 组装失败: %s\n', ME.message);
    fprintf('     at %s:%d\n', ME.stack(1).name, ME.stack(1).line);
    rethrow(ME);
end

% ====== 保存 .mat 文件 ======
fprintf('\n[Save] 保存 11 个 .mat 变量...\n');
try
    save(fullfile(output_data_dir, 'phs_tissue.mat'),  'data');
    fprintf('  ✅ phs_tissue.mat\n');
catch ME
    fprintf('  ❌ 保存失败: %s\n', ME.message);
    rethrow(ME);
end

fprintf('\n✅ 加载完成！phs_tissue: %s (ppm)\n', mat2str(size(data.phs_tissue)));
end

%% =========================================================================
%% 安全辅助函数
%% =========================================================================
function v = safe_field_str(s, fname, default)
v = default;
if ~isfield(s, fname) || isempty(s.(fname)), return; end
val = s.(fname);
if ischar(val), v = strtrim(val);
elseif isstring(val), v = char(val);
elseif iscell(val) && ~isempty(val) && ischar(val{1}), v = strtrim(val{1});
elseif isnumeric(val) && isscalar(val), v = num2str(val);
else, v = default; end
end

function v = scalarize(x)
if isnumeric(x) && isscalar(x), v = double(x);
elseif isnumeric(x) && ~isempty(x), v = double(x(1));
elseif iscell(x) && ~isempty(x)
    if isnumeric(x{1}), v = double(x{1});
    elseif ischar(x{1}) && ~isempty(x{1}), v = str2double(x{1});
    else, v = 0; end
elseif ischar(x) && ~isempty(x), v = str2double(x);
elseif isstring(x) && ~isempty(x), v = double(x(1));
else, v = 0;
end
end

function r = iff(c, a, b), if c, r = a; else, r = b; end, end

function B0 = detect_B0(varargin)
B0 = NaN;
for k = 1:nargin
    info = varargin{k};
    if isempty(info), continue; end
    if isfield(info, 'FieldStrength') && ~isempty(info.FieldStrength)
        v = scalarize(info.FieldStrength);
        if isnan(v) || v <= 0, continue; end
        B0 = v; return;
    end
end
if isnan(B0), B0 = 3; end
end

function [slope, intercept] = safe_rescale(info)
slope = 1; intercept = 0;
if isfield(info, 'RescaleSlope') && ~isempty(info.RescaleSlope)
    slope = scalarize(info.RescaleSlope);
    if slope == 0, slope = 1; end
end
if isfield(info, 'RescaleIntercept') && ~isempty(info.RescaleIntercept)
    intercept = scalarize(info.RescaleIntercept);
end
end

%% =========================================================================
%% 内部: DICOM 发现
%% =========================================================================
function file_list = discover_all_dicom(root_dir)
file_list = {};
all_files = dir(fullfile(root_dir, '**', '*'));
all_files = all_files(~[all_files.isdir]);
seen = {};
for k = 1:length(all_files)
    [~, ~, ext] = fileparts(all_files(k).name);
    is_dcm = any(strcmpi(ext, {'.dcm', '.dicom', '.ima', '.001'}));
    if ~is_dcm && isempty(ext)
        is_dcm = check_magic(fullfile(all_files(k).folder, all_files(k).name));
    end
    if is_dcm
        fp = fullfile(all_files(k).folder, all_files(k).name);
        if ischar(fp) && ~any(strcmp(seen, fp))
            file_list{end+1} = fp;
            seen{end+1} = fp;
        end
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
catch, end
fclose(fid);
end

%% =========================================================================
%% 内部: 序列分类
%% =========================================================================
function series_map = classify_series_simple(file_list)
series_map = containers.Map('KeyType', 'char', 'ValueType', 'any');

for k = 1:length(file_list)
    try
        info = dicominfo(file_list{k});
        uid = '';
        if isfield(info, 'SeriesInstanceUID') && ~isempty(info.SeriesInstanceUID)
            raw = info.SeriesInstanceUID;
            if ischar(raw), uid = strtrim(raw);
            elseif iscell(raw) && ~isempty(raw), uid = strtrim(raw{1});
            elseif isstring(raw), uid = char(raw);
            end
        end
        if isempty(uid), uid = sprintf('__NO_UID_%d', k); end

        if isKey(series_map, uid)
            s = series_map(uid);
            s.file_paths{end+1} = file_list{k};
            series_map(uid) = s;
        else
            series_map(uid) = struct( ...
                'file_paths', {{file_list{k}}}, ...
                'info', info);
        end
    catch ME
        fprintf('  ⚠️ 跳过 %s: %s\n', file_list{k}, ME.message);
    end
end
end

%% =========================================================================
%% 内部: 找出 T1 / Mag / Phase
%% =========================================================================
function [t1_info, mag_info, pha_info] = find_key_series(series_map)
t1_info = []; mag_info = []; pha_info = [];

keys = series_map.keys;
for k = 1:length(keys)
    s = series_map(keys{k});
    info = s.info;

    sd = '';
    it = '';
    if isfield(info, 'SeriesDescription'), sd = lower(char(info.SeriesDescription)); end
    if isfield(info, 'ImageType'),         it = lower(char(info.ImageType)); end

    if contains(sd, 'mprage') || contains(sd, 't1')
        t1_info = info;
    elseif contains(it, '\p\')
        if isempty(pha_info) || ~contains(it, 'norm')
            pha_info = info;
        end
    elseif contains(it, '\m\')
        if contains(it, 'swi') || contains(it, 'mnip')
            continue;
        end
        if isempty(mag_info) || ~contains(it, 'norm')
            mag_info = info;
        end
    end
end
end

%% =========================================================================
%% 内部: 加载 Magnitude (多回波 → 沿 echo 维平均)
%% =========================================================================
function [vol, info_used] = load_magnitude_volume(info, root_dir)
uid = extract_uid(info);
files = find_files_by_uid(root_dir, uid);

[slope, intercept] = safe_rescale(info);
fprintf('  RescaleSlope=%.4f, RescaleIntercept=%.4f\n', slope, intercept);

if isempty(files)
    error('未找到 Magnitude DICOM');
end

% 检测多回波: 按 (EchoNumber, InstanceNumber) 排序后分组
vol4d = build_4d_volume(files, slope, intercept);

fprintf('  体素: [%d %d %d] (slice=%d, echo=%d)\n', ...
    size(vol4d,1), size(vol4d,2), size(vol4d,3), ...
    size(vol4d,3), size(vol4d,4));

% Magnitude: 沿 echo 维平均 (提升 SNR)
vol = mean(vol4d, 4);

info_used = info;
end

%% =========================================================================
%% 内部: 加载 Phase (多回波 → 取最后一个 echo)
%% =========================================================================
function [vol, info_used] = load_phase_volume_rad(info, root_dir)
uid = extract_uid(info);
files = find_files_by_uid(root_dir, uid);

[slope, intercept] = safe_rescale(info);
fprintf('  RescaleSlope=%.4f, RescaleIntercept=%.4f\n', slope, intercept);

pix_rep = 0;
if isfield(info, 'PixelRepresentation') && ~isempty(info.PixelRepresentation)
    pix_rep = scalarize(info.PixelRepresentation);
end
bits = 12;
if isfield(info, 'BitsStored') && ~isempty(info.BitsStored)
    bits = scalarize(info.BitsStored);
end

if isempty(files)
    error('未找到 Phase DICOM');
end

% 检测多回波: 按 (EchoNumber, InstanceNumber) 排序后分组
% 注意: Phase 在加载时就要处理 PixelRepresentation 和缩放
vol4d = build_4d_volume(files, slope, intercept, pix_rep, bits);

fprintf('  体素: [%d %d %d] (slice=%d, echo=%d)\n', ...
    size(vol4d,1), size(vol4d,2), size(vol4d,3), ...
    size(vol4d,3), size(vol4d,4));

% Phase: Siemens 通常只在最后一个 TE 输出 phase → 取最后 echo
% 但如果有多个 TE 的 phase，做 echo 拟合更好
% 这里先简单取最后 echo
vol = vol4d(:,:,:,end);

info_used = info;
end

%% =========================================================================
%% 内部: 加载 T1
%% =========================================================================
function vol = load_t1_volume(info, root_dir)
uid = extract_uid(info);
files = find_files_by_uid(root_dir, uid);
files = sort_by_position(files);

[slope, intercept] = safe_rescale(info);

n = length(files);
if n == 0
    warning('未找到 T1 DICOM');
    vol = []; return;
end

sample = dicomread(files{1});
sz = size(sample);
vol = zeros(sz, 'double');

for k = 1:n
    X = double(dicomread(files{k}));
    X = X * slope + intercept;
    vol(:,:,k) = X;
end
end

%% =========================================================================
%% 内部: 工具函数
%% =========================================================================
function uid = extract_uid(info)
uid = '';
if ~isfield(info, 'SeriesInstanceUID'), return; end
raw = info.SeriesInstanceUID;
if isempty(raw), return; end
if ischar(raw), uid = strtrim(raw);
elseif iscell(raw) && ~isempty(raw), uid = strtrim(raw{1});
elseif isstring(raw), uid = char(raw);
end
end

function files = find_files_by_uid(root_dir, target_uid)
files = {};
all_dcm = dir(fullfile(root_dir, '**', '*.dcm'));
for k = 1:length(all_dcm)
    fp = fullfile(all_dcm(k).folder, all_dcm(k).name);
    if ~ischar(fp), continue; end
    try
        info = dicominfo(fp);
        cu = extract_uid(info);
        if ~isempty(cu) && strcmp(cu, target_uid)
            files{end+1} = fp;
        end
    catch
    end
end
end

function files = sort_by_echo(files)
% 简化的回波排序 (避免与 sort_files_by_echo 冲突)
n = length(files);
en = zeros(n, 1);
in = zeros(n, 1);
for k = 1:n
    try
        info = dicominfo(files{k});
        en(k) = safe_dicom_num(info, 'EchoNumber', k);
        in(k) = safe_dicom_num(info, 'InstanceNumber', k);
    catch
        en(k) = k;
        in(k) = k;
    end
end
% 用 index-based sort 避免 sortrows
[~, ord_e] = sort(en);
[~, ord_in] = sort(in);
% 用 lex 排序 (先 echo 后 instance)
combined = en * 1e6 + in;
[~, ord] = sort(combined);
files = files(ord);
end

%% =========================================================================
%% 内部: 安全提取 DICOM 数值字段
%% =========================================================================
function v = safe_dicom_num(info, fname, default)
v = default;
if ~isfield(info, fname), return; end
val = info.(fname);
if isempty(val), return; end
if isnumeric(val) && isscalar(val)
    v = double(val);
elseif isnumeric(val) && ~isempty(val)
    v = double(val(1));
elseif iscell(val) && ~isempty(val) && isnumeric(val{1})
    v = double(val{1});
elseif ischar(val) && ~isempty(val)
    v = str2double(val);
else
    v = default;
end
end

%% =========================================================================
%% 内部: 构建 4D 体数据 [x, y, slice, echo]
%% =========================================================================
function vol4d = build_4d_volume(files, slope, intercept, pix_rep, bits)
% 读取所有 DICOM 并组织成 4D 数组 [x, y, n_slices, n_echoes]

n = length(files);
if n == 0
    vol4d = []; return;
end

% 提取每个文件的 (echo, instance)
echo_nums = zeros(n, 1);
inst_nums = zeros(n, 1);
for k = 1:n
    try
        info = dicominfo(files{k});
        echo_nums(k) = safe_dicom_num(info, 'EchoNumber', 1);
        inst_nums(k) = safe_dicom_num(info, 'InstanceNumber', k);
    catch
        echo_nums(k) = 1;
        inst_nums(k) = k;
    end
end

% 按 (echo, instance) 排序 (lex sort)
combined = double(echo_nums) * 1e6 + double(inst_nums);
[~, ord] = sort(combined);
files = files(ord);
echo_nums = echo_nums(ord);
inst_nums = inst_nums(ord);

% 唯一 echo 和 slice
unique_echoes = unique(echo_nums);
n_echoes = length(unique_echoes);

% 每个 echo 应该有相同数量的 slice
unique_inst_per_echo = unique(inst_nums(echo_nums == unique_echoes(1)));
n_slices = length(unique_inst_per_echo);

% 检查
for e = 2:n_echoes
    inst_e = unique(inst_nums(echo_nums == unique_echoes(e)));
    if length(inst_e) ~= n_slices
        fprintf('  ⚠️ Echo %g 有 %d slice，Echo 1 有 %d\n', ...
            unique_echoes(e), length(inst_e), n_slices);
    end
end

% 读取第一个文件确定尺寸
sample = dicomread(files{1});
sz = size(sample);
if length(sz) < 2, sz = [sz 1]; end
sz_xy = sz(1:2);

% 创建 4D 数组
vol4d = zeros(sz_xy(1), sz_xy(2), n_slices, n_echoes, 'double');

% 填充
for k = 1:n
    X = double(dicomread(files{k}));
    if nargin >= 4 && ~isempty(pix_rep) && pix_rep == 1
        if nargin < 5 || isempty(bits), bits = 12; end
        X = X - 2^(bits-1);
    end
    X = X * slope + intercept;

    % 找这个文件对应哪个 (slice, echo)
    e_idx = find(unique_echoes == echo_nums(k), 1);
    s_idx = find(unique_inst_per_echo == inst_nums(k), 1);
    if isempty(e_idx) || isempty(s_idx)
        fprintf('  ⚠️ 跳过: file %d (echo=%g, inst=%g)\n', ...
            k, echo_nums(k), inst_nums(k));
        continue;
    end

    vol4d(:,:,s_idx,e_idx) = X;
end
end

function files = sort_by_position(files)
n = length(files);
in = zeros(n, 1);
for k = 1:n
    try
        info = dicominfo(files{k});
        in(k) = scalarize(getfield_or(info, 'InstanceNumber', k));
    catch
        in(k) = k;
    end
end
[~, ord] = sort(in);
files = files(ord);
end

function v = getfield_or(s, fname, default)
if isfield(s, fname) && ~isempty(s.(fname))
    v = s.(fname);
else
    v = default;
end
end

%% =========================================================================
%% 内部: 脑 mask
%% =========================================================================
function mask = generate_brain_mask(magn)
v = magn(:);
v_max = prctile(v, 99);
thr = 0.12 * v_max;
mask = magn > thr;
mask = imfill(mask, 'holes');
se = strel('sphere', 2);
mask = imerode(mask, se);
mask = imdilate(mask, se);

CC = bwconncomp(mask);
if CC.NumObjects > 1
    sizes = cellfun(@numel, CC.PixelIdxList);
    [~, idx_max] = max(sizes);
    mask = false(size(mask));
    mask(CC.PixelIdxList{idx_max}) = true;
end
end

%% =========================================================================
%% 内部: 重采样
%% =========================================================================
function out = resize_volume_nn(vol, target_size)
sz = size(vol);
out = zeros(target_size, 'double');
for d = 1:3
    if sz(d) == target_size(d)
        idx{d} = 1:sz(d);
    else
        idx{d} = round(linspace(1, sz(d), target_size(d)));
    end
end
out = vol(idx{1}, idx{2}, idx{3});
end

function p = wrap_to_pi(x)
p = mod(x + pi, 2*pi) - pi;
end
