function SWI202606_dicom_scanner()
% SWI202606_dicom_scanner.m  (v4 — 稳定版)
% ============================================================================
%   关键修复：
%     - 文件发现：用 dir('**/*') (v2 验证可用) + 防御性去重
%     - 任意扩展名都做 magic number 检查（防止非 .dcm 漏掉）
%     - 保留 v3 的 cellfun/cellarray 防御性
% ============================================================================

ROOT_DIR = 'D:\MRI_PRO\MRILAB_X\20170327_qsm2016_recon_challenge\data_course\SWI202606';
OUTPUT_DIR = fullfile(ROOT_DIR, '_scan_report');
CSV_REPORT  = fullfile(OUTPUT_DIR, 'dicom_metadata_report.csv');
MAT_REPORT = fullfile(OUTPUT_DIR, 'dicom_metadata_struct.mat');

if ~exist(OUTPUT_DIR, 'dir'), mkdir(OUTPUT_DIR); end

fprintf('============================================================\n');
fprintf(' Siemens SWI 数据扫描器 (v4)\n');
fprintf(' 根目录: %s\n', ROOT_DIR);
fprintf('============================================================\n\n');

if ~exist(ROOT_DIR, 'dir'), error('路径不存在: %s', ROOT_DIR); end

%% [Step 1] 文件发现
fprintf('[Step 1/5] 文件发现...\n');
file_list = discover_dicom_files(ROOT_DIR);
fprintf('  -> 发现 %d 个 DICOM 文件\n\n', length(file_list));
if isempty(file_list)
    fprintf('  ❌ 未找到任何 DICOM 文件！请检查:\n');
    fprintf('     1) 路径是否正确\n');
    fprintf('     2) 文件扩展名是否为 .dcm/.dicom/.IMA/.001\n');
    fprintf('     3) 用 dcm2niix 先转换可能更稳\n');
    error('扫描失败');
end

%% [Step 2] dicominfo 解析
fprintf('[Step 2/5] dicominfo 解析...\n');
info_list = parse_with_dicominfo(file_list);
n_ok = sum(cellfun(@(x) ~isempty(x), info_list));
fprintf('  -> 成功解析 %d / %d 个文件\n\n', n_ok, length(file_list));

%% [Step 3] 分组
fprintf('[Step 3/5] 按 SeriesInstanceUID 分组...\n');
series_map = group_by_series_robust(info_list);
fprintf('  -> 共 %d 个不同序列\n\n', series_map.Count);

%% [Step 4] 分类
fprintf('[Step 4/5] 分类 + QSM 可行性评估...\n');
classified = classify_with_matlab(series_map, info_list);

%% [Step 5] 报告
fprintf('[Step 5/5] 生成报告...\n');
write_csv_report(classified, CSV_REPORT);
save_struct_report(classified, info_list, MAT_REPORT);
print_summary(classified);

fprintf('\n============================================================\n');
fprintf(' 完成！\n  CSV: %s\n  MAT: %s\n============================================================\n', ...
    CSV_REPORT, MAT_REPORT);
end

%% =========================================================================
% [Step 1] 文件发现 — 多策略 + 去重
% =========================================================================
function file_list = discover_dicom_files(root_dir)
file_list = {};
seen_paths = {};

% 策略 1: 用 '**' 递归（v2 验证可用，能找到 6096 个）
try
    all_files = dir(fullfile(root_dir, '**', '*'));
    all_files = all_files(~[all_files.isdir]);
catch
    all_files = struct('name',{},'folder',{},'isdir',{},'bytes',{},'date',{},'datenum',{});
end

% 策略 2: 如果 '**' 没找到，尝试显式逐子目录扫描
if isempty(all_files)
    fprintf('  [DEBUG] ** 递归失败，尝试显式扫描...\n');
    sub_dirs = {root_dir};
    dd = dir(root_dir);
    for k = 1:length(dd)
        if dd(k).isdir && ~any(strcmp(dd(k).name, {'.', '..'}))
            sub_dirs{end+1} = fullfile(root_dir, dd(k).name);
        end
    end
    for d = 1:length(sub_dirs)
        sub_files = dir(fullfile(sub_dirs{d}, '*'));
        for k = 1:length(sub_files)
            if ~sub_files(k).isdir
                all_files(end+1) = sub_files(k);
            end
        end
    end
end

fprintf('  [DEBUG] dir() 返回 %d 个候选文件\n', length(all_files));

% 过滤 + 去重 + 防御性 magic 检查
for k = 1:length(all_files)
    fp = fullfile(all_files(k).folder, all_files(k).name);

    % 去重
    if any(strcmp(seen_paths, fp)), continue; end

    [~, ~, ext] = fileparts(all_files(k).name);
    is_dcm = false;

    % 已知 DICOM 扩展名
    if any(strcmpi(ext, {'.dcm', '.dicom', '.ima', '.001', '.img'}))
        is_dcm = true;
    end

    % Magic number 检查（任何扩展名都查，更稳）
    if ~is_dcm
        try
            fid = fopen(fp, 'r');
            if fid ~= -1
                fseek(fid, 128, 'bof');
                magic = fread(fid, 4, 'uint8=>char')';
                fclose(fid);
                if length(magic) == 4 && strcmp(magic, 'DICM')
                    is_dcm = true;
                end
            end
        catch
        end
    end

    if is_dcm
        file_list{end+1} = fp;
        seen_paths{end+1} = fp;
    end
end
end

%% =========================================================================
% [Step 2] dicominfo 解析
%% =========================================================================
function info_list = parse_with_dicominfo(file_list)
n = length(file_list);
info_list = cell(1, n);
for k = 1:n
    try
        info_list{k} = dicominfo(file_list{k});
    catch ME
        info_list{k} = [];
        if k <= 5 || mod(k, 100) == 0
            fprintf('  ⚠️ 文件 %d 失败: %s\n', k, ME.message);
        end
    end
end
end

%% =========================================================================
% [Step 3] 分组 — UID 归一化 + Map
%% =========================================================================
function series_map = group_by_series_robust(info_list)
series_map = containers.Map('KeyType', 'char', 'ValueType', 'any');

for k = 1:length(info_list)
    info = info_list{k};
    if isempty(info), continue; end

    uid = normalize_string(safe_field(info, 'SeriesInstanceUID', ''));
    if isempty(uid)
        uid = sprintf('__NO_UID_%d', k);
    end

    if isKey(series_map, uid)
        s = series_map(uid);
        s.file_paths{end+1} = safe_field(info, 'Filename', '');
        s.info_indices(end+1) = k;
        series_map(uid) = s;
    else
        series_map(uid) = struct( ...
            'file_paths', {{safe_field(info, 'Filename', '')}}, ...
            'info_indices', k);
    end
end
end

function s = normalize_string(v)
s = '';
if isempty(v), return; end
if ischar(v), s = strtrim(v);
elseif isstring(v), s = strtrim(char(v));
elseif iscell(v) && length(v) >= 1 && ischar(v{1}), s = strtrim(v{1});
end
end

%% =========================================================================
% 通用：安全字段访问
%% =========================================================================
function val = safe_field(info, fname, default)
if isempty(info) || ~isfield(info, fname) || isempty(info.(fname))
    val = default; return;
end
raw = info.(fname);
if ischar(raw), val = strtrim(raw);
elseif isstring(raw), val = char(raw);
elseif iscell(raw) && ~isempty(raw) && ischar(raw{1}), val = strtrim(raw{1});
else, val = raw;
end
end

function val = safe_field_num(info, fname, default)
val = safe_field(info, fname, default);
if ~isnumeric(val) || ~isscalar(val) || isnan(val), val = default; end
end

%% =========================================================================
% [Step 4] 分类
%% =========================================================================
function classified = classify_with_matlab(series_map, info_list)
classified = {};
keys_arr = series_map.keys;

for k = 1:length(keys_arr)
    uid = keys_arr{k};
    s   = series_map(uid);
    if isempty(s.info_indices), continue; end

    info = info_list{s.info_indices(1)};

    sd = lower(char(safe_field(info, 'SeriesDescription', '')));
    pn = lower(char(safe_field(info, 'ProtocolName', '')));
    it = lower(char(safe_field(info, 'ImageType', '')));

    is_multi_echo = is_multi_echo_series(info_list, s.info_indices);

    [cat, conf, notes] = classify_one_series(sd, pn, it, info, info_list, s.info_indices, is_multi_echo);
    [qsm_ok, qsm_notes] = assess_qsm(cat, info);

    classified{end+1} = struct( ...
        'uid', uid, ...
        'info_indices', s.info_indices, ...
        'file_paths', {s.file_paths}, ...
        'info', info, ...
        'category', cat, ...
        'confidence', conf, ...
        'notes', {notes}, ...
        'qsm_usable', qsm_ok, ...
        'qsm_notes', {qsm_notes});
end

% 按 SeriesNumber 排序
ser_nums = cellfun(@(c) safe_field_num(c.info, 'SeriesNumber', 9999), classified);
[~, ord] = sort(ser_nums);
classified = classified(ord);
end

function [cat, conf, notes] = classify_one_series(sd, pn, it, info, info_list, indices, is_multi_echo)
notes = {};

if contains(sd, 'mprage') || contains(sd, 'mp-rage') || ...
   contains(sd, 't1') || contains(pn, 't1')
    cat = 'T1_STRUCTURAL'; conf = 0.95;
    notes{end+1} = 'T1 结构像（MEDI 先验）'; return;
end

if contains(sd, 'mip') || contains(sd, 'min ip')
    cat = 'MIP'; conf = 0.9;
    notes{end+1} = 'mIP 最小强度投影';
    if contains(sd, 'sw'), notes{end+1} = 'SWI mIP'; end
    return;
end

if contains(sd, 'swi') || contains(pn, 'swi') || contains(sd, 'swan')
    cat = 'SWI'; conf = 0.9;
    notes{end+1} = 'SWI 后处理图'; return;
end

if contains(it, 'phase') || contains(it, '\p\')
    cat = 'PHASE'; conf = 0.95;
    notes{end+1} = '🌟 Phase 原始数据（QSM 关键）';
    slope = safe_field_num(info, 'RescaleSlope', 1);
    intercept = safe_field_num(info, 'RescaleIntercept', 0);
    if abs(slope - 1) > 1e-6 || abs(intercept) > 1e-6
        notes{end+1} = sprintf('⚠️ Siemens 缩放: pixel=raw*%.6f+%.6f', slope, intercept);
    end
    return;
end

if contains(it, 'magnitude') || contains(it, '\m\') || contains(sd, 'mag')
    cat = 'MAGNITUDE'; conf = 0.95;
    if is_multi_echo
        te_str = mat2str(round(get_echo_times(info_list, indices), 2));
        notes{end+1} = sprintf('多回波 TE: %s ms', te_str);
    else
        te = safe_field_num(info, 'EchoTime', NaN);
        notes{end+1} = sprintf('单回波 TE=%.2f ms', te);
    end
    return;
end

cat = 'UNKNOWN'; conf = 0.3;
notes{end+1} = sprintf('SeriesDescription="%s", ImageType="%s"', ...
    char(safe_field(info, 'SeriesDescription', '?')), ...
    char(safe_field(info, 'ImageType', '?')));
end

function tf = is_multi_echo_series(info_list, indices)
te_list = [];
for k = 1:length(indices)
    te = safe_field_num(info_list{indices(k)}, 'EchoTime', NaN);
    if ~isnan(te), te_list(end+1) = te; end
end
tf = length(unique(te_list)) > 1;
end

function te_list = get_echo_times(info_list, indices)
te_list = [];
for k = 1:length(indices)
    te = safe_field_num(info_list{indices(k)}, 'EchoTime', NaN);
    if ~isnan(te), te_list(end+1) = te; end
end
te_list = unique(te_list);
end

function [ok, notes] = assess_qsm(cat, info)
notes = {}; ok = false;
switch cat
    case 'PHASE'
        ok = true; notes{end+1} = '✅ QSM phase 输入';
    case 'MAGNITUDE'
        ok = true; notes{end+1} = '✅ QSM magnitude 输入';
    case 'T1_STRUCTURAL'
        ok = true; notes{end+1} = '✅ MEDI 结构先验';
    case {'SWI', 'MIP'}
        notes{end+1} = '⚠️ 仅作视觉对比';
    otherwise
        notes{end+1} = '❌ 类别未知';
end
end

%% =========================================================================
% [Step 5a] CSV 报告
%% =========================================================================
function write_csv_report(classified, csv_path)
fid = fopen(csv_path, 'w', 'n', 'UTF-8');

header = { ...
    'SeriesNumber', 'Category', 'Confidence', 'Description', ...
    'NFiles', 'Rows', 'Columns', 'PixelSpacingX', 'PixelSpacingY', 'SliceThickness', ...
    'Manufacturer', 'FieldStrength_T', 'TR_ms', 'FA_deg', 'EchoTrainLength', ...
    'EchoTime_ms', 'EchoNumber', 'ImageType', ...
    'RescaleSlope', 'RescaleIntercept', 'BitsStored', ...
    'PatientID', 'StudyDate', 'QSM_Usable', 'Notes'};
fprintf(fid, '%s\n', strjoin(header, ','));

for k = 1:length(classified)
    c = classified{k};
    info = c.info;

    ser_num    = safe_field_num(info, 'SeriesNumber', NaN);
    sd         = char(safe_field(info, 'SeriesDescription', ''));
    n_files    = length(c.file_paths);
    rows       = safe_field_num(info, 'Rows', NaN);
    cols       = safe_field_num(info, 'Columns', NaN);
    pix        = safe_field(info, 'PixelSpacing', [NaN NaN]);
    ps_x = NaN; ps_y = NaN;
    if length(pix) >= 2, ps_x = pix(1); ps_y = pix(2); end
    slice_thk  = safe_field_num(info, 'SliceThickness', NaN);
    manu       = char(safe_field(info, 'Manufacturer', ''));
    field_t    = safe_field_num(info, 'FieldStrength', NaN);
    tr         = safe_field_num(info, 'RepetitionTime', NaN);
    fa         = safe_field_num(info, 'FlipAngle', NaN);
    etl        = safe_field_num(info, 'EchoTrainLength', NaN);
    te         = safe_field_num(info, 'EchoTime', NaN);
    en         = safe_field_num(info, 'EchoNumber', NaN);
    image_type = char(safe_field(info, 'ImageType', ''));
    slope      = safe_field_num(info, 'RescaleSlope', 1);
    intercept  = safe_field_num(info, 'RescaleIntercept', 0);
    bits       = safe_field_num(info, 'BitsStored', NaN);
    pat_id     = char(safe_field(info, 'PatientID', ''));
    study_dt   = char(safe_field(info, 'StudyDate', ''));
    qsm_ok     = char(iff(c.qsm_usable, 'YES', 'NO'));
    notes_str  = escape_csv(strjoin([c.notes, c.qsm_notes], ' | '));

    fprintf(fid, '%d,%s,%.2f,"%s",%d,%d,%d,%.4f,%.4f,%.4f,"%s",%.2f,%.4f,%.2f,%d,%.4f,%d,"%s",%.6f,%.6f,%d,"%s","%s",%s,"%s"\n', ...
        ser_num, c.category, c.confidence, escape_csv(sd), ...
        n_files, rows, cols, ps_x, ps_y, slice_thk, ...
        escape_csv(manu), field_t, tr, fa, etl, ...
        te, en, escape_csv(image_type), ...
        slope, intercept, bits, ...
        escape_csv(pat_id), escape_csv(study_dt), ...
        qsm_ok, notes_str);
end
fclose(fid);
fprintf('  -> CSV: %s\n', csv_path);
end

function s = iff(c, a, b), if c, s = a; else, s = b; end, end
function s = escape_csv(s), s = strrep(s, '"', '""'); end

%% =========================================================================
% [Step 5b] MAT
%% =========================================================================
function save_struct_report(classified, info_list, mat_path)
save(mat_path, 'classified', 'info_list', '-v7.3');
fprintf('  -> MAT: %s\n', mat_path);
end

%% =========================================================================
% [Step 5c] 控制台摘要
%% =========================================================================
function print_summary(classified)
fprintf('\n============================================================\n');
fprintf(' 序列分类摘要\n');
fprintf('============================================================\n');
fprintf('%-4s %-18s %-25s %-6s %-8s %s\n', ...
    'Ser#', 'Category', 'Description', 'Files', 'QSM', 'Notes');
fprintf('%s\n', repmat('-', 1, 100));

for k = 1:length(classified)
    c = classified{k};
    info = c.info;
    ser = safe_field_num(info, 'SeriesNumber', NaN);
    sd  = char(safe_field(info, 'SeriesDescription', ''));
    if length(sd) > 25, sd = [sd(1:22) '...']; end

    notes_str = strjoin([c.notes, c.qsm_notes], '; ');
    if length(notes_str) > 60, notes_str = [notes_str(1:57) '...']; end

    fprintf('%-4d %-18s %-25s %-6d %-8s %s\n', ...
        ser, c.category, sd, length(c.file_paths), ...
        iff(c.qsm_usable, '✅ YES', '⚠️ NO'), notes_str);
end

cats = cellfun(@(c) c.category, classified, 'UniformOutput', false);
fprintf('\n【分类统计】\n');
all_cats = {'PHASE', 'MAGNITUDE', 'T1_STRUCTURAL', 'SWI', 'MIP', 'UNKNOWN'};
for k = 1:length(all_cats)
    cnt = sum(strcmp(cats, all_cats{k}));
    if cnt > 0
        fprintf('  %-15s : %d\n', all_cats{k}, cnt);
    end
end

n_phase = sum(strcmp(cats, 'PHASE'));
n_mag   = sum(strcmp(cats, 'MAGNITUDE'));
n_t1    = sum(strcmp(cats, 'T1_STRUCTURAL'));
fprintf('\n【QSM 可行性】\n');
if n_phase >= 1 && n_mag >= 1
    fprintf(' ✅ 满足 QSM 重建条件（有 PHASE + MAGNITUDE）\n');
    if n_t1 >= 1
        fprintf(' ✅ 有 T1 结构像 → MEDI 结构先验可用\n');
    else
        fprintf(' ⚠️ 无 T1 结构像 → MEDI 无结构先验\n');
    end
else
    fprintf(' ❌ 不满足 QSM 重建条件\n');
end

fprintf('\n【Siemens Phase 缩放】\n');
for k = 1:length(classified)
    c = classified{k};
    if strcmp(c.category, 'PHASE')
        slope = safe_field_num(c.info, 'RescaleSlope', 1);
        intercept = safe_field_num(c.info, 'RescaleIntercept', 0);
        ser = safe_field_num(c.info, 'SeriesNumber', NaN);
        if abs(slope - 1) > 1e-6 || abs(intercept) > 1e-6
            fprintf(' ⚠️ Ser#%d: pixel=raw*%.6f+%.6f\n', ser, slope, intercept);
        else
            fprintf(' ✅ Ser#%d: 无需缩放\n', ser);
        end
    end
end
end
