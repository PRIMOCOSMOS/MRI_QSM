function subjects = discover_subjects(data_root)
% discover_subjects.m  (v4 - 稳健版, 用 dir(**) 单次扫描)
% ============================================================================
%   用 MATLAB 原生 dir(**) 一次扫描所有 .dcm 文件，避免递归 bug
%   智能识别被试根目录
% ============================================================================

if nargin < 1 || isempty(data_root)
    data_root = 'D:\MRI_PRO\MRILAB_X\20170327_qsm2016_recon_challenge\data_course';
end

fprintf('============================================================\n');
fprintf(' 被试目录发现 (v4 - 单次全扫描)\n');
fprintf(' 根目录: %s\n', data_root);
fprintf('============================================================\n\n');

if ~exist(data_root, 'dir')
    error('data_root 不存在: %s', data_root);
end

% ====== 黑名单（不当作被试目录）======
blacklist = {'_scan_report', '_qsm_comparison_results', ...
             'output', 'results', 'figures', 'Models', ...
             '.', '..'};

% ====== 单次全扫描所有 .dcm 文件 ======
fprintf('[1/4] 全局扫描所有 .dcm 文件...\n');
all_dcm = dir(fullfile(data_root, '**', '*.dcm'));

% 过滤掉黑名单目录中的文件
valid_dcm_paths = {};
for k = 1:length(all_dcm)
    fp = fullfile(all_dcm(k).folder, all_dcm(k).name);

    % 检查路径是否含黑名单目录
    skip = false;
    for b = 1:length(blacklist)
        if ~isempty(strfind(fp, fullfile(filesep, blacklist{b}, filesep))) || ...
           ~isempty(strfind(fp, ['\' blacklist{b} '\'])) || ...
           ~isempty(strfind(fp, ['/' blacklist{b} '/']))
            skip = true;
            break;
        end
    end
    if ~skip
        valid_dcm_paths{end+1} = fp;
    end
end

% 去重
if ~isempty(valid_dcm_paths)
    valid_dcm_paths = unique(valid_dcm_paths);
end

fprintf('  -> 找到 %d 个有效 .dcm 文件（去除黑名单、去重后）\n\n', ...
    length(valid_dcm_paths));

if isempty(valid_dcm_paths)
    fprintf('  ❌ 在 %s 下未找到任何 .dcm 文件\n', data_root);
    fprintf('     请检查:\n');
    fprintf('     1) 路径是否正确\n');
    fprintf('     2) DICOM 文件扩展名 (.dcm / .IMA / .001)\n');
    fprintf('     3) 用 SWI202606_dicom_scanner() 进一步诊断\n');
    error('未找到 DICOM 文件');
end

% ====== 智能归并到被试根目录 ======
fprintf('[2/4] 识别被试根目录...\n');

% 提取每个 dcm 文件的目录（系列目录）
series_dirs = unique(cellfun(@fileparts, valid_dcm_paths, 'UniformOutput', false));

% 找这些系列目录的父目录
parent_dirs = unique(cellfun(@(d) fileparts(d), series_dirs, 'UniformOutput', false));

% 规则 1: 如果某个 parent_dir 下有 ≥2 个系列目录 → 它是被试根目录
subject_roots = {};
for k = 1:length(parent_dirs)
    pdir = parent_dirs{k};
    if is_blacklisted_dir(pdir, blacklist), continue; end

    % 数这个目录下有多少系列
    cnt = sum(cellfun(@(s) strcmp(fileparts(s), pdir), series_dirs));

    if cnt >= 2
        if ~any(strcmp(subject_roots, pdir))
            subject_roots{end+1} = pdir;
        end
    end
end

% 规则 2: 如果 series_dir 本身就在 data_root 下 → 它本身就是被试根目录
for k = 1:length(series_dirs)
    s = series_dirs{k};
    parent = fileparts(s);
    if strcmpi(parent, data_root) || isempty(parent)
        if ~is_blacklisted_dir(s, blacklist) && ~any(strcmp(subject_roots, s))
            subject_roots{end+1} = s;
        end
    end
end

% 规则 3: 兜底 - 用 series_dirs 的 grandparent
if isempty(subject_roots)
    grandparent_dirs = unique(cellfun(@(d) fileparts(fileparts(d)), ...
                                      series_dirs, 'UniformOutput', false));
    for k = 1:length(grandparent_dirs)
        g = grandparent_dirs{k};
        if is_blacklisted_dir(g, blacklist), continue; end
        if ~any(strcmp(subject_roots, g))
            subject_roots{end+1} = g;
        end
    end
end

subject_roots = unique(subject_roots);
fprintf('  -> 识别出 %d 个被试根目录:\n', length(subject_roots));
for k = 1:length(subject_roots)
    fprintf('    [%d] %s\n', k, subject_roots{k});
end
fprintf('\n');

% ====== 读取每个被试的 DICOM 元数据 ======
fprintf('[3/4] 读取 DICOM 元数据...\n\n');

subjects = struct('name', {}, 'path', {}, 'group', {}, ...
    'patient_id', {}, 'patient_name', {}, 'patient_age', '', ...
    'patient_birth_date', '', 'patient_sex', '', ...
    'study_date', '', 'age_years', NaN);

for k = 1:length(subject_roots)
    spath = subject_roots{k};
    [~, sname] = fileparts(spath);

    fprintf('───────────────────────────────────────────────\n');
    fprintf('被试 %d: %s\n', k, sname);
    fprintf('路径: %s\n', spath);

    % 找这个被试的所有 dcm 文件
    subj_dcm_paths = valid_dcm_paths(cellfun(@(p) ...
        strcmp(fileparts(fileparts(p)), spath) || strcmp(fileparts(p), spath), ...
        valid_dcm_paths, 'UniformOutput', true));

    if isempty(subj_dcm_paths)
        fprintf('  ⚠️ 此目录下无 .dcm 文件\n\n');
        continue;
    end

    % 读第一个 DICOM
    fp = subj_dcm_paths{1};
    pid = ''; pname = ''; pbd = ''; page_raw = ''; psex = ''; sdate = '';
    age_years = NaN;

    try
        info = dicominfo(fp);
        pid = safe_str(info, 'PatientID', '');
        pname = safe_str(info, 'PatientName', '');
        pbd = safe_str(info, 'PatientBirthDate', '');
        page_raw = safe_str(info, 'PatientAge', '');
        psex = safe_str(info, 'PatientSex', '');
        sdate = safe_str(info, 'StudyDate', '');

        age_years = parse_age(page_raw, pbd, sdate);

        fprintf('  DICOM 文件: %s\n', fp);
        fprintf('  PatientID   : %s\n', pid);
        fprintf('  PatientName : %s\n', pname);
        if ~isempty(page_raw), fprintf('  PatientAge  : %s\n', page_raw); end
        if ~isempty(pbd),      fprintf('  BirthDate   : %s\n', pbd); end
        fprintf('  PatientSex  : %s\n', psex);
        fprintf('  StudyDate   : %s\n', sdate);
        if ~isnan(age_years)
            fprintf('  → 计算年龄: %.1f 岁\n', age_years);
        end
    catch ME
        fprintf('  ⚠️ dicominfo 失败: %s\n', ME.message);
    end

    % 默认分组 UNKNOWN（除非 ID/姓名 或 年龄明确）
    group = 'UNKNOWN';
    id_l = lower([pid ' ' pname]);
    if contains(id_l, {'old', 'elder', 'aged', 'senior'})
        group = 'ELDERLY';
    elseif contains(id_l, {'young', 'control', 'normal'})
        group = 'NORMAL';
    end

    % 🔴 新增: 基于年龄自动分组 (优先级最高)
    % 阈值: ≥60 岁 = ELDERLY, <60 岁 = NORMAL
    if isnan(age_years)
        fprintf('  ⚠️ 无年龄信息，group=UNKNOWN（需手动指定）\n');
    elseif age_years >= 60
        if strcmp(group, 'UNKNOWN')
            group = 'ELDERLY';
        end
        fprintf('  → 年龄 %.0f ≥ 60: 自动 ELDERLY\n', age_years);
    else
        if strcmp(group, 'UNKNOWN')
            group = 'NORMAL';
        end
        fprintf('  → 年龄 %.0f < 60: 自动 NORMAL\n', age_years);
    end

    subjects(end+1) = struct( ...
        'name', sname, 'path', spath, 'group', group, ...
        'patient_id', pid, 'patient_name', pname, ...
        'patient_age', page_raw, 'patient_birth_date', pbd, ...
        'patient_sex', psex, 'study_date', sdate, ...
        'age_years', age_years);
    fprintf('\n');
end

% ====== 报告 ======
fprintf('[4/4] 报告:\n\n');
fprintf('  %-25s %-12s %-15s %-15s %-8s %-10s\n', ...
    'Name', 'Group', 'PatientID', 'PatientName', 'Sex', 'Age');
fprintf('  %s\n', repmat('-', 1, 95));

for k = 1:length(subjects)
    s = subjects(k);
    age_str = 'N/A';
    if ~isnan(s.age_years), age_str = sprintf('%.0f', s.age_years); end
    fprintf('  %-25s %-12s %-15s %-15s %-8s %-10s\n', ...
        s.name, s.group, s.patient_id, s.patient_name, ...
        s.patient_sex, age_str);
end

n_normal  = sum(strcmp({subjects.group}, 'NORMAL'));
n_elderly = sum(strcmp({subjects.group}, 'ELDERLY'));
n_unk     = sum(strcmp({subjects.group}, 'UNKNOWN'));
fprintf('\n【分组统计】 NORMAL=%d  ELDERLY=%d  UNKNOWN=%d\n', ...
    n_normal, n_elderly, n_unk);

if n_normal == 0 || n_elderly == 0
    fprintf('\n需要手动指定 NORMAL/ELDERLY:\n');
    fprintf('  1) 跑 inspect_subjects 看更详细的 DICOM 元数据\n');
    fprintf('  2) 或手动设置:\n');
    for k = 1:length(subjects)
        fprintf('       subjects(%d).group = ''NORMAL'';  %% 或 ''ELDERLY''\n', k);
    end
end
end

%% =========================================================================
% 检查目录是否在黑名单中
%% =========================================================================
function tf = is_blacklisted_dir(dir_path, blacklist)
tf = false;
[~, name] = fileparts(dir_path);
for k = 1:length(blacklist)
    if strcmp(name, blacklist{k})
        tf = true;
        return;
    end
end
% 也检查路径中是否含 _scan_report 或 _qsm_
if ~isempty(strfind(dir_path, '_scan_report')) || ...
   ~isempty(strfind(dir_path, '_qsm_'))
    tf = true;
end
end

function v = safe_str(s, fname, default)
% 安全提取 DICOM 字段为字符串
% 处理 DICOM 特有的类型:
%   - PatientName 是 struct (FamilyName, GivenName, ...)
%   - 有些字段是 cell 数组
%   - 有些是 char/string
v = default;
if ~isfield(s, fname), return; end
val = s.(fname);
if isempty(val), return; end

% DICOM PatientName 是 struct
if isstruct(val)
    parts = {};
    fns = {'FamilyName', 'GivenName', 'MiddleName', ...
           'NamePrefix', 'NameSuffix', 'Ideographic', 'Phonetic'};
    for k = 1:length(fns)
        if isfield(val, fns{k}) && ~isempty(val.(fns{k}))
            v_str = val.(fns{k});
            if ischar(v_str)
                parts{end+1} = strtrim(v_str);
            elseif isstring(v_str)
                parts{end+1} = strtrim(char(v_str));
            end
        end
    end
    if isempty(parts)
        all_fn = fieldnames(val);
        for k = 1:length(all_fn)
            try
                v_str = val.(all_fn{k});
                if ischar(v_str) && ~isempty(v_str)
                    parts{end+1} = strtrim(v_str);
                end
            catch
            end
        end
    end
    if ~isempty(parts)
        v = strjoin(parts, ' ');
    end
    return;
end

% cell 数组
if iscell(val)
    if ~isempty(val) && ischar(val{1})
        v = strtrim(val{1});
        return;
    end
end

% char / string
if ischar(val)
    v = strtrim(val);
elseif isstring(val)
    v = strtrim(char(val));
end
end

function age = parse_age(page_raw, pbd, sdate)
age = NaN;
if ~isempty(page_raw)
    num_str = regexprep(page_raw, '[^0-9]', '');
    if ~isempty(num_str)
        age = str2double(num_str);
        return;
    end
end
if ~isempty(pbd) && ~isempty(sdate) && length(pbd)==8 && length(sdate)==8
    try
        birth = datenum(pbd, 'yyyymmdd');
        study = datenum(sdate, 'yyyymmdd');
        age = (study - birth) / 365.25;
    catch
    end
end
end
