function inspect_subjects(data_root)
% inspect_subjects.m  (v3 - 稳健版, 用 dir(**) 单次扫描)
% ============================================================================
%   自动从 DICOM 元数据提取每个被试的年龄/性别/ID
%   使用 dir(**) 一次性扫描，避免递归 bug
% ============================================================================

if nargin < 1 || isempty(data_root)
    data_root = 'D:\MRI_PRO\MRILAB_X\20170327_qsm2016_recon_challenge\data_course';
end

fprintf('\n');
fprintf('╔══════════════════════════════════════════════════════════════╗\n');
fprintf('║  被试信息深度检查 (v3 - 单次全扫描)                               ║\n');
fprintf('╚══════════════════════════════════════════════════════════════╝\n\n');

if ~exist(data_root, 'dir')
    error('路径不存在: %s', data_root);
end

% ====== 单次全扫描所有 .dcm 文件 ======
fprintf('[1/3] 全局扫描所有 .dcm 文件...\n');
blacklist = {'_scan_report', '_qsm_comparison_results', 'output', ...
             'results', 'figures', 'Models'};

all_dcm = dir(fullfile(data_root, '**', '*.dcm'));
valid_dcm_paths = {};
for k = 1:length(all_dcm)
    fp = fullfile(all_dcm(k).folder, all_dcm(k).name);
    skip = false;
    for b = 1:length(blacklist)
        if ~isempty(strfind(fp, ['\' blacklist{b} '\'])) || ...
           ~isempty(strfind(fp, ['/' blacklist{b} '/']))
            skip = true; break;
        end
    end
    if ~skip
        valid_dcm_paths{end+1} = fp;
    end
end
if ~isempty(valid_dcm_paths), valid_dcm_paths = unique(valid_dcm_paths); end

fprintf('  -> 找到 %d 个有效 .dcm 文件\n\n', length(valid_dcm_paths));

if isempty(valid_dcm_paths)
    fprintf('  ❌ 未找到任何 .dcm 文件\n');
    return;
end

% ====== 识别被试根目录 ======
fprintf('[2/3] 识别被试根目录...\n');

series_dirs = unique(cellfun(@fileparts, valid_dcm_paths, 'UniformOutput', false));
parent_dirs = unique(cellfun(@(d) fileparts(d), series_dirs, 'UniformOutput', false));

subject_roots = {};
for k = 1:length(parent_dirs)
    pdir = parent_dirs{k};
    [~, pname] = fileparts(pdir);
    if any(strcmp(pname, blacklist)), continue; end

    cnt = sum(cellfun(@(s) strcmp(fileparts(s), pdir), series_dirs));
    if cnt >= 2
        subject_roots{end+1} = pdir;
    end
end

% 兜底
if isempty(subject_roots)
    subject_roots = parent_dirs;
    subject_roots = subject_roots(~cellfun(@isempty, subject_roots));
end

subject_roots = unique(subject_roots);
fprintf('  -> %d 个被试根目录:\n', length(subject_roots));
for k = 1:length(subject_roots)
    fprintf('    [%d] %s\n', k, subject_roots{k});
end
fprintf('\n');

% ====== 逐个分析 ======
fprintf('[3/3] 分析每个被试...\n\n');

subjects_info = struct('name', {}, 'path', {}, ...
    'patient_id', {}, 'patient_name', {}, ...
    'patient_birth_date', '', 'patient_age_raw', '', ...
    'patient_sex', '', 'study_date', '', 'study_description', '', ...
    'age_years', NaN, 'age_source', '', 'group_guess', '');

for k = 1:length(subject_roots)
    spath = subject_roots{k};
    [~, sname] = fileparts(spath);

    fprintf('───────────────────────────────────────────────\n');
    fprintf('被试 %d: %s\n', k, sname);
    fprintf('路径: %s\n', spath);
    fprintf('───────────────────────────────────────────────\n');

    % 找该被试下的 dcm 文件
    subj_dcm = valid_dcm_paths(cellfun(@(p) ...
        strcmp(fileparts(fileparts(p)), spath) || strcmp(fileparts(p), spath), ...
        valid_dcm_paths));

    if isempty(subj_dcm)
        fprintf('  ⚠️ 无 DICOM，跳过\n\n');
        continue;
    end

    try
        info = dicominfo(subj_dcm{1});
    catch ME
        fprintf('  ❌ dicominfo 失败: %s\n\n', ME.message);
        continue;
    end

    pid = safe_str(info, 'PatientID', '?');
    pname = safe_str(info, 'PatientName', '?');
    pbd = safe_str(info, 'PatientBirthDate', '');
    page_raw = safe_str(info, 'PatientAge', '');
    psex = safe_str(info, 'PatientSex', '?');
    sdate = safe_str(info, 'StudyDate', '');
    sdesc = safe_str(info, 'StudyDescription', '');

    fprintf('  DICOM 文件    : %s\n', subj_dcm{1});
    fprintf('  PatientID     : %s\n', pid);
    fprintf('  PatientName   : %s\n', pname);
    fprintf('  PatientBirthDate: "%s"\n', pbd);
    fprintf('  PatientAge    : "%s"\n', page_raw);
    fprintf('  PatientSex    : %s\n', psex);
    fprintf('  StudyDate     : %s\n', sdate);
    fprintf('  StudyDesc     : %s\n', sdesc);

    % 算年龄
    age_years = NaN;
    age_source = '';
    if ~isempty(page_raw)
        num_str = regexprep(page_raw, '[^0-9]', '');
        if ~isempty(num_str)
            age_years = str2double(num_str);
            age_source = 'PatientAge';
        end
    end
    if isnan(age_years) && ~isempty(pbd) && ~isempty(sdate) && ...
       length(pbd)==8 && length(sdate)==8
        try
            birth = datenum(pbd, 'yyyymmdd');
            study = datenum(sdate, 'yyyymmdd');
            age_years = (study - birth) / 365.25;
            age_source = 'BirthDate+StudyDate';
        catch
        end
    end

    if ~isnan(age_years)
        fprintf('  → 计算年龄: %.1f 岁 (来源: %s)\n', age_years, age_source);
    else
        fprintf('  → ⚠️ 无法确定年龄\n');
    end

    % 自动猜测
    group_guess = 'UNKNOWN';
    if ~isnan(age_years)
        if age_years < 40
            group_guess = 'YOUNG/NORMAL';
        elseif age_years < 60
            group_guess = 'MIDDLE_AGE';
        else
            group_guess = 'ELDERLY';
        end
        fprintf('  → 猜测分组: %s\n', group_guess);
    end
    id_l = lower([pid ' ' pname]);
    if contains(id_l, {'old','elder','aged','senior'})
        group_guess = 'ELDERLY';
        fprintf('  → ID 含 old/elder: 强制 ELDERLY\n');
    elseif contains(id_l, {'young','control','normal'})
        group_guess = 'NORMAL';
        fprintf('  → ID 含 normal/control: 强制 NORMAL\n');
    end

    subjects_info(k) = struct( ...
        'name', sname, 'path', spath, ...
        'patient_id', pid, 'patient_name', pname, ...
        'patient_birth_date', pbd, 'patient_age_raw', page_raw, ...
        'patient_sex', psex, 'study_date', sdate, ...
        'study_description', sdesc, ...
        'age_years', age_years, 'age_source', age_source, ...
        'group_guess', group_guess);
    fprintf('\n');
end

% ====== 总结 ======
fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf('总结\n');
fprintf('═══════════════════════════════════════════════════════════════\n\n');

fprintf('%-25s %-15s %-12s %-10s %-15s\n', 'Subject', 'PatientID', 'Age', 'Sex', 'Guess');
fprintf('%s\n', repmat('-', 1, 85));

has_normal = false; has_elderly = false;
for k = 1:length(subjects_info)
    s = subjects_info(k);
    age_str = 'N/A';
    if ~isnan(s.age_years), age_str = sprintf('%.0f', s.age_years); end
    fprintf('%-25s %-15s %-12s %-10s %-15s\n', ...
        s.name, s.patient_id, age_str, s.patient_sex, s.group_guess);
    if strcmp(s.group_guess, 'ELDERLY'), has_elderly = true; end
    if any(strcmp(s.group_guess, {'NORMAL','YOUNG/NORMAL'})), has_normal = true; end
end

fprintf('\n');
if has_normal && has_elderly
    fprintf('✅ 系统已自动识别 NORMAL 和 ELDERLY\n');
    fprintf('   接下来直接运行: run_whqsm_comparison()\n');
else
    fprintf('⚠️ 无法自动判断。需要你手动指定:\n');
    fprintf('   >> subjects = discover_subjects\n');
    fprintf('   >> subjects(1).group = ''NORMAL''   %% swi_subj1\n');
    fprintf('   >> subjects(2).group = ''ELDERLY''  %% swi_subj2\n');
    fprintf('   >> run_whqsm_comparison\n');
end

% 保存
out_dir = fullfile(data_root, '_scan_report');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
save(fullfile(out_dir, 'subjects_info.mat'), 'subjects_info');
fprintf('\n💾 保存到: %s\n', fullfile(out_dir, 'subjects_info.mat'));
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
        % 兜底：取所有 char/string 类型字段
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
