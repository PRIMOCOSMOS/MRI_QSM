function subjects = discover_subjects(data_root)
% discover_subjects.m (v5 - WH-QSM real-data robust discovery)
% ============================================================================
% Discover subject roots under data_root and assign NORMAL / ELDERLY using
% DICOM age metadata when available. Supports .dcm/.dicom/.IMA/.001 and files
% without extension but with DICM magic.
% ============================================================================

if nargin < 1 || isempty(data_root)
    data_root = 'D:\MRI_PRO\MRILAB_X\20170327_qsm2016_recon_challenge\data_course';
end

data_root = char(data_root);

fprintf('============================================================\n');
fprintf(' Subject discovery for WH-QSM real-data pipeline (v5)\n');
fprintf(' Root: %s\n', data_root);
fprintf('============================================================\n\n');

if exist(data_root, 'dir') ~= 7
    error('data_root does not exist: %s', data_root);
end

blacklist = {'_scan_report','_qsm_comparison_results','output','results', ...
             'figures','Models','model','temp','tmp','.','..'};

fprintf('[1/4] Scanning DICOM files...\n');
all_dcm = discover_dicom_files(data_root, blacklist);
fprintf('  -> %d valid DICOM-like files found.\n\n', numel(all_dcm));
if isempty(all_dcm)
    error('No DICOM files found under %s. Check extension or directory.', data_root);
end

fprintf('[2/4] Inferring subject roots...\n');
subject_roots = infer_subject_roots(all_dcm, data_root, blacklist);
fprintf('  -> %d subject root(s):\n', numel(subject_roots));
for k = 1:numel(subject_roots)
    fprintf('    [%d] %s\n', k, subject_roots{k});
end
fprintf('\n');

fprintf('[3/4] Reading representative DICOM metadata...\n\n');
subjects = struct('name', {}, 'path', {}, 'group', {}, ...
    'patient_id', {}, 'patient_name', {}, 'patient_age', {}, ...
    'patient_birth_date', {}, 'patient_sex', {}, 'study_date', {}, ...
    'age_years', {});

for k = 1:numel(subject_roots)
    spath = subject_roots{k};
    [~, sname] = fileparts(spath);
    subj_files = all_dcm(starts_with_path(all_dcm, spath));
    if isempty(subj_files)
        continue;
    end

    fprintf('───────────────────────────────────────────────\n');
    fprintf('Subject %d: %s\n', k, sname);
    fprintf('Path     : %s\n', spath);

    pid = ''; pname = ''; pbd = ''; page_raw = ''; psex = ''; sdate = '';
    age_years = NaN;

    info = [];
    for f = 1:min(numel(subj_files), 20)
        try
            info = dicominfo(subj_files{f});
            break;
        catch
        end
    end

    if isempty(info)
        fprintf('  ⚠️  No readable DICOM metadata; group UNKNOWN.\n');
    else
        pid = safe_str(info, 'PatientID', '');
        pname = safe_str(info, 'PatientName', '');
        pbd = safe_str(info, 'PatientBirthDate', '');
        page_raw = safe_str(info, 'PatientAge', '');
        psex = safe_str(info, 'PatientSex', '');
        sdate = safe_str(info, 'StudyDate', '');
        age_years = parse_age(page_raw, pbd, sdate);

        fprintf('  Example DICOM : %s\n', subj_files{f});
        fprintf('  PatientID     : %s\n', pid);
        fprintf('  PatientName   : %s\n', pname);
        fprintf('  PatientAge    : %s\n', page_raw);
        fprintf('  BirthDate     : %s\n', pbd);
        fprintf('  PatientSex    : %s\n', psex);
        fprintf('  StudyDate     : %s\n', sdate);
        if ~isnan(age_years)
            fprintf('  Computed age  : %.1f years\n', age_years);
        else
            fprintf('  Computed age  : N/A\n');
        end
    end

    group = infer_group(pid, pname, sname, age_years);
    fprintf('  -> Group      : %s\n\n', group);

    subjects(end+1) = struct( ...
        'name', sname, 'path', spath, 'group', group, ...
        'patient_id', pid, 'patient_name', pname, ...
        'patient_age', page_raw, 'patient_birth_date', pbd, ...
        'patient_sex', psex, 'study_date', sdate, ...
        'age_years', age_years); %#ok<AGROW>
end

fprintf('[4/4] Summary:\n\n');
fprintf('  %-25s %-12s %-15s %-15s %-8s %-10s\n', ...
    'Name', 'Group', 'PatientID', 'PatientName', 'Sex', 'Age');
fprintf('  %s\n', repmat('-', 1, 95));
for k = 1:numel(subjects)
    s = subjects(k);
    age_str = 'N/A';
    if ~isnan(s.age_years), age_str = sprintf('%.0f', s.age_years); end
    fprintf('  %-25s %-12s %-15s %-15s %-8s %-10s\n', ...
        s.name, s.group, s.patient_id, s.patient_name, s.patient_sex, age_str);
end

n_normal  = sum(strcmp({subjects.group}, 'NORMAL'));
n_elderly = sum(strcmp({subjects.group}, 'ELDERLY'));
n_unknown = sum(strcmp({subjects.group}, 'UNKNOWN'));
fprintf('\nGroup counts: NORMAL=%d  ELDERLY=%d  UNKNOWN=%d\n', n_normal, n_elderly, n_unknown);

if n_normal == 0 || n_elderly == 0
    fprintf('\n⚠️  NORMAL and ELDERLY were not both identified.\n');
    fprintf('   Use inspect_subjects(data_root), then rename folders or edit age metadata / manual assignment.\n');
end
end

%% =========================================================================
function files = discover_dicom_files(root_dir, blacklist)
files = {};
try
    all_files = dir(fullfile(root_dir, '**', '*'));
catch
    all_files = dir(root_dir);
end
all_files = all_files(~[all_files.isdir]);
seen = containers.Map('KeyType', 'char', 'ValueType', 'logical');
for k = 1:numel(all_files)
    fp = fullfile(all_files(k).folder, all_files(k).name);
    if is_blacklisted_path(fp, blacklist), continue; end
    if isKey(seen, fp), continue; end
    [~, ~, ext] = fileparts(all_files(k).name);
    is_dcm = any(strcmpi(ext, {'.dcm','.dicom','.ima','.001','.img'}));
    if ~is_dcm
        is_dcm = has_dicom_magic(fp);
    end
    if is_dcm
        files{end+1} = fp; %#ok<AGROW>
        seen(fp) = true;
    end
end
end

function tf = has_dicom_magic(fp)
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

function subject_roots = infer_subject_roots(files, data_root, blacklist)
series_dirs = unique(cellfun(@fileparts, files, 'UniformOutput', false));
parent_dirs = unique(cellfun(@fileparts, series_dirs, 'UniformOutput', false));
subject_roots = {};

% Typical layout: data_root / subject / series / files
for k = 1:numel(parent_dirs)
    p = parent_dirs{k};
    if isempty(p) || is_blacklisted_path(p, blacklist), continue; end
    cnt_series = sum(cellfun(@(s) strcmp(fileparts(s), p), series_dirs));
    if cnt_series >= 1 && ~strcmpi(p, data_root)
        subject_roots{end+1} = p; %#ok<AGROW>
    end
end

% If files are directly under data_root/series, treat each series dir as one subject candidate.
if isempty(subject_roots)
    for k = 1:numel(series_dirs)
        s = series_dirs{k};
        parent = fileparts(s);
        if strcmpi(parent, data_root) && ~is_blacklisted_path(s, blacklist)
            subject_roots{end+1} = s; %#ok<AGROW>
        end
    end
end

% If data_root itself contains multiple series and appears to be a single subject.
if isempty(subject_roots)
    subject_roots = {data_root};
end

subject_roots = unique(subject_roots);
end

function tf = starts_with_path(paths, root)
tf = false(size(paths));
root_norm = normalize_path_local(root);
for i = 1:numel(paths)
    p_norm = normalize_path_local(paths{i});
    is_prefix = strncmpi(p_norm, root_norm, length(root_norm));
    if is_prefix && length(p_norm) > length(root_norm)
        next_char = p_norm(length(root_norm)+1);
        is_prefix = next_char == filesep || next_char == '/' || next_char == '\';
    end
    tf(i) = is_prefix || strcmpi(p_norm, root_norm);
end
end

function p = normalize_path_local(p)
p = char(p);
p = strrep(p, '/', filesep);
p = strrep(p, '\', filesep);
while length(p) > 1 && (p(end) == '/' || p(end) == '\')
    p(end) = [];
end
end

function tf = is_blacklisted_path(pathstr, blacklist)
tf = false;
parts = regexp(pathstr, '[\\/]+', 'split');
for i = 1:numel(parts)
    for j = 1:numel(blacklist)
        if strcmpi(parts{i}, blacklist{j})
            tf = true;
            return;
        end
    end
end
if contains(pathstr, '_scan_report') || contains(pathstr, '_qsm_comparison_results')
    tf = true;
end
end

function group = infer_group(pid, pname, sname, age_years)
text = lower([pid ' ' pname ' ' sname]);
if contains_any(text, {'old','elder','aged','senior'})
    group = 'ELDERLY';
elseif contains_any(text, {'young','control','normal','adult'})
    group = 'NORMAL';
elseif ~isnan(age_years) && age_years >= 60
    group = 'ELDERLY';
elseif ~isnan(age_years) && age_years < 60
    group = 'NORMAL';
else
    group = 'UNKNOWN';
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

function v = safe_str(s, fname, default)
v = default;
if ~isfield(s, fname) || isempty(s.(fname)), return; end
val = s.(fname);
if isstruct(val)
    parts = {};
    fns = fieldnames(val);
    for k = 1:numel(fns)
        item = val.(fns{k});
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
elseif iscell(val) && ~isempty(val) && ischar(val{1})
    v = strtrim(val{1});
elseif isnumeric(val) && isscalar(val)
    v = num2str(val);
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
if ~isempty(pbd) && ~isempty(sdate) && length(pbd) == 8 && length(sdate) == 8
    try
        birth = datenum(pbd, 'yyyymmdd');
        study = datenum(sdate, 'yyyymmdd');
        age = (study - birth) / 365.25;
    catch
        age = NaN;
    end
end
end
