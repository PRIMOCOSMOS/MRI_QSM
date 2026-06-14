function check_and_force_update()
% check_and_force_update.m
% ============================================================================
%   诊断 + 强制更新脚本
%   检查 MATLAB 是否真的用了最新的 dicom_loader_subject.m
%   解决 path 缓存导致的"修改不生效"问题
% ============================================================================

fprintf('\n');
fprintf('╔══════════════════════════════════════════════════════════════╗\n');
fprintf('║  检查 MATLAB 是否使用了最新的 dicom_loader_subject.m            ║\n');
fprintf('╚══════════════════════════════════════════════════════════════╝\n\n');

% ====== 1. 查找所有 dicom_loader_subject.m ======
fprintf('[1/5] 查找所有 dicom_loader_subject.m...\n');

which_paths = {};
all_paths = {};

% 方法 A: which -all
try
    [~, all_paths_A] = which('dicom_loader_subject', '-all');
    if ~isempty(all_paths_A)
        for k = 1:length(all_paths_A)
            all_paths{end+1} = all_paths_A{k};
        end
    end
catch
end

% 方法 B: 在 path 中搜索
p = path;
sep = pathsep;
parts = strsplit(p, sep);
for k = 1:length(parts)
    if isempty(parts{k}), continue; end
    candidate = fullfile(parts{k}, 'dicom_loader_subject.m');
    if exist(candidate, 'file') == 2
        all_paths{end+1} = candidate;
    end
end

% 去重
all_paths = unique(all_paths);

fprintf('  找到 %d 个文件:\n', length(all_paths));
for k = 1:length(all_paths)
    info = dir(all_paths{k});
    fprintf('    [%d] %s (修改: %s)\n', k, all_paths{k}, info.date);
end

% ====== 2. 检查哪个会被 MATLAB 实际调用 ======
fprintf('\n[2/5] MATLAB 实际使用哪个?\n');
[~, primary_path] = which('dicom_loader_subject');
if isempty(primary_path)
    fprintf('  ❌ MATLAB 找不到 dicom_loader_subject！请检查路径\n');
    return;
end
fprintf('  ✅ MATLAB 用: %s\n', primary_path);

% ====== 3. 验证文件内容 ======
fprintf('\n[3/5] 验证文件内容是否最新...\n');
fid = fopen(primary_path, 'r');
if fid == -1
    fprintf('  ❌ 无法打开文件\n');
    return;
end
content = fread(fid, inf, 'uint8=>char')';
fclose(fid);

checks = {
    'vol4d = build_4d_volume(files, slope, intercept)',  '✅ 修复 (无 echo_info)';
    '[vol4d, echo_info] = build_4d_volume',              '❌ 旧版 (有 echo_info)';
    'function v = safe_dicom_num(info, fname, default)', '✅ 修复 (safe_dicom_num)';
    'function vol4d = build_4d_volume',                  '✅ 修复 (build_4d_volume)';
    'combined = double(echo_nums) * 1e6',                '✅ 修复 (lex sort)';
};

fprintf('  文件内容检查:\n');
for k = 1:size(checks, 1)
    pattern = checks{k, 1};
    label = checks{k, 2};
    if ~isempty(strfind(content, pattern))
        fprintf('    %s 包含: "%s"\n', label, pattern);
    end
end

% ====== 4. 强制更新 ======
fprintf('\n[4/5] 强制更新 MATLAB 缓存...\n');
try
    rehash;
    fprintf('  ✅ rehash 完成\n');
catch ME
    fprintf('  ⚠️ rehash 失败: %s\n', ME.message);
end

try
    clear functions;
    fprintf('  ✅ clear functions 完成\n');
catch ME
    fprintf('  ⚠️ clear functions 失败: %s\n', ME.message);
end

% 重新 which
[~, new_path] = which('dicom_loader_subject');
if strcmp(new_path, primary_path)
    fprintf('  ✅ MATLAB 现在用: %s\n', new_path);
else
    fprintf('  ⚠️ 路径变了: %s → %s\n', primary_path, new_path);
end

% ====== 5. 总结 ======
fprintf('\n[5/5] 总结\n');

% 检查关键修复
if ~isempty(strfind(content, '[vol4d, echo_info] = build_4d_volume'))
    fprintf('  ❌ 你的工作区文件还有 OLD BUG！\n');
    fprintf('     请确认 dicom_loader_subject.m 是最新的版本\n');
    fprintf('     本工作区路径: %s\n', primary_path);
elseif ~isempty(strfind(content, 'vol4d = build_4d_volume(files, slope, intercept)'))
    fprintf('  ✅ 工作区文件已是最新版\n');
    fprintf('     关键修复: vol4d = build_4d_volume (无 echo_info)\n');
else
    fprintf('  ⚠️ 文件内容未知，可能还有其他问题\n');
end

fprintf('\n');
fprintf('╔══════════════════════════════════════════════════════════════╗\n');
fprintf('║  建议\n');
fprintf('╚══════════════════════════════════════════════════════════════╝\n');
fprintf('  1) 在 MATLAB 中运行此脚本，检查上面的输出\n');
fprintf('  2) 如果文件有多个版本，请删除旧版本或调整 path\n');
fprintf('  3) 如果还是不行，尝试: clear all; rehash; run_whqsm_comparison\n');
fprintf('\n');
end
