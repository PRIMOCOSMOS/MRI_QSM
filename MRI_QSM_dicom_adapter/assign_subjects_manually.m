function subjects = assign_subjects_manually(data_root)
% assign_subjects_manually.m
% ============================================================================
%   应急脚本: 手动指定 NORMAL/ELDERLY 分组
%   当自动分组失败时使用
%
%   用法:
%     1) 先看 inspect_subjects 输出，确定谁是 normal，谁是 elderly
%     2) 编辑下面的 NORMAL_NAME 和 ELDERLY_NAME
%     3) 运行此脚本
%     4) 再运行 run_whqsm_comparison
% ============================================================================

if nargin < 1 || isempty(data_root)
    data_root = 'D:\MRI_PRO\MRILAB_X\20170327_qsm2016_recon_challenge\data_course\SWI202606';
end

% ====== 在这里手动指定 ======
NORMAL_NAME  = 'swi_subj1';   % ← 改成你的 normal 被试文件夹名
ELDERLY_NAME = 'swi_subj2';   % ← 改成你的 elderly 被试文件夹名
% =================================

fprintf('============================================================\n');
fprintf(' 手动指定被试分组\n');
fprintf(' NORMAL  : %s\n', NORMAL_NAME);
fprintf(' ELDERLY : %s\n', ELDERLY_NAME);
fprintf('============================================================\n\n');

subjects = discover_subjects(data_root);

% 按名称匹配设置 group
for k = 1:length(subjects)
    if strcmp(subjects(k).name, NORMAL_NAME)
        subjects(k).group = 'NORMAL';
        fprintf('✅ 设置 %s → NORMAL\n', subjects(k).name);
    elseif strcmp(subjects(k).name, ELDERLY_NAME)
        subjects(k).group = 'ELDERLY';
        fprintf('✅ 设置 %s → ELDERLY\n', subjects(k).name);
    else
        fprintf('⚠️  跳过 %s\n', subjects(k).name);
    end
end

fprintf('\n');
fprintf('【当前分组】\n');
for k = 1:length(subjects)
    fprintf('  %s : %s\n', subjects(k).name, subjects(k).group);
end

n_normal  = sum(strcmp({subjects.group}, 'NORMAL'));
n_elderly = sum(strcmp({subjects.group}, 'ELDERLY'));

if n_normal >= 1 && n_elderly >= 1
    fprintf('\n✅ 准备好！运行 run_whqsm_comparison() 即可。\n');
else
    fprintf('\n❌ 分组仍不完整，请检查 NORMAL_NAME 和 ELDERLY_NAME 是否正确\n');
end
end
