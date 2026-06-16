function RUN_CHISEP_ONLY(whichSubject)
% RUN_CHISEP_ONLY.m
% Root-level wrapper: run susceptibility separation only from existing WH-QSM
% outputs (whqsm_*_complete.mat). 适合手头已有预处理/反演结果、不想重跑
% DICOM/WH-QSM 的情况。
%
% 行为由 whqsm_local_paths.m 的 P.chisepRunMethodCompare 控制:
%   true  : 运行"方法对比"(深度学习 χ-sepnet vs 传统优化)
%   false : 只运行单一方法 (cfg.sep.adapter_function 指定的那个)
%
% 用法:
%   RUN_CHISEP_ONLY              % normal_* 和 elderly_* 都处理
%   RUN_CHISEP_ONLY('normal')
%   RUN_CHISEP_ONLY('D:\...\normal_subjX')
if nargin < 1
    whichSubject = 'all';
end
repoRoot = fileparts(mfilename('fullpath'));
addpath(fullfile(repoRoot, 'modules'), '-begin');
addpath(fullfile(repoRoot, 'MRI_QSM_dicom_adapter'), '-begin');
addpath(fullfile(repoRoot, 'Utils_self'), '-begin');
run_chisep_only_impl(whichSubject);
end
