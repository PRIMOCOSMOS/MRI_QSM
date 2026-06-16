function DIAGNOSE_MAT_INTEGRITY(whichSubject)
% DIAGNOSE_MAT_INTEGRITY.m
% ============================================================================
% Check integrity of all WH-QSM output .mat files and report which are
% readable / corrupt, file sizes, and whether recovery from companion files
% is possible. Helps confirm WHETHER it is a save-logic bug vs a one-off
% interrupted write.
%
% Usage (adapter dir):
%   DIAGNOSE_MAT_INTEGRITY            % all subjects
%   DIAGNOSE_MAT_INTEGRITY('elderly')
% ============================================================================

if nargin < 1, whichSubject = 'all'; end

P = whqsm_local_paths();
outRoot = fullfile(P.dataRoot, '_qsm_comparison_results');
if exist(outRoot,'dir') ~= 7
    error('结果根目录不存在: %s', outRoot);
end

dirs = [dir(fullfile(outRoot,'normal_*')); dir(fullfile(outRoot,'elderly_*'))];
dirs = dirs([dirs.isdir]);

fprintf('\n================= .mat 完整性诊断 =================\n');
fprintf('根目录: %s\n', outRoot);

anyCorrupt = false;
for i = 1:numel(dirs)
    nm = dirs(i).name;
    if ~strcmpi(whichSubject,'all') && ~contains(lower(nm), lower(whichSubject))
        continue;
    end
    subDir = fullfile(dirs(i).folder, nm);
    fprintf('\n----- %s -----\n', nm);

    files = [dir(fullfile(subDir,'whqsm_*_complete.mat')); ...
             dir(fullfile(subDir,'chi_*.mat')); ...
             dir(fullfile(subDir,'qsm2016_format','data_full.mat'))];
    if isempty(files)
        fprintf('  (无 .mat 文件)\n'); continue;
    end
    for k = 1:numel(files)
        f = fullfile(files(k).folder, files(k).name);
        szMB = files(k).bytes/1e6;
        [ok, vars, err] = probe_mat(f);
        if ok
            fprintf('  ✓ %-32s %8.1f MB  vars: %s\n', files(k).name, szMB, strjoin(vars, ','));
        else
            anyCorrupt = true;
            fprintf('  ✗ %-32s %8.1f MB  损坏: %s\n', files(k).name, szMB, err);
        end
    end

    % recovery possibility for the complete file
    cFile = dir(fullfile(subDir,'whqsm_*_complete.mat'));
    if ~isempty(cFile)
        [okc,~,~] = probe_mat(fullfile(cFile(1).folder, cFile(1).name));
        if ~okc
            chiOk = ~isempty(dir(fullfile(subDir,'chi_*.mat')));
            dataOk = exist(fullfile(subDir,'qsm2016_format','data_full.mat'),'file')==2;
            if chiOk && dataOk
                fprintf('  → 可从 chi_*.mat + data_full.mat 恢复(RUN_CHISEP_ONLY 会自动恢复)。\n');
            else
                fprintf('  → 无法恢复(缺备用文件)。需对该被试重跑 WH-QSM。\n');
            end
        end
    end
end

fprintf('\n=================================================\n');
if anyCorrupt
    fprintf(['结论: 存在损坏文件。本版已改用 save_mat_atomic(原子+校验)保存，\n' ...
             '可避免再次产生半截文件。已损坏的旧文件需重跑或用备用文件恢复。\n']);
else
    fprintf('结论: 所有 .mat 文件可正常读取。\n');
end
end

function [ok, vars, err] = probe_mat(f)
ok = false; vars = {}; err = '';
try
    info = whos('-file', f);
    if isempty(info)
        err = '无变量(空文件)';
    else
        vars = {info.name};
        ok = true;
    end
catch ME
    err = ME.message;
end
end
