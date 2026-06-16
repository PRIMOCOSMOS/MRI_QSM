function RUN_WHQSM_LAMBDA_SWEEP(whichSubject, lambdas)
% RUN_WHQSM_LAMBDA_SWEEP.m
% ============================================================================
% 客观定位 "QSM 脑内偏淡" 是否由正则化(lambda)过强引起。
%
% 对一个已有 WH-QSM 结果的被试，用一组 lambda 重新跑 WH-QSM 重建，输出：
%   1) 各 lambda 的 QSM 三平面对比图（统一窗位）
%   2) 全脑 χ 的标准差 / p1 / p99（直方图宽度，反映对比强弱）
%   3) 若能找到深部核团 ROI（mask 文件），给出 ROI 均值 ppm 表
%
% 判定方法：
%   - lambda 增大 → std 下降、p99 下降 → 越平滑越"淡"。
%   - 选 std/对比适中、且 streaking 可接受 的 lambda。
%
% 用法（项目根目录）：
%   RUN_WHQSM_LAMBDA_SWEEP                       % 第一个被试, 默认 lambda 组
%   RUN_WHQSM_LAMBDA_SWEEP('normal')
%   RUN_WHQSM_LAMBDA_SWEEP('elderly', [5e-4 3e-4 2e-4 1e-4])
% ============================================================================

if nargin < 1 || isempty(whichSubject), whichSubject = 'all'; end
if nargin < 2 || isempty(lambdas),      lambdas = [3e-4 2e-4 1e-4 5e-5 2e-5]; end

repoRoot = fileparts(mfilename('fullpath'));
addpath(fullfile(repoRoot, 'modules'), '-begin');
addpath(fullfile(repoRoot, 'MRI_QSM_dicom_adapter'), '-begin');
addpath(fullfile(repoRoot, 'Utils_self'), '-begin');

P = whqsm_local_paths();
if isfield(P,'sepiaRoot') && exist(P.sepiaRoot,'dir')==7
    addpath(P.sepiaRoot,'-begin'); addpath(genpath(P.sepiaRoot),'-begin');
end

outRoot = fullfile(P.dataRoot, '_qsm_comparison_results');
subDirs = resolve_subject_dirs(outRoot, whichSubject);
if isempty(subDirs)
    error('未找到被试结果目录（先跑一次 WH-QSM）：%s', outRoot);
end

clim = getf(P,'qsmDisplayClim',[-0.10 0.10]);

for s = 1:numel(subDirs)
    subDir = subDirs{s};
    fprintf('\n==================== %s ====================\n', subDir);
    [data, cfg] = load_subject(subDir, P);

    sweepDir = fullfile(subDir, 'lambda_sweep');
    if ~exist(sweepDir,'dir'), mkdir(sweepDir); end
    cfg.resultDir = sweepDir;

    Mask = logical(data.Mask);
    nL = numel(lambdas);
    chis = cell(1,nL);
    stats = zeros(nL,3);   % [std p1 p99]

    for i = 1:nL
        lam = lambdas(i);
        fprintf('\n--- lambda = %.3g (%d/%d) ---\n', lam, i, nL);
        cfg.whqsm.lambda  = lam;
        cfg.whqsm.beta    = getf2(P,'whqsmBeta',150);
        cfg.whqsm.muh     = getf2(P,'whqsmMuh',5);
        cfg.whqsm.maxiter = getf2(P,'whqsmMaxIter',200);
        cfg.whqsm.tol     = getf2(P,'whqsmTol',1e-5);
        try
            chi = mod_whqsm_reconstruction(data, cfg);
        catch ME
            warning('lambda=%.3g 失败: %s', lam, ME.message);
            continue;
        end
        chi(~Mask) = 0;
        chis{i} = chi;
        v = chi(Mask); v = v(isfinite(v));
        stats(i,:) = [std(v), prctile(v,1), prctile(v,99)];
        save(fullfile(sweepDir, sprintf('chi_lambda_%.3g.mat', lam)), 'chi', 'lam', '-v7.3');
    end

    % ---- 统计表 ----
    fprintf('\n  lambda      std(ppm)    p1(ppm)    p99(ppm)\n');
    for i = 1:nL
        fprintf('  %-10.3g  %-9.5f  %-9.5f  %-9.5f\n', lambdas(i), stats(i,1), stats(i,2), stats(i,3));
    end
    T = array2table([lambdas(:) stats], 'VariableNames', {'lambda','std_ppm','p1_ppm','p99_ppm'});
    writetable(T, fullfile(sweepDir, 'lambda_sweep_stats.csv'));

    % ---- 文献参考 + "肉眼淡 vs 数值低" 判读 ----
    fprintf(['\n  [文献参考] 3T 健康成人深部核团 χ 大致量级(ppm):\n' ...
             '    苍白球 GP ~0.10-0.18 | 红核 RN ~0.08-0.14 | 黑质 SN ~0.08-0.14\n' ...
             '    壳核 Put ~0.02-0.06 | 尾状核 CN ~0.02-0.05 | 白质 ~ -0.02~0.0\n']);
    fprintf(['  判读: 比较各 lambda 的 p99 与上面"铁核上限(~0.15-0.18)"。\n' ...
             '    - 若 p99 明显低于 0.10 -> 可能确实过平滑(lambda 偏大)，继续调小。\n' ...
             '    - 若 p99 已达 0.12-0.18 但图仍"淡" -> 是显示窗位问题，缩窄 clim 即可。\n']);
    fprintf('  当前对比图显示窗位 clim = [%.3g %.3g] ppm\n', clim(1), clim(2));

    % ---- 可选 ROI（深部核团）----
    roiTable = try_roi_table(subDir, chis, lambdas, Mask);
    if ~isempty(roiTable)
        writetable(roiTable, fullfile(sweepDir, 'lambda_sweep_roi.csv'));
        disp(roiTable);
    end

    % ---- 对比图 ----
    make_compare_figure(chis, lambdas, Mask, clim, sweepDir);
    fprintf('\n结果已保存到: %s\n', sweepDir);
end

fprintf(['\n判读建议: lambda 增大通常 std/p99 下降(更平滑更淡)。\n' ...
         '选 std 适中、深部核团对比足够、且 streaking 可接受的 lambda，\n' ...
         '然后把 P.whqsmLambda 设成该值。\n']);
end

%% ========================================================================
function [data, cfg] = load_subject(subDir, P)
[~, name] = fileparts(subDir);
if startsWith(lower(name),'elderly_'), label='elderly';
elseif startsWith(lower(name),'normal_'), label='normal'; else, label='subject'; end
f = fullfile(subDir, ['whqsm_' label '_complete.mat']);
if exist(f,'file')~=2
    d = dir(fullfile(subDir,'whqsm_*_complete.mat'));
    if isempty(d), error('未找到 whqsm_*_complete.mat: %s', subDir); end
    f = fullfile(d(1).folder, d(1).name);
end
S = load(f);
if ~isfield(S,'data'), error('complete.mat 缺少 data: %s', f); end
data = S.data;
if isfield(S,'cfg'), cfg = S.cfg; else, cfg = struct(); end
end

function subDirs = resolve_subject_dirs(outRoot, whichSubject)
whichSubject = char(whichSubject);
if exist(whichSubject,'dir')==7, subDirs = {whichSubject}; return; end
allD = [dir(fullfile(outRoot,'normal_*')); dir(fullfile(outRoot,'elderly_*'))];
allD = allD([allD.isdir]); subDirs = {};
for i=1:numel(allD)
    nm = allD(i).name;
    if strcmpi(whichSubject,'all') || contains(lower(nm),lower(whichSubject))
        subDirs{end+1} = fullfile(allD(i).folder, nm); %#ok<AGROW>
    end
end
end

function T = try_roi_table(subDir, chis, lambdas, Mask)
T = [];
% 尝试常见 ROI/atlas 文件（标签体）
cand = [dir(fullfile(subDir,'*label*.mat')); dir(fullfile(subDir,'*roi*.mat')); ...
        dir(fullfile(subDir,'*atlas*.mat'))];
labels = []; names = {};
for i=1:numel(cand)
    try
        L = load(fullfile(cand(i).folder, cand(i).name));
        fn = fieldnames(L);
        for j=1:numel(fn)
            v = L.(fn{j});
            if isnumeric(v) && isequal(size(v), size(Mask)) && max(v(:))>1
                labels = round(v); break;
            end
        end
    catch
    end
    if ~isempty(labels), break; end
end
if isempty(labels), return; end
ids = unique(labels(labels>0));
rows = {};
for i=1:numel(chis)
    if isempty(chis{i}), continue; end
    chi = chis{i};
    for k=1:numel(ids)
        m = (labels==ids(k)) & Mask;
        if nnz(m) < 10, continue; end
        rows(end+1,:) = {lambdas(i), ids(k), mean(chi(m)), std(chi(m)), nnz(m)}; %#ok<AGROW>
    end
end
if isempty(rows), return; end
T = cell2table(rows, 'VariableNames', {'lambda','roi_id','mean_ppm','std_ppm','nvox'});
end

function make_compare_figure(chis, lambdas, Mask, clim, sweepDir)
valid = find(~cellfun(@isempty, chis));
if isempty(valid), return; end
sz = round(size(Mask,3)/2);
% find a representative slice with most brain
[~, sz] = max(squeeze(sum(sum(Mask,1),2)));
fig = figure('Color','w','Position',[60 60 360*numel(valid) 380]);
tiledlayout(1, numel(valid), 'Padding','compact','TileSpacing','compact');
for ii = 1:numel(valid)
    i = valid(ii);
    nexttile;
    img = rot90(squeeze(chis{i}(:,:,sz)));
    m   = rot90(squeeze(Mask(:,:,sz)));
    h = imagesc(img, clim); set(h,'AlphaData',double(m));
    axis image off; set(gca,'Color','k'); colormap(gca, gray(256));
    title(sprintf('\\lambda = %.3g', lambdas(i)));
end
sgtitle(sprintf('WH-QSM lambda sweep (clim=[%.2g %.2g] ppm)', clim(1), clim(2)));
try
    exportgraphics(fig, fullfile(sweepDir,'lambda_sweep_compare.png'), 'Resolution', 200);
catch
    saveas(fig, fullfile(sweepDir,'lambda_sweep_compare.png'));
end
end

function v = getf(s,n,d), if isstruct(s)&&isfield(s,n)&&~isempty(s.(n)), v=s.(n); else, v=d; end, end
function v = getf2(s,n,d), if isstruct(s)&&isfield(s,n)&&~isempty(s.(n)), v=s.(n); else, v=d; end, end
