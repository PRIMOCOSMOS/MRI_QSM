function RUN_QSM_ROI_CHECK(whichSubject)
% RUN_QSM_ROI_CHECK.m
% ============================================================================
% 客观测量深部核团(苍白球/壳核)的 χ 数值，判断 WH-QSM 是否真的偏低。
%
% 背景: 全脑 p99 会被血管/边缘伪影撑高，不能代表苍白球。本工具用"深部高磁化率
% 区域"自动提取(不依赖 atlas)，给出真实 ROI 均值，并与文献参考对照。
%
% 方法(无 atlas 也能用):
%   1) 取脑中心区域(排除外围 25%，避开皮层/血管/边缘伪影);
%   2) 在中心区内取 χ 高分位(默认 top 2%)作为"铁核候选";
%   3) 报告该区 χ 的 mean/median/p95，并与文献 GP(0.10-0.18ppm) 对照。
% 若提供 atlas 标签(P.roiLabelFile)，则改用真实 GP/Put 标签(更准)。
%
% 用法(项目根目录):
%   RUN_QSM_ROI_CHECK              % 所有被试
%   RUN_QSM_ROI_CHECK('normal')
% ============================================================================

if nargin < 1 || isempty(whichSubject), whichSubject = 'all'; end
repoRoot = fileparts(mfilename('fullpath'));
addpath(fullfile(repoRoot,'modules'),'-begin');
addpath(fullfile(repoRoot,'MRI_QSM_dicom_adapter'),'-begin');
addpath(fullfile(repoRoot,'Utils_self'),'-begin');

P = whqsm_local_paths();
outRoot = fullfile(P.dataRoot, '_qsm_comparison_results');
subDirs = resolve_dirs(outRoot, whichSubject);
if isempty(subDirs), error('未找到被试结果目录: %s', outRoot); end

roiFile = '';
if isfield(P,'roiLabelFile'), roiFile = char(P.roiLabelFile); end

fprintf('\n================ 深部核团 χ 数值核查 ================\n');
fprintf('文献参考(3T 健康成人, ppm):\n');
fprintf('  苍白球 GP ~0.10-0.18 | 壳核 Put ~0.02-0.06 | 红核 ~0.08-0.14\n');
fprintf('----------------------------------------------------\n');

for s = 1:numel(subDirs)
    subDir = subDirs{s};
    [chi, Mask, label] = load_chi(subDir);
    if isempty(chi), continue; end
    fprintf('\n[%s]  %s\n', upper(label), subDir);

    wb = chi(Mask); wb=wb(isfinite(wb));
    fprintf('  全脑 χ 分布 : median=%.4f  p90=%.4f  p95=%.4f  p99=%.4f\n', ...
        median(wb), prctile(wb,90), prctile(wb,95), prctile(wb,99));
    fprintf('            (p99 易被血管/边缘伪影抬高，不代表苍白球)\n');

    if ~isempty(roiFile) && exist(roiFile,'file')==2
        % --- 可靠路径: 用 atlas 标签精确测 GP/Put ---
        report_atlas_roi(roiFile, chi, Mask);
    else
        % --- 无 atlas: 只给手动 ROI 指引 + 直方图,不做不可靠的自动圈核 ---
        fprintf(['  [未提供 atlas] 无法可靠自动分离苍白球(易与血管混淆)。\n' ...
                 '   两种可靠测量方式(任选其一):\n' ...
                 '   1) 在 MATLAB 里手动量: 加载该被试 chi，在苍白球处画 ROI 取 mean。\n' ...
                 '      已为你导出 chi 到: %s\n' ...
                 '   2) 配准 SNU chi-separation atlas 标签到本被试，设 P.roiLabelFile。\n'], ...
                 export_chi_nii(subDir, chi, Mask, label));
        % 直方图(全脑)便于看整体动态范围是否被压缩
        save_hist(chi, Mask, subDir, label);
    end
end
fprintf('\n====================================================\n');
end

%% ----------------------------------------------------------------------
function report_atlas_roi(roiFile, chi, Mask)
L = load(roiFile); fn=fieldnames(L); labels=[];
for j=1:numel(fn)
    v=L.(fn{j});
    if isnumeric(v) && isequal(size(v),size(Mask)) && max(v(:))>1, labels=round(v); break; end
end
if isempty(labels)
    fprintf('  [atlas] 文件中未找到整型标签体，跳过。\n'); return;
end
ids=unique(labels(labels>0));
fprintf('  [atlas ROI]  id    mean(ppm)   median   nvox\n');
for k=1:numel(ids)
    m=(labels==ids(k))&Mask; if nnz(m)<10, continue; end
    fprintf('               %-4d  %-9.4f  %-7.4f  %d\n', ids(k), mean(chi(m)), median(chi(m)), nnz(m));
end
fprintf('  对照文献: GP 0.10-0.18 | Put 0.02-0.06 | RN 0.08-0.14 ppm\n');
end

function p = export_chi_nii(subDir, chi, Mask, label)
p = fullfile(subDir, sprintf('chi_for_roi_%s.nii', label));
try
    chi2=chi; chi2(~Mask)=0;
    niftiwrite(single(chi2), p);
catch ME
    p = ['(导出失败: ' ME.message ')'];
end
end

function save_hist(chi, Mask, subDir, label)
try
    v=chi(Mask); v=v(isfinite(v) & abs(v)<0.3);
    fig=figure('Color','w','Visible','off');
    histogram(v, 200); xlabel('\chi (ppm)'); ylabel('voxels');
    title(sprintf('%s WH-QSM \\chi histogram (in-mask)', label));
    xline(0.10,'r--','GP~0.10'); xline(0.18,'r--','0.18');
    f=fullfile(subDir, sprintf('chi_hist_%s.png', label));
    try, exportgraphics(fig,f,'Resolution',150); catch, saveas(fig,f); end
    close(fig);
    fprintf('  全脑 χ 直方图已存: %s\n', f);
catch
end
end

function [chi, Mask, label] = load_chi(subDir)
chi=[]; Mask=[]; label='subject';
[~,nm]=fileparts(subDir);
if startsWith(lower(nm),'elderly_'),label='elderly';elseif startsWith(lower(nm),'normal_'),label='normal';end
f=fullfile(subDir,['whqsm_' label '_complete.mat']);
if exist(f,'file')~=2
    d=dir(fullfile(subDir,'whqsm_*_complete.mat'));
    if isempty(d), d=dir(fullfile(subDir,'chi_*.mat')); end
    if isempty(d), warning('无结果文件: %s',subDir); return; end
    f=fullfile(d(1).folder,d(1).name);
end
try, S=load(f); catch ME, warning('加载失败 %s: %s',f,ME.message); return; end
if isfield(S,'chi'), chi=double(S.chi); end
if isfield(S,'data')&&isfield(S.data,'Mask'), Mask=logical(S.data.Mask);
elseif isfield(S,'mask'), Mask=logical(S.mask); end
if isempty(chi)||isempty(Mask), warning('缺 chi/Mask: %s',f); chi=[]; end
end

function d = resolve_dirs(outRoot, which)
which=char(which);
if exist(which,'dir')==7, d={which}; return; end
a=[dir(fullfile(outRoot,'normal_*'));dir(fullfile(outRoot,'elderly_*'))];a=a([a.isdir]);d={};
for i=1:numel(a)
    if strcmpi(which,'all')||contains(lower(a(i).name),lower(which))
        d{end+1}=fullfile(a(i).folder,a(i).name); %#ok<AGROW>
    end
end
end
