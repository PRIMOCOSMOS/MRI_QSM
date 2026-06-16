function DIAGNOSE_GP_LOCATION(whichSubject)
% DIAGNOSE_GP_LOCATION.m
% ============================================================================
% 终极判别: 我一路判定"苍白球偏低", 但用的是【自动 ROI(R2*高值 or 中心区)】,
% 可能根本没框到真苍白球。本脚本直接检验:
%   1) TKD chi 的【全脑高值区】在哪? 它的数值是多少?
%   2) chi 高值区 与 R2* 高值区 是否【空间重合】?
%   3) 导出 3 张图(chi / R2* / 重合) 的【关键层面截图 PNG】供肉眼判断。
%
% 逻辑:
%   - 若 chi 全脑 p99~0.15(正常) 且 chi 高值区与 R2* 高值区重合于深部
%     -> 苍白球 chi 其实【正常】, 是之前 ROI 定位错导致误判"偏低"!
%   - 若 chi 高值区只在血管/边缘, 深部确实空 -> 真有问题。
%
% 用法(adapter 目录): DIAGNOSE_GP_LOCATION('normal')
% ============================================================================

if nargin<1||isempty(whichSubject), whichSubject='all'; end
repoRoot=fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(repoRoot,'modules'),'-begin');
addpath(fullfile(repoRoot,'MRI_QSM_dicom_adapter'),'-begin');
P=whqsm_local_paths();
outRoot=fullfile(P.dataRoot,'_qsm_comparison_results');
dirs=[dir(fullfile(outRoot,'normal_*'));dir(fullfile(outRoot,'elderly_*'))];
dirs=dirs([dirs.isdir]);

for d=1:numel(dirs)
    nm=dirs(d).name;
    if ~strcmpi(whichSubject,'all')&&~contains(lower(nm),lower(whichSubject)), continue; end
    subDir=fullfile(dirs(d).folder,nm);
    fprintf('\n========= GP LOCATION: %s =========\n',nm);

    % 用已导出的 TKD chi (thr=0.1)
    tkdFile=fullfile(subDir,'diag_TKD_thr10.nii');
    if exist(tkdFile,'file')~=2
        fprintf('  未找到 %s, 请先运行 DIAGNOSE_TKD_SANITY。\n', tkdFile); continue;
    end
    chi=double(niftiread(tkdFile));
    S=load_c(subDir); if isempty(S), continue; end
    data=S.data; Mask=logical(data.Mask);
    R2=[]; if isfield(data,'R2star_Hz'), R2=double(data.R2star_Hz); R2(~Mask)=0; end

    chi(~Mask)=0;
    fprintf('[chi 全脑] p95=%.4f p99=%.4f max=%.4f (文献GP 0.10-0.18)\n', ...
        prctile(chi(Mask),95), prctile(chi(Mask),99), max(chi(Mask)));

    % chi 高值区 (top 1%) 的质心 vs R2* 高值区质心
    chiHi = chi > prctile(chi(Mask),99);
    cc=centroid(chiHi);
    fprintf('[chi 高值区(top1%%)] 质心 voxel=[%.0f %.0f %.0f], 体素数=%d, 均值=%.4f\n', ...
        cc(1),cc(2),cc(3), nnz(chiHi), mean(chi(chiHi)));
    if ~isempty(R2)
        r2Hi = (R2 > prctile(R2(Mask),99)) & Mask;
        rc=centroid(r2Hi);
        fprintf('[R2* 高值区(top1%%)] 质心 voxel=[%.0f %.0f %.0f]\n', rc(1),rc(2),rc(3));
        overlap = nnz(chiHi & r2Hi)/max(nnz(r2Hi),1);
        fprintf('[重合] chi高值∩R2*高值 / R2*高值 = %.1f%%\n', overlap*100);
        % chi 在 R2* 高值区的值
        v=chi(r2Hi);
        fprintf('[R2*高值区的 chi] mean=%.4f median=%.4f p90=%.4f\n', mean(v),median(v),prctile(v,90));
    end

    % 中心深部区(去掉外围35%)的 chi 高值
    deep=center_region(Mask,0.35);
    vd=chi(deep&Mask);
    fprintf('[深部中央区 chi] p95=%.4f p99=%.4f max=%.4f\n', prctile(vd,95),prctile(vd,99),max(vd));

    % 截图: chi / R2* 在 chi 高值质心所在层
    z=max(1,min(size(Mask,3), round(cc(3))));
    save_png(subDir, nm, chi, R2, Mask, z);
    fprintf('已存截图(chi高值层 z=%d): diag_gp_location.png\n', z);

    fprintf('\n判读:\n');
    fprintf('  若 [深部中央区 chi p99] 达 0.10-0.18 -> 苍白球 chi 正常, 之前是ROI误判!\n');
    fprintf('  若 [重合]%% 高 且 [R2*高值区chi] 达 0.1 -> 苍白球正常显现!\n');
    fprintf('  若深部全空、chi高值只在边缘 -> 深部确实缺失。\n');
end
end

%% ----
function c=centroid(bw)
[x,y,z]=ind2sub(size(bw),find(bw));
if isempty(x), c=[NaN NaN NaN]; else, c=[mean(x) mean(y) mean(z)]; end
end

function reg=center_region(Mask,edge)
N=size(Mask); idx=find(Mask);[x,y,z]=ind2sub(N,idx);
cx=mean(x);cy=mean(y);cz=mean(z);rx=(max(x)-min(x))/2;ry=(max(y)-min(y))/2;rz=(max(z)-min(z))/2;
reg=false(N);
reg(max(1,round(cx-rx*(1-edge))):min(N(1),round(cx+rx*(1-edge))), ...
    max(1,round(cy-ry*(1-edge))):min(N(2),round(cy+ry*(1-edge))), ...
    max(1,round(cz-rz*(1-edge))):min(N(3),round(cz+rz*(1-edge))))=true;
end

function save_png(subDir,nm,chi,R2,Mask,z)
try
    fig=figure('Color','w','Position',[40 40 1400 480],'Visible','off');
    tiledlayout(1,3,'Padding','compact','TileSpacing','compact');
    nexttile; show(chi(:,:,z),Mask(:,:,z),[-0.1 0.15]); title('TKD \chi (z=GP层)'); colorbar;
    if ~isempty(R2), nexttile; show(R2(:,:,z),Mask(:,:,z),[0 60]); title('R2*'); colorbar; end
    nexttile; show(chi(:,:,z),Mask(:,:,z),[0.08 0.18]); title('\chi 窗[0.08 0.18] 看铁核'); colorbar;
    sgtitle(sprintf('%s GP location check (z=%d)',nm,z),'Interpreter','none');
    exportgraphics(fig,fullfile(subDir,'diag_gp_location.png'),'Resolution',150);
    close(fig);
catch
end
end

function show(img,mask,cl)
img=rot90(squeeze(img)); mask=rot90(squeeze(mask));
h=imagesc(img,cl); set(h,'AlphaData',double(mask)); axis image off; set(gca,'Color','k');
colormap(gca,turbo(256));
end

function S=load_c(subDir)
S=[];[~,nm]=fileparts(subDir);
if startsWith(lower(nm),'elderly_'),lb='elderly';elseif startsWith(lower(nm),'normal_'),lb='normal';else,lb='subject';end
f=fullfile(subDir,['whqsm_' lb '_complete.mat']);
if exist(f,'file')~=2
    dd=dir(fullfile(subDir,'whqsm_*_complete.mat'));
    if isempty(dd),warning('无complete.mat');return;end
    f=fullfile(dd(1).folder,dd(1).name);
end
try,S=load(f);catch ME,warning('%s',ME.message);S=[];end
if ~isempty(S)&&~isfield(S,'data'),S=[];end
end
