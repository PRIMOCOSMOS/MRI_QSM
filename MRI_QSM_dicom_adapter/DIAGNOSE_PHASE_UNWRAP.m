function DIAGNOSE_PHASE_UNWRAP(whichSubject)
% DIAGNOSE_PHASE_UNWRAP.m
% ============================================================================
% 一锤定音: 判断"苍白球深部偏低"是否由【缺少空间相位解缠】引起。
%
% 背景: 真实 DICOM loader 只做了回波维解缠(unwrap_echo_phase_local), 没有
% 空间解缠。高分辨率+长TE+背景场下, 单回波在空间上会有 2π wrap, 线性拟合
% 会把高场区(深部核团)的场折叠/低估 -> 总场深部偏平 -> chi 深部偏平。
%
% 本工具:
%   1) 检查每个回波相位的取值范围(是否接近/超过 ±π, 提示有 wrap);
%   2) 用 SEPIA/MEDI 的【空间解缠】(Laplacian/ROMEO)重新处理 phase_rad_4d,
%      再线性拟合得到"空间解缠后的总场", 与现有(仅回波解缠)总场对比;
%   3) 报告两者在深部核团(R2* 高值区)的场量级差异;
%   4) 导出两版总场 NIfTI 供肉眼对比。
% 若空间解缠后深部场明显增强 -> 证实根因, 应在 loader 加空间解缠。
%
% 用法(adapter 目录): DIAGNOSE_PHASE_UNWRAP('normal')
% ============================================================================

if nargin<1||isempty(whichSubject), whichSubject='all'; end
repoRoot=fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(repoRoot,'modules'),'-begin');
addpath(fullfile(repoRoot,'MRI_QSM_dicom_adapter'),'-begin');
addpath(fullfile(repoRoot,'Utils_self'),'-begin');
P=whqsm_local_paths();
if isfield(P,'sepiaRoot')&&exist(P.sepiaRoot,'dir')==7
    addpath(genpath(P.sepiaRoot),'-begin');
    try, sepia_addpath; catch, end
end
if isfield(P,'mediRoot')&&exist(P.mediRoot,'dir')==7
    addpath(genpath(P.mediRoot),'-begin');
end

outRoot=fullfile(P.dataRoot,'_qsm_comparison_results');
dirs=[dir(fullfile(outRoot,'normal_*'));dir(fullfile(outRoot,'elderly_*'))];
dirs=dirs([dirs.isdir]);

for d=1:numel(dirs)
    nm=dirs(d).name;
    if ~strcmpi(whichSubject,'all')&&~contains(lower(nm),lower(whichSubject)), continue; end
    subDir=fullfile(dirs(d).folder,nm);
    fprintf('\n========= PHASE UNWRAP DIAGNOSIS: %s =========\n',nm);
    S=load_c(subDir); if isempty(S), continue; end
    data=S.data; Mask=logical(data.Mask);
    gyro=42.57747892; B0=double(data.B0);

    if ~isfield(data,'phase_rad_4d')||isempty(data.phase_rad_4d)
        fprintf('  无 phase_rad_4d, 无法诊断空间解缠。请用保存了逐回波相位的版本重跑 loader。\n');
        continue;
    end
    ph4d=double(data.phase_rad_4d);
    TE=double(data.echo_times_ms(:).')/1000;
    nE=size(ph4d,4);

    % ---- 1) 每回波相位范围(是否有 wrap 迹象) ----
    fprintf('\n[1] 逐回波相位范围 (|相位| 接近/超过 π 提示空间 wrap)\n');
    for e=1:nE
        v=ph4d(:,:,:,e); v=v(Mask); v=v(isfinite(v));
        fprintf('    echo%d TE=%.1fms: min=%.2f max=%.2f (rad), |max|/pi=%.2f\n', ...
            e, TE(e)*1000, min(v), max(v), max(abs(v))/pi);
    end
    frac_wrap = mean(abs(ph4d(repmat(Mask,[1 1 1 nE])))>pi*0.9);
    fprintf('    脑内 |相位|>0.9π 的体素比例 = %.1f%% (高 -> 很可能有空间 wrap)\n', frac_wrap*100);

    % ---- 2) 空间解缠后重拟合 vs 仅回波解缠 ----
    fprintf('\n[2] 空间解缠重拟合 对比\n');
    field_echo = double(data.fieldmap_Hz); field_echo(~Mask)=0;  % 现有(仅回波解缠)
    field_sp = spatial_unwrap_refit(ph4d, TE, Mask, data.spatial_res);
    if isempty(field_sp)
        fprintf('    空间解缠工具不可用(需 SEPIA/MEDI 的 unwrapLaplacian)。\n');
    else
        field_sp(~Mask)=0;
        rep('    仅回波解缠总场(Hz)', field_echo, Mask);
        rep('    空间解缠后总场(Hz)', field_sp, Mask);
        % 深部核团区(R2* 高值)对比
        if isfield(data,'R2star_Hz')
            gp = deepcore(double(data.R2star_Hz), Mask);
            fe=field_echo(gp); fs=field_sp(gp);
            fprintf('    [苍白球区] 仅回波解缠 std=%.2fHz | 空间解缠后 std=%.2fHz\n', std(fe),std(fs));
            fprintf('    比值 = %.2f  (>>1 表示空间解缠救回了深部场 = 证实根因)\n', std(fs)/max(std(fe),eps));
        end
        export_nii(subDir,'diag_field_echoUnwrap_ppm', field_echo/(gyro*B0*1e6), Mask);
        export_nii(subDir,'diag_field_spatialUnwrap_ppm', field_sp/(gyro*B0*1e6), Mask);
        fprintf('    已导出两版总场 NIfTI 供对比(注意: 此处用正确 gyro*1e6 转 ppm)。\n');
    end
    fprintf('\n判读: 若 [1] wrap 比例高 且 [2] 空间解缠后深部场明显增强,\n');
    fprintf('      则根因确认 = loader 缺空间解缠, 应在相位拟合前加 Laplacian/ROMEO。\n');
end
fprintf('\n==================================================\n');
end

%% ----
function field_Hz = spatial_unwrap_refit(ph4d, TE, Mask, vox)
% 对每个回波做空间解缠(Laplacian), 再线性拟合 slope -> Hz
field_Hz=[]; N=size(Mask); nE=size(ph4d,4);
have=exist('unwrapLaplacian','file')==2;
if ~have, return; end
ph_uw=zeros(size(ph4d));
for e=1:nE
    try
        ph_uw(:,:,:,e)=unwrapLaplacian(ph4d(:,:,:,e), N, double(vox(:).'));
    catch
        return;
    end
end
% 线性拟合 phase = 2*pi*f*TE
t=reshape(TE,[1 1 1 nE]); t0=mean(TE); tc=t-t0; denom=sum((TE-t0).^2);
pm=mean(ph_uw,4);
slope=sum((ph_uw-pm).*tc,4)./max(denom,eps);  % rad/s
field_Hz=slope/(2*pi);
end

function g=deepcore(vol,Mask)
N=size(Mask); idx=find(Mask);[x,y,z]=ind2sub(N,idx);
cx=mean(x);cy=mean(y);cz=mean(z);rx=(max(x)-min(x))/2;ry=(max(y)-min(y))/2;rz=(max(z)-min(z))/2;
core=false(N);
core(max(1,round(cx-rx*0.5)):min(N(1),round(cx+rx*0.5)), ...
     max(1,round(cy-ry*0.5)):min(N(2),round(cy+ry*0.5)), ...
     max(1,round(cz-rz*0.4)):min(N(3),round(cz+rz*0.4)))=true;
m=Mask&core; v=vol(m); thr=prctile(v(isfinite(v)),90);
g=m&(vol>=thr);
end

function rep(name,vol,Mask)
v=double(vol(logical(Mask)));v=v(isfinite(v));
if isempty(v),fprintf('%s: 空\n',name);return;end
fprintf('%s: median=%.4g std=%.4g p1=%.4g p99=%.4g\n',name,median(v),std(v),prctile(v,1),prctile(v,99));
end

function export_nii(subDir,name,vol,Mask)
try, v=double(vol);v(~Mask)=0; niftiwrite(single(v),fullfile(subDir,[name '.nii'])); catch, end
end

function S=load_c(subDir)
S=[];[~,nm]=fileparts(subDir);
if startsWith(lower(nm),'elderly_'),lb='elderly';elseif startsWith(lower(nm),'normal_'),lb='normal';else,lb='subject';end
f=fullfile(subDir,['whqsm_' lb '_complete.mat']);
if exist(f,'file')~=2
    dd=dir(fullfile(subDir,'whqsm_*_complete.mat'));
    if isempty(dd),warning('无 complete.mat');return;end
    f=fullfile(dd(1).folder,dd(1).name);
end
try,S=load(f);catch ME,warning('%s',ME.message);S=[];end
if ~isempty(S)&&~isfield(S,'data'),S=[];end
end
