function DIAGNOSE_FIELD_CHAIN(whichSubject)
% DIAGNOSE_FIELD_CHAIN.m
% ============================================================================
% 深度诊断"为什么局部场偏平/苍白球偏低": 把场图链路的每一步关键量化指标都打印,
% 并导出 NIfTI 供你直接在 ITK-SNAP/FSLeyes 里逐张核对。
%
% 它不重算 WH-QSM, 只加载已有结果 + 复跑场图制备, 逐步报告:
%   1) 相位转换方法/单位 (是否正确缩放到 radian)
%   2) 总场 total field 的量级与空间结构 (是否本身就异常)
%   3) R2* 是否正常 (交叉验证数据质量: 苍白球应是 R2* 高值区)
%   4) 局部场 local field: 在苍白球粗ROI处的均值/结构
%   5) 导出 total_field/local_field/R2star 三个 NIfTI 供肉眼核对
%
% 用法(adapter 目录):
%   DIAGNOSE_FIELD_CHAIN            % 第一个被试
%   DIAGNOSE_FIELD_CHAIN('normal')
% ============================================================================

if nargin < 1 || isempty(whichSubject), whichSubject = 'all'; end
repoRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(repoRoot,'modules'),'-begin');
addpath(fullfile(repoRoot,'MRI_QSM_dicom_adapter'),'-begin');
addpath(fullfile(repoRoot,'Utils_self'),'-begin');

P = whqsm_local_paths();
if isfield(P,'mediRoot') && exist(P.mediRoot,'dir')==7
    addpath(genpath(P.mediRoot),'-begin');
end
if isfield(P,'sepiaRoot') && exist(P.sepiaRoot,'dir')==7
    addpath(genpath(P.sepiaRoot),'-begin');
end

outRoot = fullfile(P.dataRoot, '_qsm_comparison_results');
dirs = [dir(fullfile(outRoot,'normal_*')); dir(fullfile(outRoot,'elderly_*'))];
dirs = dirs([dirs.isdir]);

for d = 1:numel(dirs)
    nm = dirs(d).name;
    if ~strcmpi(whichSubject,'all') && ~contains(lower(nm),lower(whichSubject)), continue; end
    subDir = fullfile(dirs(d).folder, nm);
    fprintf('\n================ FIELD CHAIN DIAGNOSIS: %s ================\n', nm);

    S = load_complete(subDir);
    if isempty(S), continue; end
    data = S.data; Mask = logical(data.Mask);
    gyro = 42.57747892; B0 = double(data.B0);

    % ---- 1) 相位/场基本信息 ----
    fprintf('\n[1] 采集与场基本信息\n');
    fprintf('    B0 = %.3g T, CF = %.4g MHz\n', B0, gyro*B0);
    if isfield(data,'echo_times_ms'), fprintf('    TE = %s ms\n', mat2str(data.echo_times_ms,5)); end
    if isfield(data,'delta_TE'), fprintf('    delta_TE = %.4g ms\n', data.delta_TE*1000); end
    if isfield(data,'spatial_res'), fprintf('    voxel = [%.4g %.4g %.4g] mm\n', data.spatial_res); end
    if isfield(data,'phase_fit_method'), fprintf('    phase fit = %s\n', data.phase_fit_method); end

    % ---- 2) 总场 ----
    if isfield(data,'fieldmap_Hz')
        tf = double(data.fieldmap_Hz); tf(~Mask)=0;
        fprintf('\n[2] 总场 total field\n');
        rep('    total field (Hz)', tf, Mask);
        rep('    total field (ppm)', tf/(gyro*B0), Mask);
    end

    % ---- 3) R2* 交叉验证(苍白球应是 R2* 高值) ----
    if isfield(data,'R2star_Hz') && ~isempty(data.R2star_Hz)
        r2 = double(data.R2star_Hz); r2(~Mask)=0;
        fprintf('\n[3] R2* (交叉验证数据质量; 苍白球应是高 R2* 区)\n');
        rep('    R2* (1/s)', r2, Mask);
        gp = deep_core(r2, Mask);  % 深部高 R2* 区作为苍白球粗ROI
        fprintf('    深部高 R2* 区均值 = %.1f /s (苍白球典型 30-60/s @3T)\n', mean(gp.vals));
        prep_gpmask = gp.mask;     % 复用此 ROI 去 local field 取值
    else
        fprintf('\n[3] 无 R2*，跳过交叉验证\n'); prep_gpmask=[];
    end

    % ---- 4) 复跑场图制备, 看局部场在苍白球ROI的结构 ----
    fprintf('\n[4] 复跑 BFR -> 局部场\n');
    cfg = struct(); cfg.resultDir = fullfile(subDir,'results');
    cfg.whqsm.do_bfr=true; cfg.whqsm.bfr_method='LBV';
    cfg.whqsm.bfr_tol=0.005; cfg.whqsm.bfr_peel=2; cfg.whqsm.do_spatial_unwrap=false;
    try
        [lf_Hz, prep] = mod_field_preprocess(data, Mask, cfg);
        lfMask = logical(prep.mask_after_bfr);
        rep('    local field (Hz)', lf_Hz, lfMask);
        rep('    local field (ppm)', lf_Hz/(gyro*B0), lfMask);
        if ~isempty(prep_gpmask)
            m = prep_gpmask & lfMask;
            if nnz(m)>10
                fprintf('    苍白球粗ROI 局部场: mean=%.4g ppm, std=%.4g ppm (应有非零偶极结构)\n', ...
                    mean(lf_Hz(m))/(gyro*B0), std(lf_Hz(m))/(gyro*B0));
            end
        end
        % ---- 5) 导出 NIfTI 供肉眼核对 ----
        export_nii(subDir, 'diag_total_field_ppm', double(data.fieldmap_Hz)/(gyro*B0), Mask);
        export_nii(subDir, 'diag_local_field_ppm', lf_Hz/(gyro*B0), lfMask);
        if isfield(data,'R2star_Hz'), export_nii(subDir, 'diag_R2star', double(data.R2star_Hz), Mask); end
        fprintf('\n[5] 已导出 diag_*.nii 到: %s\n', subDir);
        fprintf('    请在 ITK-SNAP/FSLeyes 打开 diag_local_field_ppm.nii,\n');
        fprintf('    窗位设 [-0.03 0.03] ppm, 看苍白球处是否有正负偶极场结构。\n');
    catch ME
        fprintf('    BFR 复跑失败: %s\n', ME.message);
    end
end
fprintf('\n=========================================================\n');
end

%% ---- helpers ----
function rep(name, vol, Mask)
v=double(vol(logical(Mask))); v=v(isfinite(v));
if isempty(v), fprintf('%s: 空\n',name); return; end
fprintf('%s: median=%.4g std=%.4g p1=%.4g p99=%.4g\n', name, median(v), std(v), prctile(v,1), prctile(v,99));
end

function g = deep_core(vol, Mask)
% 取脑中心区(去外围40%)内的高值作为深部核团粗ROI(用于 R2*: 苍白球 R2* 高)
N=size(Mask); idx=find(Mask); [x,y,z]=ind2sub(N,idx);
cx=mean(x);cy=mean(y);cz=mean(z); rx=(max(x)-min(x))/2;ry=(max(y)-min(y))/2;rz=(max(z)-min(z))/2;
core=false(N);
xr=max(1,round(cx-rx*0.5)):min(N(1),round(cx+rx*0.5));
yr=max(1,round(cy-ry*0.5)):min(N(2),round(cy+ry*0.5));
zr=max(1,round(cz-rz*0.4)):min(N(3),round(cz+rz*0.4));
core(xr,yr,zr)=true; m=Mask&core;
v=vol(m); thr=prctile(v(isfinite(v)),90);
gmask=m & (vol>=thr);
g.mask=gmask; g.vals=vol(gmask);
end

function export_nii(subDir, name, vol, Mask)
try
    v=double(vol); v(~Mask)=0;
    niftiwrite(single(v), fullfile(subDir, [name '.nii']));
catch
end
end

function S = load_complete(subDir)
S=[]; [~,nm]=fileparts(subDir);
if startsWith(lower(nm),'elderly_'),label='elderly';elseif startsWith(lower(nm),'normal_'),label='normal';else,label='subject';end
f=fullfile(subDir,['whqsm_' label '_complete.mat']);
if exist(f,'file')~=2
    dd=dir(fullfile(subDir,'whqsm_*_complete.mat'));
    if isempty(dd), warning('无 complete.mat: %s',subDir); return; end
    f=fullfile(dd(1).folder,dd(1).name);
end
try, S=load(f); catch ME, warning('加载失败: %s',ME.message); S=[]; end
if ~isempty(S) && ~isfield(S,'data'), warning('无 data 字段'); S=[]; end
end
