function DIAGNOSE_TKD_SANITY(whichSubject)
% DIAGNOSE_TKD_SANITY.m
% ============================================================================
% 终极区分测试: 用最简单的 TKD 反演(无正则、无参数、无SEPIA)直接对【你的局部场】
% 做偶极反演, 看苍白球能否出来。
%
% 逻辑:
%   - 局部场已被审计为【正常】(量级对、单位对、偶极场内部mean≈0是正常物理)。
%   - 若 TKD(纯数学反演)能出苍白球 -> 数据/场都好, 问题在 WH-QSM/FANSI 的参数或平滑。
%   - 若 TKD 也出不来苍白球 -> 场虽量级对但【空间结构】有问题(深部偶极结构缺失)。
%   这是不依赖任何工具箱、不依赖参数的"地面真值"测试。
%
% 同时打印 B0 来源、局部场在苍白球(R2*高值区)的偶极结构指标。
%
% 用法(adapter 目录): DIAGNOSE_TKD_SANITY('normal')
% ============================================================================

if nargin<1||isempty(whichSubject), whichSubject='all'; end
repoRoot=fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(repoRoot,'modules'),'-begin');
addpath(fullfile(repoRoot,'MRI_QSM_dicom_adapter'),'-begin');
addpath(fullfile(repoRoot,'Utils_self'),'-begin');
P=whqsm_local_paths();

outRoot=fullfile(P.dataRoot,'_qsm_comparison_results');
dirs=[dir(fullfile(outRoot,'normal_*'));dir(fullfile(outRoot,'elderly_*'))];
dirs=dirs([dirs.isdir]);

for d=1:numel(dirs)
    nm=dirs(d).name;
    if ~strcmpi(whichSubject,'all')&&~contains(lower(nm),lower(whichSubject)), continue; end
    subDir=fullfile(dirs(d).folder,nm);
    fprintf('\n========= TKD SANITY: %s =========\n',nm);
    S=load_c(subDir); if isempty(S), continue; end
    data=S.data; Mask=logical(data.Mask);
    gyro=42.57747892; B0=double(data.B0);
    vox=double(data.spatial_res(:).');
    B0dir=[0 0 1];
    if isfield(data,'B0_dir')&&~isempty(data.B0_dir)
        B0dir=double(data.B0_dir(:).'); B0dir=B0dir/max(norm(B0dir),eps);
    end

    % B0 来源
    fprintf('[B0] = %.4g T, B0_dir = [%.3g %.3g %.3g]\n', B0, B0dir);
    if isfield(data,'phase_fit_method'), fprintf('[phase] %s\n', data.phase_fit_method); end

    % 局部场(ppm): 优先 BFR 后的; 否则用 fieldmap_ppm
    lf_ppm = get_local_ppm(data, Mask, gyro, B0, P);
    rep('[局部场 ppm]', lf_ppm, Mask);

    % 偶极核(考虑各向异性体素 + B0方向)
    N=size(Mask);
    D = dipole_kernel(N, vox, B0dir);
    fprintf('[偶极核] min=%.3f max=%.3f (应 -2/3..1/3)\n', min(D(:)), max(D(:)));

    % --- TKD 反演 (多个阈值) ---
    for thr=[0.1 0.2]
        chi = tkd(lf_ppm, D, Mask, thr);
        rep(sprintf('[TKD chi thr=%.2f]',thr), chi, Mask);
        % 苍白球区(R2*高值)
        if isfield(data,'R2star_Hz')
            gp=deepcore(double(data.R2star_Hz),Mask);
            v=chi(gp); v=v(isfinite(v));
            fprintf('   苍白球区 chi: mean=%.4f median=%.4f (文献GP 0.10-0.18ppm)\n', mean(v), median(v));
        end
        export_nii(subDir, sprintf('diag_TKD_thr%02d',round(thr*100)), chi, Mask);
    end
    fprintf('\n判读: TKD 苍白球 mean 接近 0.1-0.18 -> 数据+场都好, 问题在 FANSI;\n');
    fprintf('      TKD 苍白球也接近 0 -> 场的深部偶极结构本身缺失(更上游)。\n');
    fprintf('      已导出 diag_TKD_*.nii 供肉眼看苍白球是否显现。\n');
end
end

%% ----
function lf = get_local_ppm(data, Mask, gyro, B0, P)
% 优先 BFR 后局部场; 否则 fieldmap_ppm
lf=[];
try
    if exist('mod_field_preprocess','file')==2
        cfg=struct(); cfg.resultDir=tempname;
        cfg.whqsm.do_bfr=true; cfg.whqsm.bfr_method='LBV';
        cfg.whqsm.bfr_tol=0.005; cfg.whqsm.bfr_peel=2; cfg.whqsm.do_spatial_unwrap=false;
        [lfHz,~]=mod_field_preprocess(data,Mask,cfg);
        lf=lfHz/(gyro*B0);  % 与现有 fieldmap_ppm 同定义(gyro MHz)
    end
catch
end
if isempty(lf)
    if isfield(data,'local_field_ppm'), lf=double(data.local_field_ppm);
    elseif isfield(data,'phs_tissue'), lf=double(data.phs_tissue);
    else, lf=double(data.fieldmap_Hz)/(gyro*B0); end
end
lf(~Mask)=0;
end

function D=dipole_kernel(N,vox,B0dir)
if exist('create_dipole_kernel','file')==2
    D=create_dipole_kernel(N,vox,B0dir); return;
end
kx=ifftshift((-floor(N(1)/2):ceil(N(1)/2)-1)/(N(1)*vox(1)));
ky=ifftshift((-floor(N(2)/2):ceil(N(2)/2)-1)/(N(2)*vox(2)));
kz=ifftshift((-floor(N(3)/2):ceil(N(3)/2)-1)/(N(3)*vox(3)));
[KX,KY,KZ]=ndgrid(kx,ky,kz); k2=KX.^2+KY.^2+KZ.^2;
kd=KX*B0dir(1)+KY*B0dir(2)+KZ*B0dir(3);
D=zeros(N); idx=k2>0; D(idx)=1/3-(kd(idx).^2./k2(idx));
end

function chi=tkd(field,D,Mask,thr)
Di=D; s=sign(Di); s(s==0)=1;
Di(abs(D)<thr)=thr*s(abs(D)<thr);
chi=real(ifftn(fftn(field)./Di)).*Mask;
end

function g=deepcore(vol,Mask)
N=size(Mask);idx=find(Mask);[x,y,z]=ind2sub(N,idx);
cx=mean(x);cy=mean(y);cz=mean(z);rx=(max(x)-min(x))/2;ry=(max(y)-min(y))/2;rz=(max(z)-min(z))/2;
core=false(N);
core(max(1,round(cx-rx*0.5)):min(N(1),round(cx+rx*0.5)), ...
     max(1,round(cy-ry*0.5)):min(N(2),round(cy+ry*0.5)), ...
     max(1,round(cz-rz*0.4)):min(N(3),round(cz+rz*0.4)))=true;
m=Mask&core; v=vol(m); thr=prctile(v(isfinite(v)),90); g=m&(vol>=thr);
end

function rep(name,vol,Mask)
v=double(vol(logical(Mask)));v=v(isfinite(v));
if isempty(v),fprintf('%s: 空\n',name);return;end
fprintf('%s median=%.4g std=%.4g p1=%.4g p99=%.4g\n',name,median(v),std(v),prctile(v,1),prctile(v,99));
end

function export_nii(subDir,name,vol,Mask)
try,v=double(vol);v(~Mask)=0;niftiwrite(single(v),fullfile(subDir,[name '.nii']));catch,end
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
