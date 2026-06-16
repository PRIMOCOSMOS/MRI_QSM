function [localField_Hz, prep] = mod_field_preprocess(data, Mask, cfg)
% mod_field_preprocess.m
% ============================================================================
% 独立的"场图制备"阶段: 把 DICOM 相位拟合得到的【总场(total field)】转换成
% 偶极反演所需的【局部场/组织场(local/tissue field)】。
%
% 这是 QSM 共识(ISMRM 2024)规定的反演前必需步骤, 包含:
%   (1) 空间相位解缠 (Laplacian, 经 SEPIA/MEDI; 现有 loader 仅做了回波维解缠)
%   (2) 背景场去除 BFR: LBV 优先(文献: 配 WH-QSM 最佳), 回退 V-SHARP
%
% 重要: 本阶段【不触碰 WH-QSM 反演】。WH-QSM 的弱谐波项只处理【残余】背景场,
%       不能替代完整 BFR(文献明确)。详见 docs/FIELD_PREPROCESS_NOTES.md。
%
% 参考:
%   - QSM Consensus (ISMRM EMTP Study Group), 2024: 解缠+回波合并 → mask →
%     SHARP/PDF 类背景场去除 → 稀疏正则偶极反演。
%   - SEPIA 教程 Exercise 3: 背景场去除后 globus pallidus/red nuclei/
%     substantia nigra 才清晰可见。
%   - Investigating masking & BFR (QSM Challenge phantom): WH-QSM 配 LBV 最佳。
%
% 输入:
%   data  : 含 fieldmap_Hz(总场,Hz) 或 phs_unwrap, spatial_res, B0, B0_dir,
%           (可选) phase_unwrapped_4d / TE 用于空间解缠
%   Mask  : 脑 mask
%   cfg   : cfg.whqsm.bfr_method ('LBV'|'VSHARP'|'auto'), .bfr_tol, .bfr_peel,
%           .bfr_vsharp_radius, .do_spatial_unwrap (true/false/'auto'),
%           .resultDir (QC 输出)
%
% 输出:
%   localField_Hz : 背景场去除后的局部场(Hz), 供 WH-QSM 反演
%   prep          : 结构, 记录所用方法/参数/QC 路径/前后统计
% ============================================================================

prep = struct();
Mask = logical(Mask);
N = size(Mask);
voxel_size = double(data.spatial_res(:).');
B0 = double(data.B0);
B0_dir = [0 0 1];
if isfield(data,'B0_dir') && ~isempty(data.B0_dir)
    B0_dir = double(data.B0_dir(:).'); B0_dir = B0_dir./max(norm(B0_dir),eps);
end
gyro = 42.57747892;     % MHz/T

bfrMethod = upper(char(get_cfg(cfg, {'whqsm','bfr_method'}, 'LBV')));
tol       = double(get_cfg(cfg, {'whqsm','bfr_tol'}, 0.005));
peel      = double(get_cfg(cfg, {'whqsm','bfr_peel'}, 2));
vsharpR   = get_cfg(cfg, {'whqsm','bfr_vsharp_radius'}, 1:1:12);
doUnwrap  = get_cfg(cfg, {'whqsm','do_spatial_unwrap'}, 'auto');

% -------------------------------------------------------------------------
% 0) 取总场(Hz)
% -------------------------------------------------------------------------
if isfield(data,'fieldmap_Hz') && ~isempty(data.fieldmap_Hz)
    totalField_Hz = double(data.fieldmap_Hz);
elseif isfield(data,'local_field_ppm') && ~isempty(data.local_field_ppm)
    totalField_Hz = double(data.local_field_ppm) * (gyro * B0);
else
    error('mod_field_preprocess: 找不到总场(fieldmap_Hz / local_field_ppm)。');
end
totalField_Hz(~isfinite(totalField_Hz)) = 0;
totalField_Hz(~Mask) = 0;

fprintf('\n========== 场图制备 (BFR, 反演前必需步骤) ==========\n');
print_field_stats('总场 total field (Hz)', totalField_Hz, Mask);

% -------------------------------------------------------------------------
% 1) 空间相位解缠 (可选, 默认关闭)
%    重要: loader 的 fieldmap_Hz 是【多回波线性拟合】得到的频率场(Hz), 它是一个
%    连续场, 不像单回波相位那样存在密集的 2*pi 空间 wrap。对这种已拟合的场再做
%    "Laplacian 一致化" 反而会破坏低频结构(实测: std 2.37 -> 0.28, 场被抹平)。
%    因此默认【不做】空间解缠; 仅当你确知输入是单回波 wrapped 相位时才开启。
% -------------------------------------------------------------------------
prep.spatial_unwrap = 'skipped';
needUnwrap = decide_unwrap(doUnwrap);   % 默认 false
if needUnwrap
    [unwrappedField_Hz, uw_method] = spatial_unwrap_field(data, totalField_Hz, Mask, voxel_size);
    if ~isempty(unwrappedField_Hz) && field_not_destroyed(totalField_Hz, unwrappedField_Hz, Mask)
        totalField_Hz = unwrappedField_Hz; totalField_Hz(~Mask)=0;
        prep.spatial_unwrap = uw_method;
        print_field_stats('空间解缠后总场 (Hz)', totalField_Hz, Mask);
    else
        warning('空间解缠被跳过(不可用或会破坏场)。直接用拟合频率场。');
    end
end

% -------------------------------------------------------------------------
% 2) 背景场去除 (LBV 优先 -> V-SHARP -> PDF). 复用库内成熟封装。
%    收集每个方法的真实错误并打印, 不再静默吞掉。
% -------------------------------------------------------------------------
order = bfr_order(bfrMethod);
localField_Hz = []; usedBFR = ''; bfrMask = Mask;
errLog = {};
for i = 1:numel(order)
    m = order{i};
    fprintf('  尝试背景场去除: %s ...\n', m);
    try
        switch m
            case 'LBV'
                if exist('bg_removal_lbv_medi','file')~=2
                    errLog{end+1} = 'LBV: 找不到 bg_removal_lbv_medi (MEDI/SEPIA 未加 path?)'; %#ok<AGROW>
                    continue;
                end
                [lf, nm] = bg_removal_lbv_medi(totalField_Hz, Mask, N, voxel_size, tol, peel);
                [ok,why] = check_field(lf, nm); 
                if ok, localField_Hz=lf; bfrMask=logical(nm); usedBFR='LBV'; break;
                else, errLog{end+1} = ['LBV 输出无效: ' why]; end %#ok<AGROW>
            case 'PDF'
                if exist('bg_removal_pdf_medi','file')~=2
                    errLog{end+1} = 'PDF: 找不到 bg_removal_pdf_medi'; continue; %#ok<AGROW>
                end
                lf = bg_removal_pdf_medi(totalField_Hz, Mask, N, voxel_size, B0_dir);
                [ok,why] = check_field(lf, Mask);
                if ok, localField_Hz=lf; usedBFR='PDF'; break;
                else, errLog{end+1} = ['PDF 输出无效: ' why]; end %#ok<AGROW>
            case 'VSHARP'
                if exist('bg_removal_vsharp','file')~=2
                    errLog{end+1} = 'VSHARP: 找不到 bg_removal_vsharp'; continue; %#ok<AGROW>
                end
                [lf, nm] = bg_removal_vsharp(totalField_Hz, Mask, voxel_size, vsharpR);
                [ok,why] = check_field(lf, nm);
                if ok, localField_Hz=lf; bfrMask=logical(nm); usedBFR='VSHARP'; break;
                else, errLog{end+1} = ['VSHARP 输出无效: ' why]; end %#ok<AGROW>
        end
    catch ME
        errLog{end+1} = sprintf('%s 抛异常: %s', m, ME.message); %#ok<AGROW>
        fprintf('    [%s] 异常: %s\n', m, ME.message);
        if ~isempty(ME.stack)
            fprintf('      at %s line %d\n', ME.stack(1).name, ME.stack(1).line);
        end
    end
end

if isempty(localField_Hz)
    msg = sprintf('所有背景场去除方法都失败。各方法原因:\n');
    for i=1:numel(errLog), msg = [msg sprintf('   - %s\n', errLog{i})]; end %#ok<AGROW>
    msg = [msg sprintf(['\n排查:\n' ...
        ' 1) which LBV / which PDF / which V_SHARP  看函数是否在 path;\n' ...
        ' 2) 确认 P.mediRoot / P.sepiaRoot 正确且已 addpath(genpath());\n' ...
        ' 3) 若输入已是局部场, 设 cfg.whqsm.do_bfr=false。\n'])];
    error('%s', msg);
end
localField_Hz(~bfrMask) = 0;
localField_Hz(~isfinite(localField_Hz)) = 0;

print_field_stats(sprintf('局部场 local field (Hz) [%s]', usedBFR), localField_Hz, bfrMask);

prep.bfr_method = usedBFR;
prep.bfr_tol = tol; prep.bfr_peel = peel;
prep.mask_after_bfr = bfrMask;
prep.totalField_Hz = totalField_Hz;

% -------------------------------------------------------------------------
% 3) QC: 总场 vs 局部场 三平面对比 (验证 BFR 是否生效: 局部场应露出深部核团)
% -------------------------------------------------------------------------
try
    qcDir = get_cfg(cfg, {'resultDir'}, pwd);
    if ~exist(qcDir,'dir'), mkdir(qcDir); end
    prep.qc_png = field_prep_qc(totalField_Hz, localField_Hz, bfrMask, B0, gyro, qcDir, usedBFR);
catch ME
    warning('场图 QC 出图失败(不影响结果): %s', ME.message);
end

fprintf('====================================================\n');
end

%% ========================================================================
function order = bfr_order(method)
switch method
    case 'LBV',    order = {'LBV','VSHARP','PDF'};
    case 'VSHARP', order = {'VSHARP','LBV','PDF'};
    case 'PDF',    order = {'PDF','LBV','VSHARP'};
    case 'AUTO',   order = {'LBV','VSHARP','PDF'};   % 文献: WH-QSM 配 LBV 最佳
    otherwise,     order = {'LBV','VSHARP','PDF'};
end
end

function tf = decide_unwrap(doUnwrap)
if islogical(doUnwrap) || isnumeric(doUnwrap)
    tf = logical(doUnwrap); return;
end
% 'auto' / 其他字符串: 对【多回波拟合频率场】默认不做空间解缠(它已是连续场,
% 再做会破坏低频结构)。仅当显式 true 时才解缠。
tf = false;
end

function [field_uw, method] = spatial_unwrap_field(data, field_Hz, Mask, voxel_size)
% 仅用成熟工具(SEPIA/MEDI 的 Laplacian)对【单回波 wrapped 相位】解缠。
% 不再提供自制的破坏性 fallback。若工具不可用, 返回空 -> 调用方跳过。
field_uw = []; method = '';
N = size(Mask);
try
    if exist('unwrapLaplacian','file')==2 && isfield(data,'phs_unwrap') && ~isempty(data.phs_unwrap)
        field_uw = unwrapLaplacian(double(data.phs_unwrap), N, voxel_size);
        method = 'Laplacian(MEDI unwrapLaplacian)';
    elseif exist('Wrapped2Unwrapped','file')==2 && isfield(data,'phs_unwrap') && ~isempty(data.phs_unwrap)
        field_uw = Wrapped2Unwrapped(double(data.phs_unwrap));
        method = 'Laplacian(SEPIA Wrapped2Unwrapped)';
    end
catch
    field_uw = [];
end
end


function [ok, why] = check_field(lf, Mask)
ok=false; why='';
if isempty(lf), why='空输出'; return; end
if ~isequal(size(lf),size(Mask)), why=sprintf('尺寸不符 %s vs %s',mat2str(size(lf)),mat2str(size(Mask))); return; end
v=lf(logical(Mask)); v=v(isfinite(v));
if isempty(v), why='mask 内无有限值'; return; end
if std(v)<=0, why='std=0(全常数)'; return; end
if all(v==0), why='全为0'; return; end
ok=true;
end

function tf = field_not_destroyed(orig, uw, Mask)
% 防止解缠把场抹平: 解缠后 std 不应低于原场的 30%。
vo=orig(logical(Mask)); vu=uw(logical(Mask));
vo=vo(isfinite(vo)); vu=vu(isfinite(vu));
tf = ~isempty(vo) && ~isempty(vu) && std(vu) >= 0.3*std(vo);
end

function p = field_prep_qc(total_Hz, local_Hz, Mask, B0, gyro, qcDir, usedBFR)
total_ppm = total_Hz./(gyro*B0); local_ppm = local_Hz./(gyro*B0);
[~,sz]=max(squeeze(sum(sum(Mask,1),2)));
fig=figure('Color','w','Position',[40 40 1200 480],'Visible','off');
tiledlayout(1,2,'Padding','compact','TileSpacing','compact');
nexttile; showf(total_ppm(:,:,sz),Mask(:,:,sz),[-0.1 0.1]); title('总场 total field (ppm)'); colorbar;
nexttile; showf(local_ppm(:,:,sz),Mask(:,:,sz),[-0.05 0.05]); title(sprintf('局部场 local field [%s] (ppm)',usedBFR)); colorbar;
sgtitle('场图制备 QC: 局部场应露出深部核团偶极结构(BFR 生效标志)');
p=fullfile(qcDir,'field_preprocess_qc.png');
try, exportgraphics(fig,p,'Resolution',150); catch, saveas(fig,p); end
close(fig);
fprintf('  场图 QC 已保存: %s\n', p);
end

function showf(img,mask,cl)
img=rot90(squeeze(img)); mask=rot90(squeeze(mask));
h=imagesc(img,cl); set(h,'AlphaData',double(mask)); axis image off; set(gca,'Color','k');
colormap(gca, gray(256));
end

function print_field_stats(name, vol, Mask)
v=double(vol(logical(Mask))); v=v(isfinite(v));
if isempty(v), fprintf('  %s: 空\n',name); return; end
fprintf('  %s: median=%.4g, std=%.4g, p1=%.4g, p99=%.4g (Hz)\n', ...
    name, median(v), std(v), prctile(v,1), prctile(v,99));
end

function v = get_cfg(cfg, pathCells, default)
v = default;
try
    s = cfg;
    for i=1:numel(pathCells)
        if isfield(s,pathCells{i}), s=s.(pathCells{i}); else, return; end
    end
    if ~isempty(s), v=s; end
catch, v=default; end
end
