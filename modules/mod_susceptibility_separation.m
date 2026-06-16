function sep = mod_susceptibility_separation(data, chi_total_ppm, cfg)
% mod_susceptibility_separation.m
% ============================================================================
% Susceptibility source separation add-on for the real-data WH-QSM pipeline.
%
% Priority:
%   1) Call an installed, mature susceptibility-separation toolbox if a known
%      batch entry point is available (e.g. SNU-LIST chi-separation adapter).
%   2) Always export standard inputs (QSM, R2*, local field, mask) so the
%      external toolbox can be run/re-run reproducibly.
%   3) Optional exploratory fallback creates rough para/dia maps from R2*+QSM.
%      This fallback is clearly labelled and should not be treated as a
%      validated χ-separation algorithm.
% ============================================================================

sep = struct();
if nargin < 3 || ~isfield(cfg, 'sep') || ~isfield(cfg.sep, 'enable') || ~cfg.sep.enable
    sep.status = 'disabled';
    return;
end

if ~isfield(cfg, 'resultDir') || isempty(cfg.resultDir)
    error('cfg.resultDir is required.');
end
outDir = fullfile(cfg.resultDir, 'susceptibility_separation');
if ~exist(outDir, 'dir'), mkdir(outDir); end

Mask = logical(data.Mask);
chi_total_ppm = double(chi_total_ppm);
chi_total_ppm(~Mask) = 0;

if isfield(data, 'R2star_Hz') && ~isempty(data.R2star_Hz)
    R2star_Hz = double(data.R2star_Hz);
else
    error('data.R2star_Hz is missing. Multi-echo magnitude R2* mapping is required for susceptibility separation.');
end
R2star_Hz(~Mask) = 0;
localField_Hz = double(data.fieldmap_Hz);
localField_Hz(~Mask) = 0;

sep.outDir = outDir;
sep.method_requested = get_cfg(cfg, {'sep','method'}, 'auto');
sep.started_at = datestr(now, 31);
sep.inputs = export_chisep_inputs(outDir, data, chi_total_ppm, R2star_Hz, localField_Hz, Mask);

fprintf('\n============================================================\n');
fprintf(' Susceptibility source separation add-on\n');
fprintf('============================================================\n');
fprintf('Output dir: %s\n', outDir);
fprintf('Requested method: %s\n', sep.method_requested);
print_stats('QSM total ppm', chi_total_ppm, Mask);
print_stats('R2* Hz', R2star_Hz, Mask);

% -------------------------------------------------------------------------
% Add optional toolbox path if configured
% -------------------------------------------------------------------------
chiSepRoot = get_cfg(cfg, {'sep','chiSepRoot'}, '');
if ~isempty(chiSepRoot) && exist(chiSepRoot, 'dir') == 7
    addpath(chiSepRoot, '-begin');
    addpath(genpath(chiSepRoot), '-begin');
    fprintf('chi-separation toolbox path added: %s\n', chiSepRoot);
end

% -------------------------------------------------------------------------
% 1) Try toolbox batch entry
% -------------------------------------------------------------------------
[ok, toolboxResult, msg] = try_toolbox_chiseparation(data, chi_total_ppm, R2star_Hz, localField_Hz, Mask, cfg, outDir);
if ok
    sep.status = 'toolbox_success';
    sep.method = toolboxResult.method;
    sep.chi_para = toolboxResult.chi_para;
    sep.chi_dia = toolboxResult.chi_dia;
    sep.chi_dia_abs = abs(toolboxResult.chi_dia);
    sep.extra = toolboxResult;
    save_outputs_and_figures(sep, data, chi_total_ppm, R2star_Hz, Mask, outDir, 'TOOLBOX');
    fprintf('Susceptibility separation completed using toolbox method: %s\n', sep.method);
    return;
else
    fprintf('Toolbox susceptibility separation not executed: %s\n', msg);
end

% -------------------------------------------------------------------------
% 2) Optional exploratory fallback
% -------------------------------------------------------------------------
allowFallback = get_cfg(cfg, {'sep','allow_exploratory_fallback'}, false);
if allowFallback
    sep.status = 'exploratory_fallback';
    sep.method = 'EXPLORATORY_R2star_QSM_linear_preview_NOT_VALIDATED';
    sep.warning = ['This fallback is not a validated chi-separation toolbox algorithm. ' ...
                   'Use it for QC/preview only; install/configure chi-separation/APART-QSM for scientific analysis.'];
    sep = run_exploratory_r2star_qsm_preview(sep, chi_total_ppm, R2star_Hz, Mask, cfg);
    save_outputs_and_figures(sep, data, chi_total_ppm, R2star_Hz, Mask, outDir, 'EXPLORATORY');
    warning('%s', sep.warning);
else
    sep.status = 'skipped_no_toolbox';
    sep.method = 'none';
    sep.message = ['No recognised susceptibility-separation toolbox batch function was found. ' ...
                   'Inputs were exported; install/configure SNU-LIST chi-separation or APART-QSM adapter.'];
    save(fullfile(outDir, 'susceptibility_separation_skipped.mat'), 'sep', '-v7.3');
end

sep.finished_at = datestr(now, 31);
fprintf('============================================================\n\n');
end

%% =========================================================================
function inputs = export_chisep_inputs(outDir, data, chi_total_ppm, R2star_Hz, localField_Hz, Mask)
inputs = struct();
inputs.chi_total_ppm = fullfile(outDir, 'input_chi_total_ppm.nii');
inputs.R2star_Hz = fullfile(outDir, 'input_R2star_Hz.nii');
inputs.localField_Hz = fullfile(outDir, 'input_localField_Hz.nii');
inputs.mask = fullfile(outDir, 'input_mask.nii');
inputs.mat = fullfile(outDir, 'chisep_inputs.mat');

try
    niftiwrite(single(chi_total_ppm), inputs.chi_total_ppm);
    niftiwrite(single(R2star_Hz), inputs.R2star_Hz);
    niftiwrite(single(localField_Hz), inputs.localField_Hz);
    niftiwrite(uint8(Mask), inputs.mask);
catch ME
    warning('Could not export one or more NIfTI inputs for chi-separation: %s', ME.message);
end

spatial_res = data.spatial_res; %#ok<NASGU>
echo_times_ms = data.echo_times_ms; %#ok<NASGU>
B0 = data.B0; %#ok<NASGU>
B0_dir = data.B0_dir; %#ok<NASGU>
save(inputs.mat, 'chi_total_ppm', 'R2star_Hz', 'localField_Hz', 'Mask', ...
    'spatial_res', 'echo_times_ms', 'B0', 'B0_dir', '-v7.3');
end

%% =========================================================================
function [ok, result, msg] = try_toolbox_chiseparation(data, chi_total_ppm, R2star_Hz, localField_Hz, Mask, cfg, outDir)
ok = false;
result = struct();
msg = 'no recognised batch function found';

% Preferred explicit user adapter. This avoids guessing private APIs of a
% toolbox release. The adapter must return a struct with chi_para and chi_dia.
adapterName = get_cfg(cfg, {'sep','adapter_function'}, '');
if ~isempty(adapterName) && function_exists_on_path(adapterName)
    try
        f = str2func(adapterName);
        result = f(data, chi_total_ppm, R2star_Hz, localField_Hz, Mask, cfg, outDir);
        if is_valid_sep_result(result, Mask)
            ok = true; result.method = ['adapter:' adapterName]; msg = 'ok'; return;
        else
            msg = sprintf('adapter %s returned invalid result', adapterName);
        end
    catch ME
        msg = sprintf('adapter %s failed: %s', adapterName, ME.message);
    end
    return;
end

% Conservative auto-detection for public/user batch wrappers. We intentionally
% do NOT run demo scripts such as Chisep_script.m because they often contain
% hard-coded paths and are not a callable API.
candidates = { ...
    'chi_separation_batch', ...
    'chisep_batch', ...
    'run_chisep_batch', ...
    'xseparation_batch', ...
    'apart_qsm_single_ori_batch'};

input = struct();
input.chi_total_ppm = chi_total_ppm;
input.R2star_Hz = R2star_Hz;
input.localField_Hz = localField_Hz;
input.Mask = Mask;
input.spatial_res = data.spatial_res;
input.echo_times_sec = data.echo_times_sec;
input.B0 = data.B0;
input.B0_dir = data.B0_dir;
input.outDir = outDir;

for i = 1:numel(candidates)
    name = candidates{i};
    if ~function_exists_on_path(name)
        continue;
    end
    try
        f = str2func(name);
        result = f(input);
        if is_valid_sep_result(result, Mask)
            ok = true; result.method = name; msg = 'ok'; return;
        else
            msg = sprintf('%s returned invalid output', name);
        end
    catch ME
        msg = sprintf('%s failed: %s', name, ME.message);
    end
end
end

function tf = is_valid_sep_result(result, Mask)
tf = isstruct(result) && isfield(result, 'chi_para') && isfield(result, 'chi_dia') && ...
     isequal(size(result.chi_para), size(Mask)) && isequal(size(result.chi_dia), size(Mask));
end

%% =========================================================================
function sep = run_exploratory_r2star_qsm_preview(sep, chi_total_ppm, R2star_Hz, Mask, cfg)
% Algebraic preview based on the simplified relations:
%   chi_total = chi_para + chi_dia, chi_dia <= 0
%   R2sus ≈ k * (chi_para - chi_dia)
% This is NOT a replacement for chi-separation/APART-QSM.

k = get_cfg(cfg, {'sep','r2star_to_chi_abs_HzPerPpm'}, 137.0);
base_pct = get_cfg(cfg, {'sep','r2star_baseline_percentile'}, 5.0);
r2_vals = R2star_Hz(Mask);
r2_base = prctile(r2_vals(isfinite(r2_vals)), base_pct);
R2sus = max(R2star_Hz - r2_base, 0);
chi_abs = R2sus ./ max(k, eps);
chi_abs = max(chi_abs, abs(chi_total_ppm));
chi_para = 0.5 * (chi_total_ppm + chi_abs);
chi_dia = 0.5 * (chi_total_ppm - chi_abs);
chi_para = max(chi_para, 0);
chi_dia = min(chi_dia, 0);
chi_para(~Mask) = 0;
chi_dia(~Mask) = 0;
sep.chi_para = chi_para;
sep.chi_dia = chi_dia;
sep.chi_dia_abs = abs(chi_dia);
sep.r2star_baseline_Hz = r2_base;
sep.r2star_to_chi_abs_HzPerPpm = k;
end

%% =========================================================================
function save_outputs_and_figures(sep, data, chi_total_ppm, R2star_Hz, Mask, outDir, label)
chi_para = sep.chi_para;
chi_dia = sep.chi_dia;
chi_dia_abs = abs(chi_dia);
chi_recombined = chi_para + chi_dia;

save(fullfile(outDir, 'susceptibility_separation_results.mat'), ...
    'sep', 'chi_total_ppm', 'R2star_Hz', 'chi_para', 'chi_dia', 'chi_dia_abs', 'chi_recombined', 'Mask', '-v7.3');
try
    niftiwrite(single(chi_para), fullfile(outDir, 'chi_para_ppm.nii'));
    niftiwrite(single(chi_dia), fullfile(outDir, 'chi_dia_ppm.nii'));
    niftiwrite(single(chi_dia_abs), fullfile(outDir, 'chi_dia_abs_ppm.nii'));
    niftiwrite(single(chi_recombined), fullfile(outDir, 'chi_recombined_ppm.nii'));
catch ME
    warning('Could not save susceptibility separation NIfTI outputs: %s', ME.message);
end

fig = figure('Name', ['Susceptibility separation ' label], 'Position', [60 60 1500 850], 'Color', 'w');
tiledlayout(2,3, 'Padding','compact', 'TileSpacing','compact');
% --- QC 选层：自动定位到基底节层（按 χ_para 深部顺磁热区质心）---
% 原逻辑用脑mask几何质心 sz=round(mean(z))，会偏腹侧/颅底，切不到苍白球/壳核，
% 导致 χ-sep 最该展示的深部顺磁对比缺席（"图看着淡/可疑"的真根因）。
% 改为：在腐蚀后的深部mask里，找 χ_para 强顺磁信号积分最大的轴位层。
[~,~,sz_center] = mask_center_slices(Mask);
sz = select_basal_ganglia_slice(chi_para, Mask, sz_center);
fprintf('QC 选层(基底节自动定位): z=%d (脑质心 z=%d)\n', sz, sz_center);
clim_chi = [-0.10 0.10];   % 收窄(原 ±0.15)以看清深部铁核对比
clim_comp = [0 0.15];      % χpara/|χdia| 多在 0-0.15ppm，原 0-0.20 偏宽显淡
nexttile; show_color(chi_total_ppm(:,:,sz), Mask(:,:,sz), clim_chi); title('WH-QSM total \chi ppm'); colorbar;
nexttile; show_color(R2star_Hz(:,:,sz), Mask(:,:,sz), [0 prctile(R2star_Hz(Mask),95)]); title('R2* Hz'); colorbar;
nexttile; show_color(chi_para(:,:,sz), Mask(:,:,sz), clim_comp); title('\chi_{para} ppm'); colorbar;
nexttile; show_color(chi_dia_abs(:,:,sz), Mask(:,:,sz), clim_comp); title('|\chi_{dia}| ppm'); colorbar;
nexttile; show_color(chi_recombined(:,:,sz), Mask(:,:,sz), clim_chi); title('\chi_{para}+\chi_{dia} ppm'); colorbar;
nexttile; axis off;
text(0, .9, sprintf('Method: %s', sep.method), 'Interpreter','none', 'FontWeight','bold');
text(0, .75, sprintf('Status: %s', sep.status), 'Interpreter','none');
text(0, .60, sprintf('Subject: %s', data.subject_name), 'Interpreter','none');
if isfield(sep, 'warning'), text(0, .42, sep.warning, 'Interpreter','none'); end
sgtitle(['Susceptibility source separation QC (' label ')'], 'Interpreter','none');
try
    exportgraphics(fig, fullfile(outDir, 'susceptibility_separation_qc.png'), 'Resolution', 200);
catch
    saveas(fig, fullfile(outDir, 'susceptibility_separation_qc.png'));
end
end

%% =========================================================================
function tf = function_exists_on_path(name)
code = exist(name, 'file');
tf = any(code == [2 3 6]);
end

function v = get_cfg(cfg, pathCells, default)
v = default;
try
    s = cfg;
    for i = 1:numel(pathCells)
        if isfield(s, pathCells{i})
            s = s.(pathCells{i});
        else
            return;
        end
    end
    if ~isempty(s), v = s; end
catch
    v = default;
end
end

function print_stats(name, vol, Mask)
v = double(vol(Mask)); v = v(isfinite(v));
if isempty(v), fprintf('%s: empty\n', name); return; end
fprintf('%s: median=%.6g, p01=%.6g, p99=%.6g, std=%.6g\n', name, median(v), prctile(v,1), prctile(v,99), std(v));
end

function [sx, sy, sz] = mask_center_slices(mask)
idx = find(mask);
if isempty(idx)
    N = size(mask); sx=round(N(1)/2); sy=round(N(2)/2); sz=round(N(3)/2); return;
end
[x,y,z] = ind2sub(size(mask), idx);
sx=round(mean(x)); sy=round(mean(y)); sz=round(mean(z));
N=size(mask); sx=max(1,min(N(1),sx)); sy=max(1,min(N(2),sy)); sz=max(1,min(N(3),sz));
end

function sz = select_basal_ganglia_slice(chi_para, mask, sz_fallback)
% 自动选取"基底节层"用于 QC 显示：找腐蚀后的深部脑mask里 χ_para 强顺磁信号
% 积分最大的轴位层（苍白球/壳核是脑内最强顺磁源）。带稳健回退。
%
% 思路（不靠固定中间层，不靠肉眼）：
%   1) 腐蚀脑mask，剔除皮层/边缘/静脉壁等浅层强信号，只保留深部组织。
%   2) 对每个轴位层 z，阈值取深部mask内 χ_para 的高分位(p85)，把强顺磁体素的
%      (值-阈值) 之和作为该层"深部铁核含量"得分。
%   3) 限定在脑中部 [0.30, 0.75]*Nz 的轴位范围内搜索（基底节解剖先验，
%      避免选到颅底强信号或顶部）。
%   4) 取得分最大的层；若 χ_para 全空/异常则回退到几何质心 sz_fallback。
sz = sz_fallback;
try
    N = size(mask);
    if numel(N) < 3, return; end
    Nz = N(3);

    % --- 1) 腐蚀mask到深部（核大小按面内体素，至少3）---
    er = max(3, round(min(N(1:2)) * 0.04));
    deep = erode_mask_3d(mask, er);
    if nnz(deep) < 50, deep = mask; end   % 腐蚀过头则不腐蚀

    cp = double(chi_para);
    cp(~isfinite(cp)) = 0;

    % --- 2)+3) 按轴位层在解剖先验范围内打分 ---
    z0 = max(1, round(0.30 * Nz));
    z1 = min(Nz, round(0.75 * Nz));
    scores = -inf(Nz,1);
    for z = z0:z1
        m = deep(:,:,z);
        if nnz(m) < 20, continue; end
        s = cp(:,:,z);
        vals = s(m);
        vals = vals(isfinite(vals));
        if isempty(vals), continue; end
        thr = prctile(vals, 85);          % 该层深部 χ_para 高分位阈
        if ~isfinite(thr) || thr <= 0, thr = 0; end
        pos = vals(vals > thr) - thr;     % 强顺磁体素超出阈值的部分
        scores(z) = sum(pos);             % 积分 = 深部铁核含量
    end

    [best, zbest] = max(scores);
    if isfinite(best) && best > 0
        sz = zbest;
    end
    sz = max(1, min(Nz, round(sz)));
catch ME
    warning('select_basal_ganglia_slice 失败, 回退几何质心: %s', ME.message);
    sz = sz_fallback;
end
end

function out = erode_mask_3d(mask, r)
% 简单的3D方形腐蚀（不依赖 image toolbox 的 imerode/strel）。
% 用 r 半径的 box 腐蚀：逐维 min-filter（位移取交集）。
mask = logical(mask);
out = mask;
N = size(mask);
for d = 1:3
    acc = out;
    for s = 1:r
        % 正负方向各位移 s，取与原mask的交集（腐蚀=任一邻域为0则该点为0）
        shp = shift_logical(out, d, s);
        shm = shift_logical(out, d, -s);
        acc = acc & shp & shm;
    end
    out = acc;
end
out = out & mask;
end

function y = shift_logical(x, dim, s)
% 沿 dim 位移 s 个体素，越界处补 false（保证边缘被腐蚀掉）。
y = false(size(x));
N = size(x);
idx = repmat({':'}, 1, ndims(x));
idy = idx;
if s >= 0
    src = 1:(N(dim)-s);
    dst = (1+s):N(dim);
else
    s = -s;
    src = (1+s):N(dim);
    dst = 1:(N(dim)-s);
end
if isempty(src), return; end
idx{dim} = dst; idy{dim} = src;
y(idx{:}) = x(idy{:});
end

function show_color(img, mask, clim)
img = rot90(squeeze(img)); mask = rot90(squeeze(mask));
h = imagesc(img, clim); set(h,'AlphaData',double(mask)); axis image off; set(gca,'Color','k'); colormap(gca, turbo(256));
end
