function mod_visualization(all_results, all_names, data, metrics, cfg)
% mod_visualization.m
% QSM 可视化模块
%
% 输出:
%   1. 参考图 (chi_33, COSMOS) 三平面视图
%   2. 各算法三平面对比
%   3. 误差图对比
%   4. 指标柱状图
%   5. 散点图 (pred vs ref)
%   6. MEDI lambda 扫描对比 (如有)

N = data.N;
Mask = data.Mask;
chi_ref = data.chi_33;
clim_qsm = cfg.vis.clim_qsm;
clim_err = cfg.vis.clim_err;
doSave = cfg.vis.doSave;
figDir = cfg.figDir;

slice_idx = round(N / 2);
nAlg = size(all_results, 4);

fprintf('生成可视化 (共 %d 种算法)...\n', nAlg);

%% 1. 参考图三平面
fig = show_3planes(chi_ref, Mask, 'Reference chi\_33', clim_qsm, slice_idx);
save_fig(fig, fullfile(figDir, 'ref_chi33_3planes.png'), doSave, cfg.vis.resolution);

if any(data.chi_cosmos(:) ~= 0)
    fig = show_3planes(data.chi_cosmos, Mask, 'Reference COSMOS', clim_qsm, slice_idx);
    save_fig(fig, fullfile(figDir, 'ref_cosmos_3planes.png'), doSave, cfg.vis.resolution);
end

%% 2. 各算法三平面
for i = 1:nAlg
    fig = show_3planes(all_results(:,:,:,i), Mask, all_names{i}, clim_qsm, slice_idx);
    tag = matlab.lang.makeValidName(all_names{i});
    save_fig(fig, fullfile(figDir, sprintf('%s_3planes.png', tag)), doSave, cfg.vis.resolution);
end

%% 3. 全算法 axial 对比面板
fig = comparison_panel(all_results, all_names, Mask, clim_qsm, ...
    'axi', slice_idx(3), 'All algorithms — Axial');
save_fig(fig, fullfile(figDir, 'comparison_axial.png'), doSave, cfg.vis.resolution);

fig = comparison_panel(all_results, all_names, Mask, clim_qsm, ...
    'cor', slice_idx(2), 'All algorithms — Coronal');
save_fig(fig, fullfile(figDir, 'comparison_coronal.png'), doSave, cfg.vis.resolution);

fig = comparison_panel(all_results, all_names, Mask, clim_qsm, ...
    'sag', slice_idx(1), 'All algorithms — Sagittal');
save_fig(fig, fullfile(figDir, 'comparison_sagittal.png'), doSave, cfg.vis.resolution);

%% 4. 误差图面板
fig = error_panel(all_results, all_names, chi_ref, Mask, clim_err, ...
    'axi', slice_idx(3), 'Error maps — Axial');
save_fig(fig, fullfile(figDir, 'error_axial.png'), doSave, cfg.vis.resolution);

fig = error_panel(all_results, all_names, chi_ref, Mask, clim_err, ...
    'cor', slice_idx(2), 'Error maps — Coronal');
save_fig(fig, fullfile(figDir, 'error_coronal.png'), doSave, cfg.vis.resolution);

%% 5. 指标柱状图
fig = metrics_bar_chart(metrics, all_names);
save_fig(fig, fullfile(figDir, 'metrics_bar_chart.png'), doSave, cfg.vis.resolution);

%% 6. 散点图 (pred vs ref)
fig = scatter_pred_vs_ref(all_results, all_names, chi_ref, Mask);
save_fig(fig, fullfile(figDir, 'scatter_pred_vs_ref.png'), doSave, cfg.vis.resolution);

%% 7. MEDI lambda 扫描 (如有)
medi_idx = find(startsWith(all_names, 'MEDI'));
if numel(medi_idx) > 1
    medi_vols = all_results(:,:,:,medi_idx);
    medi_names_sub = all_names(medi_idx);
    
    fig = comparison_panel(medi_vols, medi_names_sub, Mask, clim_qsm, ...
        'axi', slice_idx(3), 'MEDI lambda sweep — Axial');
    save_fig(fig, fullfile(figDir, 'medi_sweep_axial.png'), doSave, cfg.vis.resolution);
    
    fig = error_panel(medi_vols, medi_names_sub, chi_ref, Mask, clim_err, ...
        'axi', slice_idx(3), 'MEDI lambda sweep error — Axial');
    save_fig(fig, fullfile(figDir, 'medi_sweep_error_axial.png'), doSave, cfg.vis.resolution);
end

fprintf('可视化完成，图像保存于: %s\n', figDir);

end

%% =========================================================================
% 三平面视图
% =========================================================================
function fig = show_3planes(vol, Mask, titleStr, clim, slice_idx)

vol = double(vol);
Mask = logical(Mask);
cmap = qsm_diverging_cmap(clim, 256);

fig = figure('Color', 'k', 'Name', titleStr, 'Position', [100 100 1200 400]);
tiledlayout(1, 3, 'Padding', 'compact', 'TileSpacing', 'compact');

% Sagittal
sag = rot90(squeeze(vol(slice_idx(1), :, :)));
msag = rot90(squeeze(Mask(slice_idx(1), :, :)));

% Coronal
cor = rot90(squeeze(vol(:, slice_idx(2), :)));
mcor = rot90(squeeze(Mask(:, slice_idx(2), :)));

% Axial
axi = rot90(squeeze(vol(:, :, slice_idx(3))));
maxi = rot90(squeeze(Mask(:, :, slice_idx(3))));

slices = {sag, cor, axi};
masks = {msag, mcor, maxi};
names = {'Sagittal', 'Coronal', 'Axial'};

for i = 1:3
    nexttile;
    h = imagesc(slices{i}, clim);
    set(h, 'AlphaData', double(masks{i}));
    axis image off;
    set(gca, 'Color', 'k');
    colormap(gca, cmap);
    title(names{i}, 'Color', 'w', 'FontSize', 13);
end

cb = colorbar;
cb.Color = 'w';
try cb.Layout.Tile = 'east'; catch; end
ylabel(cb, 'ppm', 'Color', 'w');

sgtitle(titleStr, 'Color', 'w', 'FontSize', 18, 'Interpreter', 'none');

end

%% =========================================================================
% 对比面板
% =========================================================================
function fig = comparison_panel(vols, names, Mask, clim, plane, slice_num, panel_title)

nAlg = size(vols, 4);
nCol = min(5, nAlg);
nRow = ceil(nAlg / nCol);

cmap = qsm_diverging_cmap(clim, 256);

fig = figure('Color', 'k', 'Name', panel_title, 'Position', [50 50 nCol*280 nRow*280]);
tiledlayout(nRow, nCol, 'Padding', 'compact', 'TileSpacing', 'compact');

for i = 1:nAlg
    vol = double(vols(:,:,:,i));
    [img, m] = extract_slice(vol, Mask, plane, slice_num);
    
    nexttile;
    h = imagesc(img, clim);
    set(h, 'AlphaData', double(m));
    axis image off;
    set(gca, 'Color', 'k');
    colormap(gca, cmap);
    title(names{i}, 'Color', 'w', 'FontSize', 11, 'Interpreter', 'none');
end

cb = colorbar;
cb.Color = 'w';
try cb.Layout.Tile = 'east'; catch; end
ylabel(cb, 'ppm', 'Color', 'w');

sgtitle(sprintf('%s | clim [%.3f %.3f] ppm', panel_title, clim(1), clim(2)), ...
    'Color', 'w', 'FontSize', 16, 'Interpreter', 'none');

end

%% =========================================================================
% 误差面板
% =========================================================================
function fig = error_panel(vols, names, ref, Mask, clim_err, plane, slice_num, panel_title)

nAlg = size(vols, 4);
nCol = min(5, nAlg);
nRow = ceil(nAlg / nCol);

cmap = qsm_diverging_cmap(clim_err, 256);

fig = figure('Color', 'k', 'Name', panel_title, 'Position', [50 50 nCol*280 nRow*280]);
tiledlayout(nRow, nCol, 'Padding', 'compact', 'TileSpacing', 'compact');

for i = 1:nAlg
    err = double(vols(:,:,:,i)) - double(ref);
    [img, m] = extract_slice(err, Mask, plane, slice_num);
    
    nexttile;
    h = imagesc(img, clim_err);
    set(h, 'AlphaData', double(m));
    axis image off;
    set(gca, 'Color', 'k');
    colormap(gca, cmap);
    title([names{i} ' - ref'], 'Color', 'w', 'FontSize', 10, 'Interpreter', 'none');
end

cb = colorbar;
cb.Color = 'w';
try cb.Layout.Tile = 'east'; catch; end
ylabel(cb, 'ppm error', 'Color', 'w');

sgtitle(sprintf('%s | clim [%.3f %.3f]', panel_title, clim_err(1), clim_err(2)), ...
    'Color', 'w', 'FontSize', 16, 'Interpreter', 'none');

end

%% =========================================================================
% 指标柱状图
% =========================================================================
function fig = metrics_bar_chart(metrics, names)

fig = figure('Color', 'w', 'Name', 'Metrics', 'Position', [100 100 1400 800]);
tiledlayout(2, 3, 'Padding', 'compact', 'TileSpacing', 'compact');

metric_fields = {'RMSE', 'HFEN', 'SSIM', 'WM_ERROR', 'GM_ERROR', 'Correlation'};
metric_titles = {'RMSE (lower=better)', 'HFEN (lower=better)', 'SSIM (higher=better)', ...
    'WM Error (lower=better)', 'GM Error (lower=better)', 'Correlation (higher=better)'};

nAlg = numel(names);
colors = lines(nAlg);

for m = 1:6
    nexttile;
    
    vals = metrics.(metric_fields{m});
    
    b = bar(vals, 'FaceColor', 'flat');
    for k = 1:nAlg
        b.CData(k,:) = colors(k,:);
    end
    
    set(gca, 'XTickLabel', names, 'XTickLabelRotation', 45, 'FontSize', 9);
    title(metric_titles{m}, 'FontSize', 12);
    ylabel(metric_fields{m});
    grid on;
end

sgtitle('QSM Evaluation Metrics', 'FontSize', 16);

end

%% =========================================================================
% 散点图
% =========================================================================
function fig = scatter_pred_vs_ref(all_results, all_names, chi_ref, Mask)

nAlg = size(all_results, 4);
nCol = min(4, nAlg);
nRow = ceil(nAlg / nCol);

fig = figure('Color', 'w', 'Name', 'Scatter', 'Position', [50 50 nCol*320 nRow*300]);
tiledlayout(nRow, nCol, 'Padding', 'compact', 'TileSpacing', 'compact');

ref_vals = chi_ref(Mask);

% 下采样用于绘图 (避免过多点)
n_total = numel(ref_vals);
if n_total > 50000
    idx_sub = randperm(n_total, 50000);
else
    idx_sub = 1:n_total;
end

ref_sub = ref_vals(idx_sub);

for i = 1:nAlg
    nexttile;
    
    vol = double(all_results(:,:,:,i));
    pred_vals = vol(Mask);
    pred_sub = pred_vals(idx_sub);
    
    scatter(ref_sub, pred_sub, 1, '.', 'MarkerEdgeAlpha', 0.1);
    hold on;
    
    % 对角线
    lims = [min(ref_sub), max(ref_sub)];
    plot(lims, lims, 'r-', 'LineWidth', 1.5);
    
    % 线性拟合
    p = polyfit(ref_sub, pred_sub, 1);
    plot(lims, polyval(p, lims), 'g--', 'LineWidth', 1.2);
    
    hold off;
    axis equal tight;
    xlabel('Reference (ppm)');
    ylabel('Predicted (ppm)');
    title(sprintf('%s (slope=%.2f)', all_names{i}, p(1)), ...
        'FontSize', 10, 'Interpreter', 'none');
    grid on;
end

sgtitle('Predicted vs Reference Susceptibility', 'FontSize', 14);

end

%% =========================================================================
% 辅助: 提取切片
% =========================================================================
function [img, m] = extract_slice(vol, Mask, plane, slice_num)

switch lower(plane)
    case 'sag'
        img = squeeze(vol(slice_num, :, :));
        m = squeeze(Mask(slice_num, :, :));
    case 'cor'
        img = squeeze(vol(:, slice_num, :));
        m = squeeze(Mask(:, slice_num, :));
    otherwise  % 'axi'
        img = squeeze(vol(:, :, slice_num));
        m = squeeze(Mask(:, :, slice_num));
end

img = rot90(img);
m = rot90(m);

end

%% =========================================================================
% 辅助: 保存图像
% =========================================================================
function save_fig(fig, filename, doSave, dpi)

if ~doSave
    return;
end

try
    drawnow;
    exportgraphics(fig, filename, 'Resolution', dpi);
catch
    try
        saveas(fig, filename);
    catch ME
        warning('图像保存失败 %s: %s', filename, ME.message);
    end
end

end