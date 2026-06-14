function compare_subjects(results_normal, results_elderly, output_dir)
% compare_subjects.m
% ============================================================================
%   正常 vs 老年人 对比分析
%
%   输入:
%     results_normal  : 结构体，含 .chi (QSM 结果), .name, .mask
%     results_elderly : 同上
%     output_dir      : 输出目录
%
%   分析内容:
%     1. 三平面 QSM 对比（侧视图、冠状、轴状）
%     2. ROI 分析（深部灰质核团 — 铁沉积敏感区域）
%     3. 直方图对比
%     4. 差值图
% ============================================================================

if nargin < 3 || isempty(output_dir)
    output_dir = pwd;
end

if ~exist(output_dir, 'dir'), mkdir(output_dir); end

fprintf('============================================================\n');
fprintf(' 被试对比分析 - 正常 vs 老年人\n');
fprintf(' 正常: %s\n', results_normal.name);
fprintf(' 老年: %s\n', results_elderly.name);
fprintf(' 输出: %s\n', output_dir);
fprintf('============================================================\n\n');

chi_n = double(results_normal.chi);
chi_e = double(results_elderly.chi);
mask_n = logical(results_normal.mask);
mask_e = logical(results_elderly.mask);

% 公共 mask（两个都被试覆盖的区域）
common_mask = mask_n & mask_e;
fprintf('公共 mask 体素: %d (normal: %d, elderly: %d)\n', ...
    nnz(common_mask), nnz(mask_n), nnz(mask_e));

%% ====== 对比 1: 三平面 QSM 图 ======
fprintf('\n[1/4] 三平面 QSM 对比...\n');

% 选中间层
[Nx, Ny, Nz] = size(chi_n);
slice_x = round(Nx/2);
slice_y = round(Ny/2);
slice_z = round(Nz/2);

% 显示范围
clim = [-0.15 0.15];  % ppm

figure('Name', '三平面 QSM 对比', 'Position', [50 50 1600 900]);

% 正常被试
subplot(2, 3, 1);
imagesc(squeeze(chi_n(:, :, slice_z))' .* squeeze(mask_n(:, :, slice_z))');
axis image off; colormap(gca, redblue_cmap_local); caxis(clim);
title(sprintf('NORMAL: 轴状层 z=%d', slice_z), 'FontSize', 12);

subplot(2, 3, 2);
imagesc(squeeze(chi_n(:, slice_y, :))');
axis image off; colormap(gca, redblue_cmap_local); caxis(clim);
title(sprintf('NORMAL: 冠状层 y=%d', slice_y), 'FontSize', 12);

subplot(2, 3, 3);
imagesc(squeeze(chi_n(slice_x, :, :))');
axis image off; colormap(gca, redblue_cmap_local); caxis(clim);
title(sprintf('NORMAL: 矢状层 x=%d', slice_x), 'FontSize', 12);

% 老年人被试
subplot(2, 3, 4);
imagesc(squeeze(chi_e(:, :, slice_z))' .* squeeze(mask_e(:, :, slice_z))');
axis image off; colormap(gca, redblue_cmap_local); caxis(clim);
title(sprintf('ELDERLY: 轴状层 z=%d', slice_z), 'FontSize', 12);

subplot(2, 3, 5);
imagesc(squeeze(chi_e(:, slice_y, :))');
axis image off; colormap(gca, redblue_cmap_local); caxis(clim);
title(sprintf('ELDERLY: 冠状层 y=%d', slice_y), 'FontSize', 12);

subplot(2, 3, 6);
imagesc(squeeze(chi_e(slice_x, :, :))');
axis image off; colormap(gca, redblue_cmap_local); caxis(clim);
title(sprintf('ELDERLY: 矢状层 x=%d', slice_x), 'FontSize', 12);

% 添加公共 colorbar
colormap(redblue_cmap_local);
cb = colorbar('Position', [0.92 0.1 0.015 0.8]);
cb.Label.String = 'Susceptibility (ppm)';
cb.Label.FontSize = 12;

sgtitle(sprintf('WH-QSM QSM 对比: 正常 vs 老年人 (公共 mask 内)'), 'FontSize', 14);

saveas(gcf, fullfile(output_dir, 'compare_3view.png'));
saveas(gcf, fullfile(output_dir, 'compare_3view.fig'));
fprintf('  -> 保存: %s\n', fullfile(output_dir, 'compare_3view.png'));

%% ====== 对比 2: 深部灰质 ROI 分析 ======
fprintf('\n[2/4] 深部灰质 ROI 分析 (铁沉积敏感区域)...\n');

% 自动定义 ROIs（深部灰质核团 — 用简单阈值法）
% 这些区域铁含量高，QSM 表现为高信号（正磁化率）
% Globus Pallidus (GP), Putamen (PU), Caudate Nucleus (CN), Thalamus (TH)
ROIs = define_basal_ganglia_rois(chi_n, mask_n);

roi_names = fieldnames(ROIs);
fprintf('  ROI 定义: %s\n', strjoin(roi_names, ', '));

% 提取每个 ROI 的平均磁化率
roi_values_normal = zeros(length(roi_names), 1);
roi_values_elderly = zeros(length(roi_names), 1);

for k = 1:length(roi_names)
    rn = roi_names{k};
    roi_n = ROIs.(rn) & mask_n;
    roi_e = ROIs.(rn) & mask_e;
    roi_values_normal(k) = mean(chi_n(roi_n));
    roi_values_elderly(k) = mean(chi_e(roi_e));
end

% 绘制 ROI 对比柱状图
figure('Name', '深部灰质 ROI 对比', 'Position', [100 100 1200 600]);
bar_data = [roi_values_normal, roi_values_elderly];
b = bar(bar_data);
b(1).FaceColor = [0.2 0.6 0.8];
b(2).FaceColor = [0.8 0.4 0.2];
set(gca, 'XTickLabel', roi_names, 'FontSize', 12);
ylabel('Mean Susceptibility (ppm)', 'FontSize', 13);
title('深部灰质核团 QSM 对比 (正常 vs 老年人)', 'FontSize', 14);
legend({'NORMAL', 'ELDERLY'}, 'FontSize', 12);
grid on;

% 标注差值
for k = 1:length(roi_names)
    diff_val = roi_values_elderly(k) - roi_values_normal(k);
    text(k, max(bar_data(k,:)) + 0.005, sprintf('Δ=%.3f', diff_val), ...
        'HorizontalAlignment', 'center', 'FontSize', 11);
end

saveas(gcf, fullfile(output_dir, 'compare_roi_basal_ganglia.png'));
saveas(gcf, fullfile(output_dir, 'compare_roi_basal_ganglia.fig'));
fprintf('  -> 保存: %s\n', fullfile(output_dir, 'compare_roi_basal_ganglia.png'));

% 打印 ROI 数值
fprintf('\n  ROI 均值磁化率 (ppm):\n');
fprintf('  %-15s %-12s %-12s %-12s\n', 'ROI', 'NORMAL', 'ELDERLY', '差值 (老-正)');
fprintf('  %s\n', repmat('-', 1, 55));
for k = 1:length(roi_names)
    fprintf('  %-15s %-12.4f %-12.4f %-12.4f\n', ...
        roi_names{k}, roi_values_normal(k), roi_values_elderly(k), ...
        roi_values_elderly(k) - roi_values_normal(k));
end

%% ====== 对比 3: 直方图对比 ======
fprintf('\n[3/4] QSM 磁化率值分布直方图...\n');

figure('Name', '磁化率值分布', 'Position', [150 150 1000 600]);

chi_n_vals = chi_n(common_mask);
chi_e_vals = chi_e(common_mask);

edges = linspace(-0.2, 0.3, 80);
histogram(chi_n_vals, edges, 'FaceColor', [0.2 0.6 0.8], ...
    'FaceAlpha', 0.6, 'DisplayName', 'NORMAL');
hold on;
histogram(chi_e_vals, edges, 'FaceColor', [0.8 0.4 0.2], ...
    'FaceAlpha', 0.6, 'DisplayName', 'ELDERLY');
hold off;

xlabel('Susceptibility (ppm)', 'FontSize', 13);
ylabel('Voxel Count', 'FontSize', 13);
title('QSM 磁化率值分布对比', 'FontSize', 14);
legend('FontSize', 12); grid on;

% 标注统计量
mn = mean(chi_n_vals); me = mean(chi_e_vals);
sn = std(chi_n_vals);  se = std(chi_e_vals);
text(0.05, 0.95, sprintf('NORMAL  : μ=%.4f, σ=%.4f', mn, sn), ...
    'Units', 'normalized', 'FontSize', 11, 'Color', [0.2 0.4 0.6]);
text(0.05, 0.88, sprintf('ELDERLY : μ=%.4f, σ=%.4f', me, se), ...
    'Units', 'normalized', 'FontSize', 11, 'Color', [0.7 0.3 0.1]);

saveas(gcf, fullfile(output_dir, 'compare_histogram.png'));
saveas(gcf, fullfile(output_dir, 'compare_histogram.fig'));
fprintf('  -> 保存: %s\n', fullfile(output_dir, 'compare_histogram.png'));

fprintf('  NORMAL  均值=%.4f, 标准差=%.4f\n', mn, sn);
fprintf('  ELDERLY 均值=%.4f, 标准差=%.4f\n', me, se);

%% ====== 对比 4: 差值图（elderly - normal）======
fprintf('\n[4/4] 差值图 (elderly - normal)...\n');

% 在公共 mask 上做差值
diff_map = zeros(size(chi_n));
diff_map(common_mask) = chi_e(common_mask) - chi_n(common_mask);

% 显示
figure('Name', '差值图', 'Position', [200 200 1200 400]);
subplot(1, 3, 1);
imagesc(squeeze(diff_map(:,:,slice_z))' .* squeeze(common_mask(:,:,slice_z))');
axis image off; colormap(gca, redblue_cmap_local); caxis([-0.1 0.1]);
title(sprintf('轴状层 z=%d', slice_z), 'FontSize', 12);
subplot(1, 3, 2);
imagesc(squeeze(diff_map(:,slice_y,:))');
axis image off; colormap(gca, redblue_cmap_local); caxis([-0.1 0.1]);
title(sprintf('冠状层 y=%d', slice_y), 'FontSize', 12);
subplot(1, 3, 3);
imagesc(squeeze(diff_map(slice_x,:,:))');
axis image off; colormap(gca, redblue_cmap_local); caxis([-0.1 0.1]);
title(sprintf('矢状层 x=%d', slice_x), 'FontSize', 12);

colormap(redblue_cmap_local);
cb = colorbar;
cb.Label.String = 'Δ Susceptibility (ppm, elderly - normal)';
cb.Label.FontSize = 12;
sgtitle('差值图 (老年人 - 正常人): 红色=铁沉积增加', 'FontSize', 14);

saveas(gcf, fullfile(output_dir, 'compare_diff_map.png'));
saveas(gcf, fullfile(output_dir, 'compare_diff_map.fig'));
fprintf('  -> 保存: %s\n', fullfile(output_dir, 'compare_diff_map.png'));

%% ====== 保存数值结果 ======
fprintf('\n保存数值结果...\n');

results_table = table();
results_table.ROI = roi_names;
results_table.Normal = roi_values_normal;
results_table.Elderly = roi_values_elderly;
results_table.Diff = roi_values_elderly - roi_values_normal;

writetable(results_table, fullfile(output_dir, 'roi_comparison.csv'));
fprintf('  -> CSV: %s\n', fullfile(output_dir, 'roi_comparison.csv'));

% 全局统计
global_stats = struct( ...
    'mean_normal', mn, ...
    'std_normal', sn, ...
    'mean_elderly', me, ...
    'std_elderly', se, ...
    'mean_diff', me - mn);
save(fullfile(output_dir, 'global_stats.mat'), 'global_stats');
fprintf('  -> MAT: %s\n', fullfile(output_dir, 'global_stats.mat'));

fprintf('\n✅ 对比分析完成！\n');
end

%% =========================================================================
% 内部: 简单的基底神经节 ROI 定义
% =========================================================================
function ROIs = define_basal_ganglia_rois(chi, mask)
% 简化版：用磁化率阈值在深部灰质区域定义 ROI
% Globus Pallidus (GP): 高 χ (>0.15 ppm)
% Putamen (PU): 中高 χ (0.08-0.18 ppm)
% Caudate (CN): 中 χ (0.05-0.12 ppm)
% Thalamus (TH): 中 χ (0.03-0.10 ppm)
% White Matter (WM): 负 χ (<-0.02 ppm)

chi_in = chi .* double(mask);

ROIs = struct();
ROIs.GP = (chi_in > 0.10) & mask;
ROIs.PU = (chi_in > 0.05) & (chi_in <= 0.12) & mask;
ROIs.CN = (chi_in > 0.02) & (chi_in <= 0.08) & mask;
ROIs.TH = (chi_in > 0.0)  & (chi_in <= 0.06) & mask;
ROIs.WM = (chi_in < -0.02) & mask;
ROIs.GM = (chi_in > -0.01) & (chi_in <= 0.02) & mask;
end

%% =========================================================================
% 内部: 红蓝发散色图
%% =========================================================================
function cmap = redblue_cmap_local(n)
if nargin < 1, n = 256; end
neg = linspace(0, 1, n/2)';
pos = linspace(1, 0, n/2)';
blue = [0.05 0.18 0.68];
gray = [0.92 0.92 0.92];
red  = [0.72 0.05 0.05];

cmap = [blue + (gray-blue).*neg(1:end-1); ...
        gray; ...
        gray + (red-gray).*pos(2:end)];
cmap = max(min(cmap, 1), 0);
end
