function compare_subjects(results_normal, results_elderly, output_dir)
% compare_subjects.m (v5 - WH-QSM QC visualisation, no subtraction)
% ============================================================================
% Conservative QC visualisation for two real-subject WH-QSM outputs.
%
% This function intentionally does NOT compute elderly-normal voxel-wise
% subtraction. Without registration, subtraction is not valid. Visual outputs:
%   1) per-subject QC panels: magnitude / field map / QSM in native space
%   2) side-by-side WH-QSM three-plane view
%   3) within-mask histograms
%   4) descriptive CSV/MAT summary
% ============================================================================

if nargin < 3 || isempty(output_dir)
    output_dir = pwd;
end
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

fprintf('============================================================\n');
fprintf(' WH-QSM two-subject QC summary (no subtraction)\n');
fprintf(' NORMAL : %s\n', results_normal.name);
fprintf(' ELDERLY: %s\n', results_elderly.name);
fprintf(' Output : %s\n', output_dir);
fprintf('============================================================\n\n');

chi_n = double(results_normal.chi);
chi_e = double(results_elderly.chi);
mask_n = logical(results_normal.mask);
mask_e = logical(results_elderly.mask);

if ~isequal(size(chi_n), size(mask_n)) || ~isequal(size(chi_e), size(mask_e))
    error('QSM and mask sizes do not match for at least one subject.');
end

% 显示窗位: 深部铁核 χ 量级约 0.10-0.18ppm，用 ±0.10 比 ±0.15 更能看清对比
% (窗位越宽图越"淡")。可在 whqsm_local_paths.m 的 P.qsmDisplayClim 调整。
qsm_clim = [-0.10 0.10];
try
    P = whqsm_local_paths();
    if isfield(P,'qsmDisplayClim') && numel(P.qsmDisplayClim)==2
        qsm_clim = P.qsmDisplayClim;
    end
catch
end
field_clim = [-0.20 0.20];

%% 1. Per-subject QC panels
fprintf('[1/4] Per-subject QC panels...\n');
fig = subject_qc_panel(results_normal, 'NORMAL', qsm_clim, field_clim);
save_figure(fig, fullfile(output_dir, 'qc_normal_native_space.png'));
fprintf('  -> %s\n', fullfile(output_dir, 'qc_normal_native_space.png'));

fig = subject_qc_panel(results_elderly, 'ELDERLY', qsm_clim, field_clim);
save_figure(fig, fullfile(output_dir, 'qc_elderly_native_space.png'));
fprintf('  -> %s\n', fullfile(output_dir, 'qc_elderly_native_space.png'));

%% 2. Side-by-side 3-plane display
fprintf('\n[2/4] Side-by-side WH-QSM three-plane display...\n');
fig = figure('Name', 'WH-QSM side-by-side 3-plane QC', 'Position', [50 50 1600 900], 'Color', 'w');

plot_subject_3planes(chi_n, mask_n, results_normal, 1, qsm_clim, 'NORMAL');
plot_subject_3planes(chi_e, mask_e, results_elderly, 4, qsm_clim, 'ELDERLY');

colormap(redblue_cmap_local(256));
cb = colorbar('Position', [0.93 0.12 0.015 0.76]);
cb.Label.String = 'Susceptibility (ppm)';
cb.Label.FontSize = 12;
sgtitle('WH-QSM QC: side-by-side native-space views (no registration / no subtraction)', ...
    'FontSize', 14, 'Interpreter', 'none');

save_figure(fig, fullfile(output_dir, 'compare_3view.png'));
saveas(fig, fullfile(output_dir, 'compare_3view.fig'));
fprintf('  -> %s\n', fullfile(output_dir, 'compare_3view.png'));

%% 3. Histogram / distribution summary inside each subject mask
fprintf('\n[3/4] Within-mask histograms...\n');
vals_n = chi_n(mask_n);
vals_e = chi_e(mask_e);
vals_n = vals_n(isfinite(vals_n));
vals_e = vals_e(isfinite(vals_e));

fig = figure('Name', 'WH-QSM within-mask histogram', 'Position', [100 100 1100 650], 'Color', 'w');
edges = linspace(-0.3, 0.4, 100);
histogram(vals_n, edges, 'FaceColor', [0.2 0.55 0.85], 'FaceAlpha', 0.55, 'DisplayName', 'NORMAL');
hold on;
histogram(vals_e, edges, 'FaceColor', [0.85 0.40 0.20], 'FaceAlpha', 0.55, 'DisplayName', 'ELDERLY');
hold off;
xlabel('Susceptibility (ppm)', 'FontSize', 13);
ylabel('Voxel count', 'FontSize', 13);
title('WH-QSM within-mask susceptibility distribution', 'FontSize', 14);
legend('FontSize', 12);
grid on;

sn = summary_stats(vals_n);
se = summary_stats(vals_e);
annotation('textbox', [0.58 0.66 0.32 0.20], 'String', sprintf( ...
    ['NORMAL:  mean %.4f, std %.4f, median %.4f\n' ...
     'ELDERLY: mean %.4f, std %.4f, median %.4f\n' ...
     'Note: descriptive native-space QC only'], ...
     sn.mean, sn.std, sn.median, se.mean, se.std, se.median), ...
    'FitBoxToText', 'on', 'BackgroundColor', 'w', 'EdgeColor', [0.7 0.7 0.7]);

save_figure(fig, fullfile(output_dir, 'compare_histogram.png'));
saveas(fig, fullfile(output_dir, 'compare_histogram.fig'));
fprintf('  -> %s\n', fullfile(output_dir, 'compare_histogram.png'));

%% 4. Save descriptive summary CSV / MAT
fprintf('\n[4/4] Saving descriptive summaries...\n');
summary_table = build_summary_table(results_normal, vals_n, results_elderly, vals_e);
try
    writetable(summary_table, fullfile(output_dir, 'subject_summary.csv'));
    fprintf('  -> %s\n', fullfile(output_dir, 'subject_summary.csv'));
catch ME
    warning('Could not write subject_summary.csv: %s', ME.message);
end

results = struct();
results.normal = results_normal;
results.elderly = results_elderly;
results.normal_stats = sn;
results.elderly_stats = se;
save(fullfile(output_dir, 'all_results.mat'), 'results', 'summary_table', '-v7.3');
fprintf('  -> %s\n', fullfile(output_dir, 'all_results.mat'));

fprintf('\nDescriptive WH-QSM summary:\n');
fprintf('  %-8s %-10s %-10s %-10s %-10s %-10s %-10s %-10s\n', ...
    'Group', 'Nvox', 'Mean', 'Std', 'Median', 'P05', 'P95', 'P99');
fprintf('  %s\n', repmat('-', 1, 82));
fprintf('  %-8s %-10d %-10.5f %-10.5f %-10.5f %-10.5f %-10.5f %-10.5f\n', ...
    'NORMAL', sn.n, sn.mean, sn.std, sn.median, sn.p05, sn.p95, sn.p99);
fprintf('  %-8s %-10d %-10.5f %-10.5f %-10.5f %-10.5f %-10.5f %-10.5f\n', ...
    'ELDERLY', se.n, se.mean, se.std, se.median, se.p05, se.p95, se.p99);

fprintf('\n✅ WH-QSM QC summary complete. No voxel-wise subtraction was generated.\n');
end

%% =========================================================================
function fig = subject_qc_panel(result, label, qsm_clim, field_clim)
chi = double(result.chi);
mask = logical(result.mask);
mag = getfield_or(result, 'magn', double(mask));
field_ppm = getfield_or(result, 'local_field_ppm', zeros(size(mask)));
mag = double(mag);
field_ppm = double(field_ppm);

[sx, sy, sz] = mask_center_slices(mask);
axDim = get_display_dim_from_result(result, 'axial', 3);
coDim = get_display_dim_from_result(result, 'coronal', 2);
saDim = get_display_dim_from_result(result, 'sagittal', 1);
axIdx = center_index_along_dim(mask, axDim);
coIdx = center_index_along_dim(mask, coDim);
saIdx = center_index_along_dim(mask, saDim);

fig = figure('Name', ['WH-QSM QC ' label], 'Position', [60 60 1650 850], 'Color', 'w');
tiledlayout(2,4, 'Padding', 'compact', 'TileSpacing', 'compact');

% Row 1: patient-axial QC at orientation-aware centre
[axMag, axMask] = extract_dim_slice_local(mag, mask, axDim, axIdx);
[axField, ~] = extract_dim_slice_local(field_ppm, mask, axDim, axIdx);
[axChi, ~] = extract_dim_slice_local(chi, mask, axDim, axIdx);
nexttile; show_gray_slice(axMag, axMask); title(sprintf('%s magnitude axial dim%d=%d', label, axDim, axIdx), 'Interpreter','none');
nexttile; show_mask_slice(axMask); title('Brain mask axial', 'Interpreter','none');
nexttile; show_color_slice(axField, axMask, field_clim, redblue_cmap_local(256)); title('Input field ppm axial', 'Interpreter','none'); colorbar;
nexttile; show_color_slice(axChi, axMask, qsm_clim, redblue_cmap_local(256)); title('WH-QSM ppm axial', 'Interpreter','none'); colorbar;

% Row 2: patient-axial/coronal/sagittal three planes
[img, msk] = extract_dim_slice_local(chi, mask, axDim, axIdx); nexttile; show_color_slice(img, msk, qsm_clim, redblue_cmap_local(256)); title(sprintf('QSM axial dim%d=%d', axDim, axIdx), 'Interpreter','none');
[img, msk] = extract_dim_slice_local(chi, mask, coDim, coIdx); nexttile; show_color_slice(img, msk, qsm_clim, redblue_cmap_local(256)); title(sprintf('QSM coronal dim%d=%d', coDim, coIdx), 'Interpreter','none');
[img, msk] = extract_dim_slice_local(chi, mask, saDim, saIdx); nexttile; show_color_slice(img, msk, qsm_clim, redblue_cmap_local(256)); title(sprintf('QSM sagittal dim%d=%d', saDim, saIdx), 'Interpreter','none');
nexttile; axis off;
st = summary_stats(chi(mask));
text(0, 0.95, sprintf('%s: %s', label, result.name), 'FontWeight','bold', 'Interpreter','none');
text(0, 0.82, sprintf('Matrix: %s', mat2str(size(mask))), 'Interpreter','none');
text(0, 0.72, sprintf('Voxel: %s mm', mat2str(getfield_or(result,'spatial_res',[]), 5)), 'Interpreter','none');
text(0, 0.62, sprintf('TE: %s ms', mat2str(getfield_or(result,'echo_times_ms',[]), 5)), 'Interpreter','none');
text(0, 0.52, sprintf('B0: %.3g T', getfield_or(result,'B0',NaN)), 'Interpreter','none');
text(0, 0.42, sprintf('Fit: %s', getfield_or(result,'phase_fit_method','')), 'Interpreter','none');
text(0, 0.28, sprintf('QSM p01/p99: %.3f / %.3f ppm', st.p01, st.p99), 'Interpreter','none');
text(0, 0.18, sprintf('QSM mean/std: %.4f / %.4f ppm', st.mean, st.std), 'Interpreter','none');

sgtitle(sprintf('WH-QSM native-space QC: %s (no registration)', label), 'Interpreter','none');
end

%% =========================================================================
function plot_subject_3planes(chi, mask, result, start_subplot, clim, label)
axDim = get_display_dim_from_result(result, 'axial', 3);
coDim = get_display_dim_from_result(result, 'coronal', 2);
saDim = get_display_dim_from_result(result, 'sagittal', 1);
axIdx = center_index_along_dim(mask, axDim);
coIdx = center_index_along_dim(mask, coDim);
saIdx = center_index_along_dim(mask, saDim);

subtitle = sprintf('%s: %s | TE=%s ms | B0=%.3gT | Acq=%s', ...
    label, result.name, mat2str(getfield_or(result, 'echo_times_ms', []), 5), getfield_or(result, 'B0', NaN), ...
    getfield_or(result, 'acquisition_plane', 'UNKNOWN'));

subplot(2,3,start_subplot);
[img, msk] = extract_dim_slice_local(chi, mask, axDim, axIdx);
show_color_slice(img, msk, clim, redblue_cmap_local(256));
title(sprintf('%s axial dim%d=%d', label, axDim, axIdx), 'Interpreter', 'none');
ylabel(subtitle, 'Interpreter', 'none', 'FontSize', 9);

subplot(2,3,start_subplot+1);
[img, msk] = extract_dim_slice_local(chi, mask, coDim, coIdx);
show_color_slice(img, msk, clim, redblue_cmap_local(256));
title(sprintf('%s coronal dim%d=%d', label, coDim, coIdx), 'Interpreter', 'none');

subplot(2,3,start_subplot+2);
[img, msk] = extract_dim_slice_local(chi, mask, saDim, saIdx);
show_color_slice(img, msk, clim, redblue_cmap_local(256));
title(sprintf('%s sagittal dim%d=%d', label, saDim, saIdx), 'Interpreter', 'none');
end

function [sx, sy, sz] = mask_center_slices(mask)
idx = find(mask);
if isempty(idx)
    sz0 = size(mask);
    sx = round(sz0(1)/2); sy = round(sz0(2)/2); sz = round(sz0(3)/2);
    return;
end
[x, y, z] = ind2sub(size(mask), idx);
sx = round(mean(x)); sy = round(mean(y)); sz = round(mean(z));
sz0 = size(mask);
sx = max(1, min(sz0(1), sx));
sy = max(1, min(sz0(2), sy));
sz = max(1, min(sz0(3), sz));
end

function show_color_slice(img, mask, clim, cmap)
img = rot90(squeeze(img));
mask = rot90(squeeze(mask));
h = imagesc(img, clim);
set(h, 'AlphaData', double(mask));
axis image off;
set(gca, 'Color', [0 0 0]);
colormap(gca, cmap);
end

function show_gray_slice(img, mask)
img = rot90(squeeze(img));
mask = rot90(squeeze(mask));
v = img(mask > 0);
v = v(isfinite(v));
if isempty(v)
    clim = [min(img(:)) max(img(:))];
else
    clim = [prctile(v, 1) prctile(v, 99.5)];
end
if clim(1) == clim(2), clim = clim + [-1 1]; end
h = imagesc(img, clim);
set(h, 'AlphaData', double(mask));
axis image off;
set(gca, 'Color', [0 0 0]);
colormap(gca, gray(256));
end

function show_mask_slice(mask)
imagesc(rot90(squeeze(mask)), [0 1]);
axis image off;
colormap(gca, gray(2));
end

function save_figure(fig, filename)
try
    drawnow;
    exportgraphics(fig, filename, 'Resolution', 200);
catch
    saveas(fig, filename);
end
end

function st = summary_stats(vals)
vals = vals(:);
vals = vals(isfinite(vals));
st = struct();
st.n = numel(vals);
if isempty(vals)
    st.mean = NaN; st.std = NaN; st.median = NaN; st.min = NaN; st.max = NaN;
    st.p01 = NaN; st.p05 = NaN; st.p95 = NaN; st.p99 = NaN;
    return;
end
st.mean = mean(vals);
st.std = std(vals);
st.median = median(vals);
st.min = min(vals);
st.max = max(vals);
st.p01 = prctile(vals, 1);
st.p05 = prctile(vals, 5);
st.p95 = prctile(vals, 95);
st.p99 = prctile(vals, 99);
end

function T = build_summary_table(rn, vals_n, re, vals_e)
sn = summary_stats(vals_n);
se = summary_stats(vals_e);
Subject = {rn.name; re.name};
Group = {'NORMAL'; 'ELDERLY'};
Method = {getfield_or(rn, 'qsm_method', 'WH-QSM'); getfield_or(re, 'qsm_method', 'WH-QSM')};
B0_T = [getfield_or(rn, 'B0', NaN); getfield_or(re, 'B0', NaN)];
DeltaTE_ms = [getfield_or(rn, 'delta_TE_sec', NaN); getfield_or(re, 'delta_TE_sec', NaN)] * 1000;
PhaseFit = {getfield_or(rn, 'phase_fit_method', ''); getfield_or(re, 'phase_fit_method', '')};
Nvox = [sn.n; se.n];
Mean = [sn.mean; se.mean];
Std = [sn.std; se.std];
Median = [sn.median; se.median];
Min = [sn.min; se.min];
P01 = [sn.p01; se.p01];
P05 = [sn.p05; se.p05];
P95 = [sn.p95; se.p95];
P99 = [sn.p99; se.p99];
Max = [sn.max; se.max];
T = table(Subject, Group, Method, B0_T, DeltaTE_ms, PhaseFit, Nvox, ...
    Mean, Std, Median, Min, P01, P05, P95, P99, Max);
end

function v = getfield_or(s, name, default)
if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
    v = s.(name);
else
    v = default;
end
end

function cmap = redblue_cmap_local(n)
if nargin < 1, n = 256; end
n1 = floor(n/2);
n2 = n - n1;
blue = [0.05 0.18 0.68];
white = [0.96 0.96 0.96];
red = [0.72 0.05 0.05];
a = linspace(0, 1, n1)';
b = linspace(0, 1, n2)';
cmap = [blue + (white-blue).*a; white + (red-white).*b];
cmap = max(min(cmap, 1), 0);
end

function dim = get_display_dim_from_result(result, planeName, fallback)
fld = ['display_dim_' lower(char(planeName))];
dim = getfield_or(result, fld, fallback);
if ~(isscalar(dim) && isfinite(dim) && dim>=1 && dim<=3)
    dim = fallback;
end
end

function idx0 = center_index_along_dim(mask, dim)
idx = find(mask);
if isempty(idx)
    N = size(mask); idx0 = round(N(dim)/2); return;
end
[sub{1:3}] = ind2sub(size(mask), idx); %#ok<AGROW>
v = sub{dim};
idx0 = round(mean(v));
N = size(mask); idx0 = max(1, min(N(dim), idx0));
end

function [img, msk] = extract_dim_slice_local(vol, mask, dim, idx)
idx = round(idx);
switch dim
    case 1
        img = squeeze(vol(idx,:,:));
        msk = squeeze(mask(idx,:,:));
    case 2
        img = squeeze(vol(:,idx,:));
        msk = squeeze(mask(:,idx,:));
    otherwise
        img = squeeze(vol(:,:,idx));
        msk = squeeze(mask(:,:,idx));
end
end
