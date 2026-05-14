function metrics = mod_evaluation(all_results, all_names, data, cfg)
% mod_evaluation.m
% QSM 定量评估模块
%
% 评估指标 (对标 QSM2016 Challenge 官方):
%   - RMSE: Root Mean Square Error
%   - HFEN: High Frequency Error Norm (Laplacian of Gaussian)
%   - SSIM: Structural Similarity Index
%   - WM/GM Error: 白质/灰质区域平均绝对误差
%   - Correlation: Pearson 相关系数
%
% 额外诊断:
%   - StdRatio: 重建标准差 / 参考标准差
%   - SlopePredVsRef: 线性回归斜率 (< 1 表示过度正则化)
%   - FieldResidualNRMSE: 前向场一致性检验

%% 选择参考
if strcmpi(cfg.eval.reference, 'chi_cosmos')
    chi_ref = data.chi_cosmos;
    ref_name = 'chi_cosmos';
else
    chi_ref = data.chi_33;
    ref_name = 'chi_33';
end

Mask = data.Mask;
N = data.N;
spatial_res = data.spatial_res;
nAlg = size(all_results, 4);

fprintf('评估参考: %s\n', ref_name);
fprintf('算法数量: %d\n', nAlg);

%% 加载评估 mask (区分 WM/GM)
eval_mask = double(data.evaluation_mask);
wm_mask = eval_mask > 6;
gm_mask = eval_mask > 0 & eval_mask <= 6;
roi_mask = eval_mask > 0;

wm_n = max(sum(wm_mask(:)), 1);
gm_n = max(sum(gm_mask(:)), 1);
roi_n = max(sum(roi_mask(:)), 1);

%% 初始化指标数组
Algorithm   = all_names(:);
RMSE        = zeros(nAlg, 1);
HFEN        = zeros(nAlg, 1);
SSIM_val    = zeros(nAlg, 1);
WM_ERROR    = zeros(nAlg, 1);
GM_ERROR    = zeros(nAlg, 1);
WGM_ERROR   = zeros(nAlg, 1);
CORR        = zeros(nAlg, 1);
StdRatio    = zeros(nAlg, 1);
Slope       = zeros(nAlg, 1);
FieldNRMSE  = zeros(nAlg, 1);

%% LoG 核 (用于 HFEN)
log_kernel = create_log_kernel_3d(2.0);

%% 偶极子核 (用于前向场一致性)
kernel = create_dipole_kernel(N, spatial_res);
RDF_ppm = data.phs_tissue;
RDF_norm = norm(RDF_ppm(Mask));

%% 参考统计
ref_roi = chi_ref(roi_mask);
ref_std = std(ref_roi);
ref_centered = ref_roi - mean(ref_roi);

%% 打印表头
fprintf('%20s | %8s | %8s | %8s | %8s | %8s | %8s | %8s | %8s\n', ...
    'Algorithm', 'RMSE', 'HFEN', 'SSIM', 'WM_ERR', 'GM_ERR', 'CORR', 'StdR', 'Slope');
fprintf('%s\n', repmat('-', 1, 110));

%% 逐算法计算
for c = 1:nAlg
    vol = double(all_results(:,:,:,c));
    
    % --- RMSE ---
    d = vol(roi_mask) - chi_ref(roi_mask);
    RMSE(c) = sqrt(mean(d.^2));
    
    % --- HFEN ---
    HFEN(c) = compute_hfen_internal(vol, chi_ref, roi_mask, log_kernel);
    
    % --- SSIM ---
    SSIM_val(c) = compute_ssim_internal(vol, chi_ref, roi_mask);
    
    % --- WM / GM Error ---
    WM_ERROR(c) = sum(abs(vol(wm_mask) - chi_ref(wm_mask))) / wm_n;
    GM_ERROR(c) = sum(abs(vol(gm_mask) - chi_ref(gm_mask))) / gm_n;
    WGM_ERROR(c) = 0.5 * (WM_ERROR(c) + GM_ERROR(c));
    
    % --- Correlation ---
    pred_roi = vol(roi_mask);
    C = corrcoef(pred_roi, ref_roi);
    if size(C, 1) >= 2 && size(C, 2) >= 2
        CORR(c) = C(1,2);
    else
        CORR(c) = NaN;
    end
    
    % --- StdRatio ---
    StdRatio(c) = std(pred_roi) / max(ref_std, eps);
    
    % --- Slope (pred vs ref) ---
    pred_centered = pred_roi - mean(pred_roi);
    Slope(c) = dot(pred_centered, ref_centered) / max(dot(ref_centered, ref_centered), eps);
    
    % --- Forward field consistency ---
    field_pred = real(ifftn(kernel .* fftn(vol)));
    field_pred(~Mask) = 0;
    residual = field_pred - RDF_ppm;
    FieldNRMSE(c) = norm(residual(Mask)) / max(RDF_norm, eps);
    
    % 打印
    fprintf('%20s | %8.5f | %8.4f | %8.4f | %8.5f | %8.5f | %8.4f | %8.3f | %8.3f\n', ...
        all_names{c}, RMSE(c), HFEN(c), SSIM_val(c), ...
        WM_ERROR(c), GM_ERROR(c), CORR(c), StdRatio(c), Slope(c));
end

fprintf('%s\n', repmat('-', 1, 110));

%% 构建输出表
metrics = table(Algorithm, RMSE, HFEN, SSIM_val, WM_ERROR, GM_ERROR, WGM_ERROR, ...
    CORR, StdRatio, Slope, FieldNRMSE, ...
    'VariableNames', {'Algorithm', 'RMSE', 'HFEN', 'SSIM', ...
    'WM_ERROR', 'GM_ERROR', 'WGM_ERROR', 'Correlation', ...
    'StdRatio', 'SlopePredVsRef', 'FieldResidualNRMSE'});

%% 保存
save(fullfile(cfg.resultDir, 'evaluation_metrics.mat'), 'metrics');

try
    writetable(metrics, fullfile(cfg.resultDir, 'evaluation_metrics.csv'));
    fprintf('指标已保存: evaluation_metrics.csv\n');
catch ME
    warning('CSV 写入失败: %s', ME.message);
end

%% 打印最佳结果
[~, best_idx] = min(RMSE);
fprintf('最佳 RMSE: %s (%.6f)\n', all_names{best_idx}, RMSE(best_idx));

[~, best_ssim_idx] = max(SSIM_val);
fprintf('最佳 SSIM: %s (%.4f)\n', all_names{best_ssim_idx}, SSIM_val(best_ssim_idx));

%% 诊断提示
fprintf('诊断提示:\n');
fprintf('  StdRatio << 1 且 Slope << 1: 过度正则化 (结果偏淡)\n');
fprintf('  FieldNRMSE 很大: 前向场不一致，可能存在单位/核不匹配\n');
fprintf('  Correlation < 0: 可能存在符号翻转\n');

end

%% =========================================================================
% HFEN 计算
% =========================================================================
function hfen = compute_hfen_internal(vol, ref, roi_mask, log_kernel)
% High Frequency Error Norm
% HFEN = ||LoG(vol) - LoG(ref)||_2 / ||LoG(ref)||_2
% LoG: Laplacian of Gaussian

log_vol = convn(vol, log_kernel, 'same');
log_ref = convn(ref, log_kernel, 'same');

diff_log = log_vol(roi_mask) - log_ref(roi_mask);
ref_log_norm = norm(log_ref(roi_mask));

if ref_log_norm > eps
    hfen = norm(diff_log) / ref_log_norm;
else
    hfen = NaN;
end

end

%% =========================================================================
% SSIM 计算
% =========================================================================
function ssim_val = compute_ssim_internal(vol, ref, roi_mask)
% 3D SSIM (简化版，基于全局统计)
% 完整 SSIM 应使用滑动窗口，这里用 ROI 内全局近似

x = vol(roi_mask);
y = ref(roi_mask);

mu_x = mean(x);
mu_y = mean(y);
sigma_x = std(x);
sigma_y = std(y);
sigma_xy = mean((x - mu_x) .* (y - mu_y));

% 动态范围
L = max(y) - min(y);
C1 = (0.01 * L)^2;
C2 = (0.03 * L)^2;

ssim_val = ((2*mu_x*mu_y + C1) * (2*sigma_xy + C2)) / ...
    ((mu_x^2 + mu_y^2 + C1) * (sigma_x^2 + sigma_y^2 + C2));

end

%% =========================================================================
% LoG 核
% =========================================================================
function h = create_log_kernel_3d(sigma)
% 3D Laplacian of Gaussian 核

half_size = ceil(3 * sigma);
[x, y, z] = ndgrid(-half_size:half_size, -half_size:half_size, -half_size:half_size);
r2 = x.^2 + y.^2 + z.^2;
s2 = sigma^2;

% LoG = (r² - 3σ²) / σ⁶ * exp(-r²/(2σ²))
h = (r2 - 3*s2) / (s2^3) .* exp(-r2 / (2*s2));

% 零均值化
h = h - mean(h(:));

end