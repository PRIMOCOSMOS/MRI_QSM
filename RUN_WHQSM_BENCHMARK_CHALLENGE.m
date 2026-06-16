function RUN_WHQSM_BENCHMARK_CHALLENGE()
% RUN_WHQSM_BENCHMARK_CHALLENGE.m
% ============================================================================
% 客观找原因(有金标准): 用【你的 WH-QSM 反演】处理 Challenge 的 phs_tissue,
% 与 chi_cosmos(金标准) 算定量指标。这能区分:
%
%   - 你的 WH-QSM 实现【本身】质量如何(对照 COSMOS 真值);
%   - 视觉差异是"算法性质"(单向 QSM vs COSMOS, 本来就不同) 还是
%     "实现/参数问题"(指标差, 可调)。
%
% 关键诊断指标(mod_evaluation 已内置):
%   RMSE/HFEN/SSIM : 与 COSMOS 的整体一致性
%   StdRatio (<1)  : 对比被压缩程度(过度正则化 -> 偏小)
%   Slope    (<1)  : 系统性低估斜率(过度平滑 -> 偏小)
%
% 同时跑 Challenge 自带的 TKD/L2 作横向对照(它们在同数据上的指标是基准线)。
%
% 用法(项目根目录): RUN_WHQSM_BENCHMARK_CHALLENGE
% ============================================================================

clc;
repoRoot = fileparts(mfilename('fullpath'));
addpath(fullfile(repoRoot,'config'),'-begin');
addpath(fullfile(repoRoot,'modules'),'-begin');
addpath(fullfile(repoRoot,'Utils_self'),'-begin');

cfg = pipeline_config();
dataDir = cfg.dataDir;
fprintf('\n========== WH-QSM 金标准基准 (Challenge 数据) ==========\n');
fprintf('数据目录: %s\n', dataDir);

% --- 加载 Challenge 数据 ---
req = {'phs_tissue','chi_cosmos','msk','spatial_res','magn'};
D = struct();
for i=1:numel(req)
    f = dir(fullfile(dataDir, [req{i} '.mat']));
    if isempty(f)
        error('缺少 Challenge 数据: %s.mat (在 %s)', req{i}, dataDir);
    end
    tmp = load(fullfile(dataDir, [req{i} '.mat']));
    fn = fieldnames(tmp);
    D.(req{i}) = tmp.(fn{1});
end
phs_tissue = double(D.phs_tissue);      % ppm 局部场(输入)
chi_cosmos = double(D.chi_cosmos);      % ppm 金标准
Mask = logical(D.msk);
spatial_res = double(D.spatial_res(:).');
N = size(Mask);
phs_tissue(~Mask)=0; chi_cosmos(~Mask)=0;

fprintf('矩阵: [%d %d %d], 体素: [%.3g %.3g %.3g] mm\n', N, spatial_res);
fprintf('phs_tissue(ppm): std=%.4f p99=%.4f\n', std(phs_tissue(Mask)), prctile(phs_tissue(Mask),99));
fprintf('chi_cosmos(ppm): std=%.4f p99=%.4f\n', std(chi_cosmos(Mask)), prctile(chi_cosmos(Mask),99));

% --- 准备给反演的 data 结构 ---
data = struct();
data.Mask = Mask; data.N = N; data.spatial_res = spatial_res;
data.B0 = 3; data.B0_dir = [0 0 1];
data.magn = double(D.magn);
data.chi_cosmos = chi_cosmos; data.chi_33 = chi_cosmos;
data.phs_tissue = phs_tissue;

results = {}; names = {};

% ============ 1) 你的 WH-QSM (委托 inversion_whqsm_stable) ============
if exist('inversion_whqsm_stable','file')==2
    fprintf('\n[1] 跑你的 WH-QSM (inversion_whqsm_stable) on Challenge...\n');
    try
        chi_wh = inversion_whqsm_stable(phs_tissue, data, spatial_res);
        chi_wh = double(squeeze(chi_wh)); chi_wh(~Mask)=0;
        results{end+1}=chi_wh; names{end+1}='WH-QSM(yours)';
        fprintf('    WH-QSM done. std=%.4f p99=%.4f\n', std(chi_wh(Mask)), prctile(chi_wh(Mask),99));
    catch ME
        warning('WH-QSM 失败: %s', ME.message);
    end
else
    warning('找不到 inversion_whqsm_stable.m');
end

% ============ 2) TKD 基准线 (Challenge 标准) ============
fprintf('\n[2] 跑 TKD 基准线...\n');
kernel = create_dipole_kernel(N, spatial_res, [0 0 1]);
thr=0.15; Di=kernel; s=sign(Di); s(s==0)=1; Di(abs(kernel)<thr)=thr*s(abs(kernel)<thr);
chi_tkd = real(ifftn(fftn(phs_tissue)./Di)).*Mask;
results{end+1}=chi_tkd; names{end+1}='TKD(thr0.15)';

% ============ 3) L2 (closed-form) 基准线 ============
fprintf('[3] 跑 L2 闭式基准线...\n');
reg=9e-2; DtD=abs(kernel).^2;
[EtE] = lap_kernel(N, spatial_res);
chi_l2 = real(ifftn(conj(kernel).*fftn(phs_tissue)./(DtD + reg*EtE))).*Mask;
results{end+1}=chi_l2; names{end+1}='L2(closed)';

% --- 组装 4D 并评估 ---
all_results = zeros([N numel(results)]);
for i=1:numel(results), all_results(:,:,:,i)=results{i}; end
cfg.eval.reference = 'chi_cosmos';
cfg.resultDir = fullfile(repoRoot, 'output', 'challenge_benchmark');
if ~exist(cfg.resultDir,'dir'), mkdir(cfg.resultDir); end

fprintf('\n========== 定量指标 (vs COSMOS 金标准) ==========\n');
try
    metrics = mod_evaluation(all_results, names, data, cfg);
    disp(metrics);
catch ME
    warning('mod_evaluation 失败, 用内置简版: %s', ME.message);
    simple_metrics(all_results, names, chi_cosmos, Mask);
end

% --- ROI 苍白球(用 chi_cosmos 高值定位, 公平) ---
fprintf('\n========== 苍白球 ROI (用 COSMOS 高值定位) median 对比 ==========\n');
gp = (chi_cosmos > prctile(chi_cosmos(Mask),98)) & Mask;
fprintf('  COSMOS  苍白球 median=%.4f\n', median(chi_cosmos(gp)));
for i=1:numel(results)
    fprintf('  %-16s median=%.4f (与COSMOS比 %.0f%%)\n', names{i}, ...
        median(results{i}(gp)), median(results{i}(gp))/median(chi_cosmos(gp))*100);
end

fprintf('\n判读:\n');
fprintf('  - WH-QSM 的 RMSE/SSIM 若与 TKD/L2 同级或更好 -> 你的实现正常;\n');
fprintf('  - StdRatio/Slope 接近 1 -> 无过度正则化; 明显<1 -> 对比被压(可调小正则);\n');
fprintf('  - 苍白球 median 与 COSMOS 接近 -> 数值准确, 视觉差异主要是单向QSM固有特性。\n');
fprintf('结果保存: %s\n', cfg.resultDir);
end

%% ----
function EtE = lap_kernel(N, vox)
[k1,k2,k3]=ndgrid(0:N(1)-1,0:N(2)-1,0:N(3)-1);
E1=(1-exp(2i*pi*k1/N(1)))/vox(1);
E2=(1-exp(2i*pi*k2/N(2)))/vox(2);
E3=(1-exp(2i*pi*k3/N(3)))/vox(3);
EtE=abs(E1).^2+abs(E2).^2+abs(E3).^2;
end

function simple_metrics(all_results, names, ref, Mask)
r=ref(Mask);
for i=1:size(all_results,4)
    p=all_results(:,:,:,i); p=p(Mask);
    rmse=sqrt(mean((p-r).^2))/sqrt(mean(r.^2))*100;
    c=corrcoef(p,r); corr=c(1,2);
    fprintf('  %-16s RMSE=%.1f%% CORR=%.3f StdRatio=%.3f\n', names{i}, rmse, corr, std(p)/std(r));
end
end
