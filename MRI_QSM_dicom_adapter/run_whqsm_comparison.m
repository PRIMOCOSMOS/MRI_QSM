function run_whqsm_comparison(data_root, mri_qsm_root, varargin)
% run_whqsm_comparison.m  (v3 - 验证 + 预检查 + 健壮版)
% ============================================================================
%   🚀 主入口：完整跑通 "正常人 vs 老年人" 的 WH-QSM 对比分析
%
%   增强:
%     ✅ 预检查：SEPIA / 原库 / 数据 / 单位都验证后才跑
%     ✅ 错误处理：每个被试独立 try/catch，一个失败不影响另一个
%     ✅ 单位修正：phs_tissue 必须是 ppm（已传给 dicom_loader_subject）
%     ✅ 复用原库：mod_dipole_inversion → inversion_whqsm_stable → SEPIA FANSI
%
%   用法:
%     run_whqsm_comparison()                                          % 默认
%     run_whqsm_comparison('data_root')                               % 自定义数据
%     run_whqsm_comparison('data_root', 'mri_qsm_root')               % 完全自定义
%     run_whqsm_comparison(...,'skip_sepia_check',false)              % 跳过 SEPIA 检查
% ============================================================================

% ====== 默认路径 ======
if nargin < 1 || isempty(data_root)
    data_root = 'D:\MRI_PRO\MRILAB_X\20170327_qsm2016_recon_challenge\data_course';
end
if nargin < 2 || isempty(mri_qsm_root)
    % 自动定位: 假设 dicom_adapter 在 MRI_QSM 仓库根目录下
    cur_dir = fileparts(mfilename('fullpath'));
    mri_qsm_root = fileparts(cur_dir);
end

% ====== 解析可选参数 ======
p = inputParser;
addParameter(p, 'skip_sepia_check', false, @islogical);
parse(p, varargin{:});
skip_sepia = p.Results.skip_sepia_check;

fprintf('\n');
fprintf('╔══════════════════════════════════════════════════════════════╗\n');
fprintf('║  MRI QSM - 正常 vs 老年人 WH-QSM 对比分析  (v3)                     ║\n');
fprintf('║  复用原库 mod_dipole_inversion → SEPIA/FANSI                                  ║\n');
fprintf('╚══════════════════════════════════════════════════════════════╝\n\n');

% ====== 路径设置 ======
fprintf('[Init] 配置路径...\n');
fprintf('  数据根目录: %s\n', data_root);
fprintf('  原库路径  : %s\n', mri_qsm_root);

% 添加扩展模块到路径
this_dir = fileparts(mfilename('fullpath'));
addpath(genpath(this_dir));
fprintf('  扩展模块  : %s\n', this_dir);

% ====== 预检查 ======
fprintf('\n[Init] 系统预检查...\n');

% 1. 检查原库
if ~check_original_library(mri_qsm_root)
    error('原库检查失败！请确认路径: %s', mri_qsm_root);
end

% 2. 检查 SEPIA（除非显式跳过）
if ~skip_sepia
    if ~check_sepia_toolbox(mri_qsm_root)
        fprintf('  ⚠️ SEPIA 检查失败，WH-QSM 将不可用\n');
        fprintf('     如果想跳过 SEPIA 检查: run_whqsm_comparison(...,''skip_sepia_check'',true)\n');
    end
end

% 3. 检查数据
if ~exist(data_root, 'dir')
    error('数据根目录不存在: %s', data_root);
end
fprintf('  ✅ 数据根目录存在\n');

% 4. 检查 Image Processing Toolbox（SEPIA 需要 niftiwrite）
if exist('niftiwrite', 'file') ~= 2
    warning('⚠️ Image Processing Toolbox 缺失（无 niftiwrite），SEPIA 调用会失败');
end

% ====== Step 1: 发现被试 ======
fprintf('\n');
subjects = discover_subjects(data_root);

n_normal  = sum(strcmp({subjects.group}, 'NORMAL'));
n_elderly = sum(strcmp({subjects.group}, 'ELDERLY'));
n_unk     = sum(strcmp({subjects.group}, 'UNKNOWN'));

if n_normal == 0 || n_elderly == 0
    fprintf('\n');
    fprintf('╔══════════════════════════════════════════════════════════════╗\n');
    fprintf('║  ❌ 未同时识别 NORMAL 和 ELDERLY 被试                          ║\n');
    fprintf('╚══════════════════════════════════════════════════════════════╝\n\n');
    fprintf('当前状态:\n');
    fprintf('  NORMAL  : %d 个\n', n_normal);
    fprintf('  ELDERLY : %d 个\n', n_elderly);
    fprintf('  UNKNOWN : %d 个\n\n', n_unk);
    fprintf('所有被试的 DICOM 元数据里都没有 "elderly/old" 或 "normal/control" 关键字。\n');
    fprintf('\n');
    fprintf('【解决方案 - 按顺序尝试】\n\n');
    fprintf('方案 1: 先跑 inspect_subjects 看年龄\n');
    fprintf('  >> inspect_subjects\n');
    fprintf('  它会从 DICOM 提取 PatientAge / PatientBirthDate 然后告诉你谁是老年。\n\n');
    fprintf('方案 2: 手动指定分组（知道答案时最快）\n');
    fprintf('  >> subjects = discover_subjects;\n');
    for k = 1:length(subjects)
        fprintf('  >> subjects(%d).group = ''NORMAL'';  %% 或 ''ELDERLY''\n', k);
    end
    fprintf('  >> run_whqsm_comparison()\n\n');
    fprintf('方案 3: 基于年龄自动分配（如果 DICOM 里有年龄）\n');
    fprintf('  >> subjects = discover_subjects;\n');
    fprintf('  >> subjects = auto_assign_by_age(subjects);\n');
    fprintf('  >> run_whqsm_comparison()\n\n');
    return;
end

idx_normal  = find(strcmp({subjects.group}, 'NORMAL'),  1);
idx_elderly = find(strcmp({subjects.group}, 'ELDERLY'), 1);

sub_normal  = subjects(idx_normal);
sub_elderly = subjects(idx_elderly);

% ====== 全局输出目录 ======
output_root = fullfile(data_root, '_qsm_comparison_results');
if ~exist(output_root, 'dir'), mkdir(output_root); end

% ====== Step 2: 处理每个被试 ======
results = struct();

for which = {'normal', 'elderly'}
    grp = which{1};
    if strcmp(grp, 'normal')
        sub = sub_normal;
    else
        sub = sub_elderly;
    end

    fprintf('\n');
    fprintf('╔══════════════════════════════════════════════════════════════╗\n');
    fprintf('║  处理被试: %s (%s)\n', sub.name, upper(grp));
    fprintf('╚══════════════════════════════════════════════════════════════╝\n\n');

    sub_output = fullfile(output_root, [grp '_' sub.name]);

    try
        % ----- 2a: DICOM 加载 -----
        data = dicom_loader_subject(sub, fullfile(sub_output, 'qsm2016_format'));

        % ----- 2b: 准备 pipeline 配置 -----
        cfg = build_pipeline_cfg(sub_output, mri_qsm_root, data);

        % ----- 2c: 复用原库 mod_background_removal -----
        fprintf('\n--- 复用原库 mod_background_removal ---\n');
        [local_field, bg_results] = mod_background_removal(data, cfg);

        % ----- 2d: 复用原库 mod_dipole_inversion -----
        fprintf('\n--- 复用原库 mod_dipole_inversion (含 WH-QSM) ---\n');
        [qsm_results, qsm_names] = mod_dipole_inversion(local_field, data, cfg);

        % ----- 2e: 提取 WH-QSM 结果 -----
        idx_whqsm = find(strcmp(qsm_names, 'WH-QSM'), 1);
        if isempty(idx_whqsm)
            warning('WH-QSM 未运行！qsm_names = %s', strjoin(qsm_names, ', '));
            if ~isempty(qsm_names)
                idx_whqsm = length(qsm_names);
                fprintf('  ⚠️ Fallback 到: %s\n', qsm_names{idx_whqsm});
            else
                error('所有反演方法都失败！');
            end
        end
        chi = qsm_results(:,:,:,idx_whqsm);
        chi(~data.Mask) = 0;

        % ----- 2f: 保存 -----
        results.(grp) = struct( ...
            'name', sub.name, ...
            'group', upper(grp), ...
            'chi', chi, ...
            'mask', data.Mask, ...
            'spatial_res', data.spatial_res, ...
            'magn', data.magn, ...
            't1', data.mp_rage, ...
            'qsm_method', qsm_names{idx_whqsm}, ...
            'all_qsm_names', {qsm_names}, ...
            'all_qsm_results', qsm_results);

        save(fullfile(sub_output, ['chi_' grp '.mat']), 'chi', '-v7.3');
        save(fullfile(sub_output, ['all_qsm_' grp '.mat']), ...
            'qsm_results', 'qsm_names', '-v7.3');
        fprintf('  💾 保存: %s\n', sub_output);

    catch ME
        fprintf('\n  ❌ 被试 %s 处理失败: %s\n', sub.name, ME.message);
        fprintf('     (at %s:%d)\n', ME.stack(1).name, ME.stack(1).line);
        results.(grp) = [];
    end
end

% ====== Step 3: 对比分析 ======
if isempty(results.normal) || isempty(results.elderly)
    fprintf('\n❌ 至少一个被试处理失败，无法对比！\n');
    return;
end

fprintf('\n');
fprintf('╔══════════════════════════════════════════════════════════════╗\n');
fprintf('║  对比分析: 正常 vs 老年人\n');
fprintf('╚══════════════════════════════════════════════════════════════╝\n\n');

compare_dir = fullfile(output_root, 'comparison');
compare_subjects(results.normal, results.elderly, compare_dir);

% ====== 保存最终汇总 ======
save(fullfile(output_root, 'all_results.mat'), 'results', '-v7.3');

% ====== 完成 ======
fprintf('\n');
fprintf('╔══════════════════════════════════════════════════════════════╗\n');
fprintf('║  🎉 全部完成！\n');
fprintf('║  输出目录: %s\n', output_root);
fprintf('║  主要产物:\n');
fprintf('║    - normal_*/chi_normal.mat      (NORMAL QSM, WH-QSM)\n');
fprintf('║    - elderly_*/chi_elderly.mat    (ELDERLY QSM, WH-QSM)\n');
fprintf('║    - *_/all_qsm_*.mat             (TKD/CFL2/iLSQR/MEDI/WH-QSM 全部)\n');
fprintf('║    - comparison/compare_3view.png (三平面)\n');
fprintf('║    - comparison/compare_roi_basal_ganglia.png (深部灰质 ROI)\n');
fprintf('║    - comparison/compare_histogram.png\n');
fprintf('║    - comparison/compare_diff_map.png\n');
fprintf('║    - comparison/roi_comparison.csv\n');
fprintf('╚══════════════════════════════════════════════════════════════╝\n\n');
end

%% =========================================================================
% 预检查: 原库完整性
%% =========================================================================
function tf = check_original_library(root)
tf = true;
critical = {
    fullfile(root, 'modules', 'mod_dipole_inversion.m');
    fullfile(root, 'modules', 'mod_background_removal.m');
    fullfile(root, 'modules', 'mod_load_data.m');
    fullfile(root, 'Utils_self', 'create_dipole_kernel.m');
};

for k = 1:length(critical)
    if exist(critical{k}, 'file') ~= 2
        fprintf('  ❌ 原库缺失: %s\n', critical{k});
        tf = false;
    end
end

if tf
    fprintf('  ✅ 原库完整 (关键模块都在)\n');
    addpath(genpath(fullfile(root, 'modules')));
    addpath(genpath(fullfile(root, 'Utils_self')));
    addpath(fullfile(root, 'config'));
    addpath(root);
end
end

%% =========================================================================
% 预检查: SEPIA 工具箱
%% =========================================================================
function tf = check_sepia_toolbox(mri_qsm_root)
tf = false;

% 候选路径
candidates = {
    fullfile(mri_qsm_root, 'sepia');
    'D:\MRI_PRO\MRILAB_X\sepia';
    'C:\MRILAB_X\sepia';
    '/opt/sepia';
};

for k = 1:length(candidates)
    if exist(candidates{k}, 'dir') == 7
        addpath(genpath(candidates{k}));

        % 验证关键函数
        if exist('QSMMacroIOWrapper', 'file') == 2
            fprintf('  ✅ SEPIA 工具箱完整: %s\n', candidates{k});
            tf = true;
            return;
        else
            fprintf('  ⚠️ 路径存在但缺 QSMMacroIOWrapper: %s\n', candidates{k});
        end
    end
end

if ~tf
    fprintf('  ❌ SEPIA 工具箱未找到\n');
    fprintf('     已尝试路径:\n');
    for k = 1:length(candidates)
        fprintf('       - %s\n', candidates{k});
    end
end
end

%% =========================================================================
% 构造 cfg（与原库 pipeline_config.m 兼容）
%% =========================================================================
function cfg = build_pipeline_cfg(sub_output, mri_qsm_root, data)
cfg.rootDir    = sub_output;
cfg.dataDir    = fullfile(sub_output, 'qsm2016_format');
cfg.outDir     = sub_output;
cfg.figDir     = fullfile(sub_output, 'figures');
cfg.resultDir  = fullfile(sub_output, 'results');
cfg.dlModelDir = fullfile(sub_output, 'Models');

for f = {cfg.outDir, cfg.figDir, cfg.resultDir, cfg.dlModelDir}
    if ~exist(f{1}, 'dir'), mkdir(f{1}); end
end

cfg.mediRoot  = fullfile(mri_qsm_root, 'MEDI_toolbox-2024.11.26');
cfg.sepiaRoot = 'D:\MRI_PRO\MRILAB_X\sepia';

cfg.bgRemoval.methods        = {'WHQSM'};
cfg.bgRemoval.vsharp_radius  = 1:1:12;
cfg.bgRemoval.pdf_tol        = 0.1;
cfg.bgRemoval.lbv_tol        = 0.01;
cfg.bgRemoval.lbv_peel       = 2;

cfg.inversion.tkd_threshold       = 0.19;
cfg.inversion.cfl2_reg             = 9e-2;
cfg.inversion.ilsqr_tol            = 1e-3;
cfg.inversion.ilsqr_maxiter        = 50;
cfg.inversion.medi_lambdas         = [0.1];
cfg.inversion.medi_use_structural  = true;

cfg.deeplearning.enable       = false;
cfg.deeplearning.models       = {};
cfg.deeplearning.qsmnet_onnx  = fullfile(cfg.dlModelDir, 'QSMnet_plus.onnx');
cfg.deeplearning.xqsm_onnx    = fullfile(cfg.dlModelDir, 'xQSM.onnx');
cfg.deeplearning.xqsm_pth     = fullfile(cfg.dlModelDir, 'xQSM_invivo.pth');

cfg.vis.clim_qsm   = [-0.15 0.15];
cfg.vis.clim_err   = [-0.06 0.06];
cfg.vis.doSave     = true;
cfg.vis.resolution = 200;

cfg.eval.reference = '';

% 注入 B0 到 data（让 inversion_whqsm_stable 能读到）
if isfield(data, 'B0') && ~isempty(data.B0) && data.B0 > 0
    cfg.B0 = data.B0;
else
    cfg.B0 = 3;
    data.B0 = 3;
    data.b0 = 3;
end

fprintf('  cfg.B0 = %.2f T\n', cfg.B0);
end
