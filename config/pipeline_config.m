function cfg = pipeline_config()
% pipeline_config.m
% 统一管理所有路径和参数配置
%
% 根据 D:\MRI_PRO\MRILAB_X\ 目录结构:
%   20170327_qsm2016_recon_challenge/  — 竞赛数据与代码
%   MEDI_toolbox-2024.11.26/           — MEDI 工具箱
%   sepia/                             — SEPIA 工具箱
%   qsm_challenge_2016/               — 不使用 (非竞赛数据)

%% ========================================================================
% 路径配置
% ========================================================================
cfg.rootDir   = 'D:\MRI_PRO\MRILAB_X\20170327_qsm2016_recon_challenge';
cfg.dataDir   = fullfile(cfg.rootDir, 'data');
cfg.outDir    = fullfile(cfg.rootDir, 'output_modular_pipeline');
cfg.figDir    = fullfile(cfg.outDir, 'figures');
cfg.resultDir = fullfile(cfg.outDir, 'results');

% 工具箱路径 (根据截图修正)
cfg.mediRoot  = 'D:\MRI_PRO\MRILAB_X\MEDI_toolbox-2024.11.26';
cfg.sepiaRoot = 'D:\MRI_PRO\MRILAB_X\sepia';

% 深度学习模型路径
cfg.dlModelDir = fullfile(cfg.rootDir, 'models');

%% ========================================================================
% 创建输出目录
% ========================================================================
dirs_to_create = {cfg.outDir, cfg.figDir, cfg.resultDir, cfg.dlModelDir};
for i = 1:numel(dirs_to_create)
    if ~exist(dirs_to_create{i}, 'dir')
        mkdir(dirs_to_create{i});
    end
end

%% ========================================================================
% 背景场去除参数
% ========================================================================
% 可选方法: 'VSHARP', 'PDF', 'LBV', 'WHQSM'
cfg.bgRemoval.methods = {'VSHARP', 'PDF', 'LBV'};
cfg.bgRemoval.vsharp_radius = 1:1:12;   % V-SHARP 多尺度半径 (mm)
cfg.bgRemoval.pdf_tol = 0.1;
cfg.bgRemoval.lbv_tol = 0.01;
cfg.bgRemoval.lbv_peel = 2;

%% ========================================================================
% 偶极子反演参数
% ========================================================================
cfg.inversion.tkd_threshold = 0.19;
cfg.inversion.cfl2_reg = 9e-2;
cfg.inversion.ilsqr_tol = 1e-3;
cfg.inversion.ilsqr_maxiter = 50;

% MEDI 参数
% QSM2016 phs_tissue 已经是 ppm 单位的局部场
% lambda 不应使用 500/1000 等大值 (那是针对 Hz 单位的)
cfg.inversion.medi_lambdas = [0.01 0.03 0.06 0.1 0.3 1 3];
cfg.inversion.medi_use_structural = true;  % 使用 mp_rage 作为 MEDI 结构先验

%% ========================================================================
% 深度学习参数
% ========================================================================
cfg.deeplearning.enable = true;
cfg.deeplearning.models = {'QSMnet_plus', 'xQSM', 'MATLAB_UNet3D'};

% 预训练 ONNX 模型文件
cfg.deeplearning.qsmnet_onnx = fullfile(cfg.dlModelDir, 'QSMnet_plus.onnx');
cfg.deeplearning.xqsm_onnx   = fullfile(cfg.dlModelDir, 'xQSM.onnx');

% MATLAB 原生 U-Net 参数
cfg.deeplearning.unet_patch_size = [32 32 32];
cfg.deeplearning.unet_n_patches = 300;
cfg.deeplearning.unet_epochs = 30;
cfg.deeplearning.unet_batch = 4;
cfg.deeplearning.unet_lr = 1e-3;
cfg.deeplearning.unet_base_filters = 16;

%% ========================================================================
% 可视化参数
% ========================================================================
cfg.vis.clim_qsm = [-0.12 0.12];   % ppm, 零中心
cfg.vis.clim_err = [-0.06 0.06];    % ppm 误差
cfg.vis.doSave = true;
cfg.vis.resolution = 200;            % DPI

%% ========================================================================
% 评估参数
% ========================================================================
cfg.eval.reference = 'chi_cosmos';       % 'chi_33' 或 'chi_cosmos'

end