function cfg = pipeline_config()
% pipeline_config.m
% 统一管理所有路径和参数配置

%% ========================================================================
% 路径配置
% ========================================================================
cfg.rootDir   = 'D:\MRI_PRO\MRILAB_X\20170327_qsm2016_recon_challenge';
cfg.dataDir   = fullfile(cfg.rootDir, 'data');
cfg.outDir    = fullfile(cfg.rootDir, 'output_modular_pipeline');
cfg.figDir    = fullfile(cfg.outDir, 'figures');
cfg.resultDir = fullfile(cfg.outDir, 'results');

% 工具箱路径
cfg.mediRoot  = 'D:\MRI_PRO\MRILAB_X\MEDI_toolbox-2024.11.26';
cfg.sepiaRoot = 'D:\MRI_PRO\MRILAB_X\sepia';

% 深度学习模型路径（与你仓库目录大小写一致）
cfg.dlModelDir = fullfile(cfg.rootDir, 'Models');

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
cfg.bgRemoval.methods = {'VSHARP', 'PDF', 'LBV'};
cfg.bgRemoval.vsharp_radius = 1:1:12;
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

cfg.inversion.medi_lambdas = [0.01 0.03 0.06 0.1 0.3 1 3];
cfg.inversion.medi_use_structural = true;

%% ========================================================================
% 深度学习参数（xQSM + Python bridge + .pth）
% ========================================================================
cfg.deeplearning.enable = true;
cfg.deeplearning.models = {'xQSM'};  % 只跑 xQSM

% xQSM checkpoint (.pth)
cfg.deeplearning.xqsm_pth = fullfile(cfg.dlModelDir, 'xQSM_invivo.pth');

% xQSM 仓库根目录（需包含 python/xQSM.py）
cfg.deeplearning.xqsm_repo_root = 'D:\MRI_PRO\MRILAB_X\xQSM';

% Python 可执行文件（为空时自动尝试 pyenv，再 fallback 到 "python"）
cfg.deeplearning.python_executable = 'D:\Anaconda3\python.exe';

% 推理设备: 'auto' | 'cpu' | 'cuda'
cfg.deeplearning.xqsm_device = 'auto';

% 独立 bridge 脚本路径（推荐放在项目内）
cfg.deeplearning.xqsm_bridge_script = fullfile( ...
    cfg.rootDir, 'modules', 'DL', 'python', 'infer_xqsm_from_mat.py');

% 兼容保留字段（当前不使用 ONNX）
cfg.deeplearning.qsmnet_onnx = fullfile(cfg.dlModelDir, 'QSMnet_plus.onnx');
cfg.deeplearning.xqsm_onnx   = fullfile(cfg.dlModelDir, 'xQSM.onnx');

%% ========================================================================
% 可视化参数
% ========================================================================
cfg.vis.clim_qsm = [-0.12 0.12];
cfg.vis.clim_err = [-0.06 0.06];
cfg.vis.doSave = true;
cfg.vis.resolution = 200;

%% ========================================================================
% 评估参数
% ========================================================================
cfg.eval.reference = 'chi_cosmos';

end