function P = whqsm_local_paths()
% whqsm_local_paths.m
% ============================================================================
% 固化真实被试 WH-QSM 一键运行路径。
%
% 你的工程根目录固定为：
%   D:\MRI_PRO\MRILAB_X\20170327_qsm2016_recon_challenge
%
% 一键入口：
%   在工程根目录运行 RUN_REALDATA_WHQSM_ONECLICK
%   或在 MRI_QSM_dicom_adapter 目录运行 RUN_WHQSM_ONECLICK
%
% 如数据或 SEPIA 位置变化，只改本文件，不改 Pipeline 主体代码。
% ============================================================================

% ====== 固化工程根目录：你的代码都在这个文件夹下 ======
P.projectRoot = 'D:\MRI_PRO\MRILAB_X\20170327_qsm2016_recon_challenge';

% adapter / modules / Utils_self 均按工程根目录派生。
P.adapterDir = fullfile(P.projectRoot, 'MRI_QSM_dicom_adapter');
P.modulesDir = fullfile(P.projectRoot, 'modules');
P.utilsDir   = fullfile(P.projectRoot, 'Utils_self');

% ====== 固化真实被试 DICOM 数据根目录 ======
% 预期结构：projectRoot\data_course\<normal_subject> 和 <elderly_subject>
P.dataRoot = fullfile(P.projectRoot, 'data_course');

% ====== 固化 SEPIA 根目录，必须包含 QSMMacroIOWrapper ======
P.sepiaRoot = 'D:\MRI_PRO\MRILAB_X\sepia';

% ====== 固化 MEDI 根目录：优先调用其中的 FSL-BET MATLAB 实现做脑提取 ======
% 若该目录不存在或 BET 不可用，Pipeline 会自动 fallback 到保守的 magnitude mask。
P.mediRoot = 'D:\MRI_PRO\MRILAB_X\MEDI_toolbox-2024.11.26';

% ====== 运行选项 ======
% true  : 保留 SEPIA 中间 NIfTI/header，便于第一次 QC 和排错
% false : 成功后自动清理中间文件，只保留 WH-QSM 结果
P.keepSepiaWork = true;

% ====== Mask / skull suppression 参数 ======
% 优先级：
%   1) 调用成熟工具箱的 brain extraction（MEDI/SEPIA 提供的 FSL-BET MATLAB 实现）
%   2) 如 BET 不可用，fallback 到 magnitude threshold + 边界侵蚀
%
% 可选：'auto' | 'toolbox_bet' | 'threshold_erode'
P.maskMethod = 'auto';

% BET 参数，对应 FSL BET 常用 -f / -g。若 BET 实现不接受这些参数，会自动尝试较短签名。
P.betFractionalThreshold = 0.50;
P.betVerticalGradient = 0.0;

% 对最终 mask 进行轻度 edge peel，减少 QSM 边界环。BET 成功时建议 1~2 mm；
% 若 fallback 到 threshold_erode，可增大到 5~7 mm。
P.maskErodeMm = 1.5;
P.maskThresholdFactor = 0.12;

% ====== Susceptibility source separation / 磁化率分离 ======
% 优先调用成熟工具箱：SNU-LIST chi-separation / 用户提供 batch adapter。
% 如果没有 toolbox，默认只导出标准输入并跳过；如需预览，可打开 exploratory fallback。
P.runSusceptibilitySeparation = true;
P.chiSepRoot = 'D:\MRI_PRO\MRILAB_X\20170327_qsm2016_recon_challenge\Chisep_Toolbox_v1.2.1';
P.chiSepAdapterFunction = 'snu_chisep_v121_adapter';
P.suscepSepMethod = 'auto';
P.allowExploratorySeparationFallback = false;
P.r2starToChiAbsHzPerPpm = 137.0;
% SNU chi-sepnet 输入 local field。推荐使用 WH-QSM forward field 去除大背景项；
% 可选 'measured' 使用 DICOM phase fitting fieldmap_Hz。
P.snuLocalFieldMode = 'forward_from_whqsm';
% 严格按照 SNU Chisep_script.m 默认：resgen=false 使用 sinc 插值到 1 mm；
% HaveR2Prime=false 表示 GRE-only R2* 路线；is_scaling=false 表示使用 R2pnet。
P.snuResgen = false;
P.snuHaveR2Prime = false;
P.snuIsScaling = false;
P.snuScalingFactor = 0.19;
P.snuInterpMethod = 'sinc';
P.snuSincWindowSize = 15;
P.snuSincWindowType = 'hann';
P.snuDr = 114;

% ====== WH-QSM / FANSI 收敛与正则化参数 ======
% 你之前看到 100 次迭代跑满，因此默认提高 maxiter，并收紧 tol。
% 若仍显得过平滑/偏淡，可尝试把 lambda 调小到 3e-4 或 2e-4。
P.whqsmMaxIter = 200;
P.whqsmTol = 1e-5;
P.whqsmLambda = 5e-4;
P.whqsmBeta = 150;
P.whqsmMuh = 5;

% true  : 开始前 restoredefaultpath，最大限度避免旧 path 污染；随后重新 addpath 本库和 SEPIA
% false : 不清空用户已有 path，仅把本库和 SEPIA 放到前面。更温和，通常推荐。
P.resetMatlabPath = false;

% 输出日志目录。留空时自动使用 dataRoot/_qsm_comparison_results/logs
P.logDir = '';

end
