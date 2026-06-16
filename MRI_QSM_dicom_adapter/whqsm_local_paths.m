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

% ------------------------------------------------------------------------
% chi-separation 后端选择
% ------------------------------------------------------------------------
% MATLAB 的 importNetworkFromONNX / importONNXNetwork 依赖 onnxmex.mexw64，
% 在部分 Windows/MATLAB 安装上会报 "onnxmex.mexw64 无效：找不到指定的程序"。
% 这是 VC++/支持包依赖损坏，不是模型问题（SNU-LIST issue #5 同样问题）。
%
% 解决：把 useOnnxRuntimeChiSep 设为 true，改用 Python(onnxruntime) 直接加载
% 同一批 .onnx 模型推理，完全绕过 MATLAB 的 ONNX 导入。结果与官方一致。
%   true  : 使用 snu_chisep_onnxruntime_adapter（推荐，绕过 onnxmex）
%   false : 使用 snu_chisep_v121_adapter（调用 SNU p-code，会触发 onnxmex）
P.useOnnxRuntimeChiSep = true;

if P.useOnnxRuntimeChiSep
    P.chiSepAdapterFunction = 'snu_chisep_onnxruntime_adapter';
else
    P.chiSepAdapterFunction = 'snu_chisep_v121_adapter';
end

% ----- ONNX Runtime 桥接所需配置 -----
% Python 可执行文件：必须安装 numpy / scipy / onnxruntime（有 GPU 用 onnxruntime-gpu）
P.onnxPythonExecutable = 'D:\Anaconda3\python.exe';
% 桥接脚本（项目内，无需修改）
P.onnxBridgeScript = fullfile(P.modulesDir, 'DL', 'python', 'infer_chisep_from_mat.py');
% .onnx 模型与归一化文件。留空则自动在 <chiSepRoot>\models 下查找常见文件名。
% 已按你 models\ 目录实际文件名填死（截图 2026-06-15）：
P.onnxQsmModel     = fullfile(P.chiSepRoot, 'models', '240904_QSMnet.onnx');
P.onnxXsepModel    = fullfile(P.chiSepRoot, 'models', '240904_xsepnet.onnx');     % χ-sepnet 本体(3入2出)
P.onnxR2primeModel = fullfile(P.chiSepRoot, 'models', '240531_R2PRIMEnet.onnx');  % 3T；7T 用 R2PNET_7T.onnx
P.onnxNormFactor   = fullfile(P.chiSepRoot, 'models', 'norm_factor.mat');
% 流程: 'auto'（有 R2 用 r2'，否则 r2*）| 'r2p' | 'r2s'
P.onnxPipeline     = 'auto';
% χ-sepnet 的 QSM 输入来源（统一 QSM 来源开关）：
%   'qsmnet'   : 用工具箱自带 QSMnet 重建 QSM（官方默认）
%   'external' : 用你主流程的 WH-QSM 结果(chi_total)作为 QSM 输入，
%                两条线 QSM 来源一致；会用 cosmos_sus_mean/std 归一化到网络输入空间。
% 注意：选 'external' 建议把 onnxLocalFieldMode 设为 'forward_from_whqsm'（默认），
%       使 local field 与该 QSM 自洽。
P.onnxQsmSource    = 'qsmnet';
% local field 单位: 'Hz' | 'radian' | 'ppm'（forward_from_whqsm 时内部固定 Hz）
P.onnxFieldUnit    = 'Hz';
% 推理设备: 'auto' | 'cpu' | 'cuda'
P.onnxDevice       = 'auto';
% 分辨率泛化(官方 resolution-generalization, Ji 2023): 数据非 1mm 时先 k-space
% 重采样到 1mm 等效矩阵再喂固定尺寸网络，推理后还原。
%   'auto' : 体素≠1mm 时自动启用(推荐)
%   'on'   : 强制启用   'off' : 关闭(仅当数据已是 1mm 且矩阵≤网络尺寸)
P.onnxResgen       = 'auto';
% 可选: R2 map（来自 SE），用于 r2' 流程。可为 [] / 变量 / .mat 文件路径。
P.onnxR2Map        = [];

% ------------------------------------------------------------------------
% 方法对比 / 传统优化 χ-separation（不依赖 ONNX，原生 MATLAB 可跑）
% ------------------------------------------------------------------------
% RUN_CHISEP_ONLY / 总入口里是否进入"方法对比"模式：
%   true  : 同时跑深度学习 + 传统优化，输出对比（mod_chisep_method_comparison）
%   false : 只跑单一方法（cfg.sep.adapter_function 指定的那个）
P.chisepRunMethodCompare = true;
% 方法对比要跑哪些方法
P.chisepCompareMethods = {'onnx','opt'};   % {'onnx'},{'opt'},或两者
P.optMethod   = 'iLSQR';   % 'iLSQR'(L2,快) | 'MEDI'(L1-TV)
P.optLambda   = 1e-2;      % 优化正则化权重
P.optWr2      = 1.0;       % R2' 数据项权重
P.optMaxIter  = 100;       % MEDI/ADMM 迭代
% 可选 ROI 标签体(.mat，整型标签，与被试同尺寸)，用于 ROI 均值对比
P.roiLabelFile = '';

% ------------------------------------------------------------------------
% 两被试(59 vs 72)配准后比较
% ------------------------------------------------------------------------
P.twoSubjFixed     = 'normal';      % 以谁为固定参考: 'normal' | 'elderly'
P.twoSubjTransform = 'rigid';       % 'rigid' | 'affine' | 'rigid+affine'

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
% 调参指南（针对"脑内颜色偏淡/过平滑"问题）：
%   - lambda 是最关键的正则化权重：lambda 越大 → 越平滑、对比越淡、streaking 越少；
%     lambda 越小 → 越锐利、深部核团(苍白球/红核)对比越强，但噪声/伪影增多。
%   - beta  : 弱谐波约束权重，主要管残余背景场，对整体亮度影响较小。
%   - muh/mu: ADMM 步长，主要影响收敛速度，不直接决定最终亮度。
%
% 因你反馈"脑内偏淡"，默认 lambda 由 5e-4 下调到 3e-4（更锐、对比更强、不易过平滑）。
%   若仍偏淡   → 进一步降到 2e-4 或 1e-4；
%   若噪声/streaking 变多 → 回调到 4e-4 ~ 5e-4。
% 建议先用 RUN_WHQSM_LAMBDA_SWEEP 做一次 lambda 扫描，按 ROI 数值客观选定。
P.whqsmMaxIter = 200;
P.whqsmTol = 1e-5;
% lambda 决策建议: 不要凭肉眼盲调(肉眼"淡"常是显示窗位错觉)。
% 先跑 RUN_WHQSM_LAMBDA_SWEEP([3e-4 2e-4 1e-4 5e-5 2e-5])，按深部核团 ROI 的
% p99 是否达到铁核量级(GP~0.10-0.18ppm)来选; 选定后填到这里并【重跑 WH-QSM】才生效
% (现有 whqsm_*_complete.mat 是旧 lambda 的结果，不会自动更新)。
P.whqsmLambda = 4e-4;     % 文献常用 1e-4~6e-4。注意: ADMM 的 mu1 会自动设为 100*lambda
                          % (在 run_whqsm_comparison 内), 这是 FANSI 收敛的关键, 勿手改成 < lambda。
P.whqsmBeta = 150;        % WH 谐波约束权重(文献 WH-QSM: 150~500)
P.whqsmMuh = 10;          % WH 谐波一致性(文献: ~10; 之前误设 5)

% 可视化窗位：窗口太宽也会让脑内"看起来淡"。3T 深部核团 χ 量级约 0.08~0.18 ppm，
% 这里把 QSM 显示窗位收紧到 ±0.10，便于看清铁核对比（不影响数值，只影响显示）。
P.qsmDisplayClim = [-0.10 0.10];

% true  : 开始前 restoredefaultpath，最大限度避免旧 path 污染；随后重新 addpath 本库和 SEPIA
% false : 不清空用户已有 path，仅把本库和 SEPIA 放到前面。更温和，通常推荐。
P.resetMatlabPath = false;

% 输出日志目录。留空时自动使用 dataRoot/_qsm_comparison_results/logs
P.logDir = '';

end
