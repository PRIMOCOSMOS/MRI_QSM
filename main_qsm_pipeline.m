function main_qsm_pipeline()
% main_qsm_pipeline.m
% QSM2016 Reconstruction Challenge — 模块化完整 Pipeline
%
% 核心步骤:
%   1. 数据加载
%   2. 相位解缠 (验证/记录)
%   3. 背景场去除 (V-SHARP / PDF / LBV / WH-QSM)
%   4. 偶极子反演 (TKD / CFL2 / iLSQR / MEDI with structural prior)
%   5. 深度学习方法 (QSMnet+ / xQSM / MATLAB 3D U-Net)
%   6. 定量评估 (RMSE / HFEN / SSIM / WM-GM Error)
%   7. 可视化 (三平面 / 对比面板 / 误差图 / 指标图)
%
% 使用方法:
%   直接运行 main_qsm_pipeline() 即可
%
% 依赖:
%   - MEDI toolbox (D:\MRI_PRO\MRILAB_X\MEDI_toolbox-2024.11.26)
%   - SEPIA (D:\MRI_PRO\MRILAB_X\sepia) [可选]
%   - Deep Learning Toolbox [可选, 用于 MATLAB U-Net]
%   - Image Processing Toolbox [用于 morphological operations]

clc; close all;
try set(0, 'DefaultFigureWindowStyle', 'docked'); catch; end

%% 加载配置
addpath(fullfile(fileparts(mfilename('fullpath')), 'config'));
addpath(fullfile(fileparts(mfilename('fullpath')), 'Utils_self'));
addpath(fullfile(fileparts(mfilename('fullpath')), 'modules'));

cfg = pipeline_config();

%% 添加工具箱路径
if exist(cfg.mediRoot, 'dir')
    addpath(genpath(cfg.mediRoot));
    fprintf('MEDI toolbox 已添加: %s\n', cfg.mediRoot);
else
    warning('MEDI toolbox 未找到: %s', cfg.mediRoot);
end

if exist(cfg.sepiaRoot, 'dir')
    addpath(genpath(cfg.sepiaRoot));
    fprintf('SEPIA 已添加: %s\n', cfg.sepiaRoot);
end

%% Step 1: 数据加载
fprintf('============================================================\n');
fprintf(' Step 1: 数据加载\n');
fprintf('============================================================\n');
data = mod_load_data(cfg);

%% Step 2: 相位解缠
fprintf('============================================================\n');
fprintf(' Step 2: 相位解缠\n');
fprintf('============================================================\n');
data = mod_phase_unwrap(data, cfg);

%% Step 3: 背景场去除
fprintf('============================================================\n');
fprintf(' Step 3: 背景场去除\n');
fprintf('============================================================\n');
[local_field, bg_results] = mod_background_removal(data, cfg);

%% Step 4: 偶极子反演（传统方法）
fprintf('============================================================\n');
fprintf(' Step 4: 偶极子反演\n');
fprintf('============================================================\n');
[qsm_results, qsm_names] = mod_dipole_inversion(local_field, data, cfg);

%% Step 5: 深度学习方法
fprintf('============================================================\n');
fprintf(' Step 5: 深度学习 QSM\n');
fprintf('============================================================\n');
[dl_results, dl_names] = mod_deep_learning(local_field, data, cfg);

% 合并所有结果
all_results = cat(4, qsm_results, dl_results);
all_names = [qsm_names; dl_names];

fprintf('总计 %d 种重建方法', numel(all_names));
fprintf('\n');

%% Step 6: 定量评估
fprintf('============================================================\n');
fprintf(' Step 6: 定量评估\n');
fprintf('============================================================\n');
metrics = mod_evaluation(all_results, all_names, data, cfg);

%% Step 7: 可视化
fprintf('============================================================\n');
fprintf(' Step 7: 可视化\n');
fprintf('============================================================\n');
mod_visualization(all_results, all_names, data, metrics, cfg);

%% 保存总结果
fprintf('保存完整 pipeline 结果...\n');
save(fullfile(cfg.resultDir, 'pipeline_complete_results.mat'), ...
    'all_results', 'all_names', 'metrics', ...
    'local_field', 'bg_results', ...
    'qsm_results', 'qsm_names', ...
    'dl_results', 'dl_names', ...
    'cfg', '-v7.3');

fprintf('============================================================\n');
fprintf(' Pipeline 完成\n');
fprintf('============================================================\n');
fprintf('输出目录: %s', cfg.outDir);
fprintf('结果文件: %s\n', fullfile(cfg.resultDir, 'pipeline_complete_results.mat'));
fprintf('评估指标: %s\n', fullfile(cfg.resultDir, 'evaluation_metrics.csv'));
fprintf('图像目录: %s\n', cfg.figDir);
fprintf('\n');

end