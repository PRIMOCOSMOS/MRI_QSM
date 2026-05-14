function [dl_results, dl_names] = mod_deep_learning(local_field, data, cfg)
% mod_deep_learning.m
% 深度学习 QSM 重建 — 调度入口
%
% 依次尝试:
%   1. QSMnet+ (ONNX 或 Python bridge)
%   2. xQSM   (ONNX 或 Python bridge)
%   3. MATLAB 原生 3D U-Net (patch-based, 可在本地训练/推理)
%
% 每个模型的实现位于 modules/dl/ 子目录

dl_results = [];
dl_names = {};

if ~cfg.deeplearning.enable
    fprintf('深度学习模块已禁用\n');
    return;
end

% 添加 dl 子模块路径
dlModuleDir = fullfile(fileparts(mfilename('fullpath')), 'dl');
addpath(dlModuleDir);

N = data.N;
Mask = data.Mask;

%% 公共预处理
[input_norm, norm_factor] = dl_prepare_input(local_field, Mask);

%% QSMnet+
fprintf('--- 深度学习：QSMnet+ ---\n');
try
    chi_qsmnet = dl_qsmnet_plus(input_norm, Mask, N, data.spatial_res, norm_factor, cfg);
    if ~isempty(chi_qsmnet)
        dl_results = cat(4, dl_results, chi_qsmnet);
        dl_names{end+1, 1} = 'QSMnet+';
        print_volume_summary('QSMnet+', chi_qsmnet, Mask);
    end
catch ME
    fprintf('QSMnet+ 失败: %s\n', ME.message);
end

%% xQSM
fprintf('--- 深度学习: xQSM ---\n');
try
    chi_xqsm = dl_xqsm(input_norm, Mask, N, data.spatial_res, norm_factor, cfg);
    if ~isempty(chi_xqsm)
        dl_results = cat(4, dl_results, chi_xqsm);
        dl_names{end+1, 1} = 'xQSM';
        print_volume_summary('xQSM', chi_xqsm, Mask);
    end
catch ME
    fprintf('xQSM 失败: %s\n', ME.message);
end

%% MATLAB 原生 3D U-Net (patch-based)
fprintf('--- 深度学习: MATLAB 原生 3D U-Net ---\n');
try
    chi_unet = dl_matlab_unet(local_field, data.chi_33, Mask, N, data.spatial_res, cfg);
    if ~isempty(chi_unet)
        dl_results = cat(4, dl_results, chi_unet);
        dl_names{end+1, 1} = 'MATLAB-UNet3D';
        print_volume_summary('MATLAB-UNet3D', chi_unet, Mask);
    end
catch ME
    fprintf('MATLAB U-Net 失败: %s\n', ME.message);
end

%% 保存
if ~isempty(dl_results)
    save(fullfile(cfg.resultDir, 'deep_learning_results.mat'), ...
        'dl_results', 'dl_names', '-v7.3');
end

fprintf('深度学习模块完成, 成功运行 %d 个模型\n', numel(dl_names));

end