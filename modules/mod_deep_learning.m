function [dl_results, dl_names] = mod_deep_learning(local_field, data, cfg)
% mod_deep_learning.m
% 深度学习 QSM 重建调度入口（当前聚焦 xQSM）

dl_results = [];
dl_names = {};

if ~cfg.deeplearning.enable
    fprintf('深度学习模块已禁用\n');
    return;
end

% 兼容 modules/DL 与 modules/dl
dlModuleDirA = fullfile(fileparts(mfilename('fullpath')), 'DL');
dlModuleDirB = fullfile(fileparts(mfilename('fullpath')), 'dl');
if exist(dlModuleDirA, 'dir')
    addpath(dlModuleDirA);
elseif exist(dlModuleDirB, 'dir')
    addpath(dlModuleDirB);
else
    warning('未找到 DL 子模块目录: modules/DL 或 modules/dl');
end

N = data.N;
Mask = data.Mask;

% 公共归一化
[input_norm, norm_factor] = dl_prepare_input(local_field, Mask);

models_to_run = cfg.deeplearning.models;
if ischar(models_to_run)
    models_to_run = {models_to_run};
end

for i = 1:numel(models_to_run)
    model_name = models_to_run{i};

    if strcmpi(model_name, 'xQSM')
        fprintf('--- 深度学习: xQSM (.pth via Python bridge) ---\n');
        try
            chi_xqsm = dl_xqsm(input_norm, Mask, N, data.spatial_res, norm_factor, cfg);
            if ~isempty(chi_xqsm)
                dl_results = cat(4, dl_results, chi_xqsm);
                dl_names{end+1, 1} = 'xQSM'; %#ok<AGROW>
                print_volume_summary('xQSM', chi_xqsm, Mask);
            end
        catch ME
            fprintf('xQSM 失败: %s\n', ME.message);
        end
    else
        fprintf('跳过未启用模型: %s\n', model_name);
    end
end

if ~isempty(dl_results)
    save(fullfile(cfg.resultDir, 'deep_learning_results.mat'), ...
        'dl_results', 'dl_names', '-v7.3');
end

fprintf('深度学习模块完成，成功运行 %d 个模型\n', numel(dl_names));

end