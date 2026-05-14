function chi = dl_python_bridge(model_name, input_norm, Mask, N, ~, norm_factor, cfg)
% dl_python_bridge.m
% 通过 MATLAB-Python 接口调用 PyTorch 深度学习模型
%
% 支持模型: 'qsmnet_plus', 'xqsm'
%
% 前提条件:
%   1. MATLAB 已配置 Python 环境 (pyenv)
%   2. Python 环境中安装了 torch, numpy
%   3. 模型权重文件 (.pth) 位于 cfg.dlModelDir
%
% 调用流程:
%   MATLAB -> numpy array -> Python inference script -> numpy array -> MATLAB

chi = [];

%% 检查 Python 环境
try
    pe = pyenv;
    if pe.Status ~= "Loaded" && isempty(pe.Executable)
        fprintf('  Python 环境未配置，跳过 %s\n', model_name);
        return;
    end
catch
    fprintf('  pyenv 不可用，跳过 Python bridge\n');
    return;
end

%% 检查 torch 可用性
try
    py.importlib.import_module('torch');
catch
    fprintf('  Python torch 不可用，跳过 %s\n', model_name);
    fprintf('  请在 Python 环境中安装: pip install torch\n');
    return;
end

%% 准备推理脚本路径
script_dir = fullfile(cfg.dlModelDir, 'python_scripts');
if ~exist(script_dir, 'dir')
    mkdir(script_dir);
end

% 生成推理脚本
script_path = fullfile(script_dir, sprintf('infer_%s.py', model_name));
generate_inference_script(script_path, model_name, cfg);

%% 保存输入数据为临时 .npy
temp_input = fullfile(cfg.resultDir, 'temp_dl_input.mat');
temp_output = fullfile(cfg.resultDir, 'temp_dl_output.mat');

save(temp_input, 'input_norm', 'Mask', 'N', 'norm_factor', '-v7');

%% 调用 Python
fprintf('  调用 Python 推理: %s\n', model_name);

try
    py.runfile(script_path, pyargs( ...
        'input_path', temp_input, ...
        'output_path', temp_output, ...
        'model_dir', cfg.dlModelDir));
catch ME
    % 备选: 使用 system 调用
    cmd = sprintf('"%s" "%s" --input "%s" --output "%s" --model_dir "%s"', ...
        char(pe.Executable), script_path, temp_input, temp_output, cfg.dlModelDir);
    [status, result] = system(cmd);
    
    if status ~= 0
        fprintf('  Python 调用失败: %s\n', result);
        cleanup_temp(temp_input, temp_output);
        return;
    end
end

%% 加载结果
if exist(temp_output, 'file')
    S = load(temp_output);
    if isfield(S, 'chi')
        chi = double(S.chi);
        chi = chi .* Mask;
        fprintf('  Python bridge 推理成功: %s\n', model_name);
    end
end

%% 清理临时文件
cleanup_temp(temp_input, temp_output);

end

%% =========================================================================
function cleanup_temp(varargin)
for i = 1:numel(varargin)
    if exist(varargin{i}, 'file')
        delete(varargin{i});
    end
end

end

%% =========================================================================
function generate_inference_script(script_path, model_name, ~)
% 生成 Python 推理脚本

fid = fopen(script_path, 'w');
if fid == -1
    return;
end

fprintf(fid, '#!/usr/bin/env python3');
fprintf(fid, '"""Auto-generated inference script for %s"""', model_name);
fprintf(fid, 'import sys, os');
fprintf(fid, 'import numpy as np');
fprintf(fid, 'import scipy.io as sio');
fprintf(fid, 'import torch');
fprintf(fid, 'import torch.nn as nn');

fprintf(fid, 'def main(input_path, output_path, model_dir):');
fprintf(fid, '    # Load input');
fprintf(fid, '    data = sio.loadmat(input_path)');
fprintf(fid, '    input_norm = data["input_norm"].astype(np.float32)');
fprintf(fid, '    mask = data["Mask"].astype(bool)');
fprintf(fid, '    norm_factor = float(data["norm_factor"])');

fprintf(fid, '    # Load model');
fprintf(fid, '    model_path = os.path.join(model_dir, "%s.pth")', model_name);
fprintf(fid, '    if not os.path.exists(model_path):');
fprintf(fid, '        print(f"Model not found: {model_path}")');
fprintf(fid, '        return');

fprintf(fid, '    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")');
fprintf(fid, '    model = torch.load(model_path, map_location=device)');
fprintf(fid, '    model.eval()');

fprintf(fid, '    # Patch-based inference');
fprintf(fid, '    patch_size = 64');
fprintf(fid, '    stride = 32');
fprintf(fid, '    N = input_norm.shape');
fprintf(fid, '    chi = np.zeros(N, dtype=np.float32)');
fprintf(fid, ' np.zeros(N, dtype=np.float32)');

fprintf(fid, '    with torch.no_grad():');
fprintf(fid, '        for i in range(0, N[0]-patch_size+1, stride):');
fprintf(fid, '            for j in range(0, N[1]-patch_size+1, stride):');
fprintf(fid, '                for k in range(0, N[2]-patch_size+1, stride):');
fprintf(fid, '                    patch = input_norm[i:i+patch_size, j:j+patch_size, k:k+patch_size]');
fprintf(fid, '                    if np.max(np.abs(patch)) < 1e-6:');
fprintf(fid, '                        continue');
fprintf(fid, '                    x = torch.from_numpy(patch[None, None]).to(device)');
fprintf(fid, '                    y = model(x).cpu().numpy()[0, 0]');
fprintf(fid, '                    chi[i:i+patch_size, j:j+patch_size, k:k+patch_size] += y');
fprintf(fid, '                    weight[i:i+patch_size, j:j+patch_size, k:k+patch_size] += 1');

fprintf(fid, '    weight[weight < 1] = 1');
fprintf(fid, '    chi = chi / weight * norm_factor');
fprintf(fid, '    chi[~mask] = 0');

fprintf(fid, '    sio.savemat(output_path, {"chi": chi})');
fprintf(fid, '    print("Inference complete")');

fprintf(fid, 'if __name__ == "__main__":');
fprintf(fid, '    import argparse');
fprintf(fid, '    parser = argparse.ArgumentParser()');
fprintf(fid, '    parser.add_argument("--input", required=True)');
fprintf(fid, '    parser.add_argument("--output", required=True)');
fprintf(fid, '    parser.add_argument("--model_dir", required=True)');
fprintf(fid, '    args = parser.parse_args()');
fprintf(fid, '    main(args.input, args.output, args.model_dir)');

fclose(fid);

end