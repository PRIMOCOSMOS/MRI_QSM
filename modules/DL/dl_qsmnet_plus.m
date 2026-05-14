function chi = dl_qsmnet_plus(input_norm, Mask, N, voxel_size, norm_factor, cfg)
% dl_qsmnet_plus.m
% QSMnet+ 推理模块
%
% 参考: Jung et al., "Overview of quantitative susceptibility mapping using
%        deep learning: Current status, challenges and opportunities",
%        NMR in Biomedicine, 2022
%        Yoon et al., "Quantitative susceptibility mapping using deep
%        neural network: QSMnet", NeuroImage, 2018
%        Jung et al., "Exploring linearity of deep neural network trained
%        QSM: QSMnet+", NeuroImage, 2020
%
% 网络架构: 3D U-Net
%   输入: 归一化局部场 patch [64x64x64]
%   输出: susceptibility map patch [64x64x64]
%
% 部署方式:
%   方式A: ONNX 导入 (MATLAB R2021b+, Deep Learning Toolbox)
%   方式B: Python bridge (PyTorch)
%   方式C: 预计算 .mat 权重 + 手写前向传播

chi = [];

onnx_path = cfg.deeplearning.qsmnet_onnx;

%% 方式 A: ONNX 导入
if exist(onnx_path, 'file')
    fprintf('  加载 QSMnet+ ONNX 模型: %s\n', onnx_path);
    
    try
        % MATLAB R2021b+ importONNXNetwork
        net = importONNXNetwork(onnx_path, ...
            'OutputLayerType', 'regression', ...
            'InputDataFormats', 'BCSSS', ...
            'OutputDataFormats', 'BCSSS');
        
        fprintf('  ONNX 模型加载成功, 执行 patch-based 推理...\n');
        chi = patch_based_inference(net, input_norm, Mask, N, [64 64 64], norm_factor);
        return;
        
    catch ME
        fprintf('  ONNX 导入失败: %s\n', ME.message);
        fprintf('  尝试 Python bridge...\n');
    end
end

%% 方式 B: Python bridge
if isempty(chi)
    chi = dl_python_bridge('qsmnet_plus', input_norm, Mask, N, voxel_size, norm_factor, cfg);
end

if ~isempty(chi)
    return;
end

%% 方式 C: 无可用模型
fprintf('  QSMnet+ 模型文件不可用\n');
fprintf('  获取方式:\n');
fprintf('    1. 从 https://github.com/SNU-LIST/QSMnet 下载预训练权重\n');
fprintf('    2. 导出为 ONNX 格式放入: %s\n', cfg.dlModelDir);
fprintf('    3. 或安装 Python 环境并配置 PyTorch\n');

end

%% =========================================================================
function chi = patch_based_inference(net, input_norm, Mask, N, patch_size, norm_factor)
% Patch-based 3D 推理
%
% 将整个体积切分为重叠 patch, 逐个送入网络, 再拼接
% 使用 Hanning 窗加权避免拼接伪影

stride = patch_size / 2;  % 50% 重叠
stride = max(stride, 1);

chi_accum = zeros(N);
weight_accum = zeros(N);

% Hanning 窗
win = hanning_3d(patch_size);

% 计算 patch 起始位置
starts = cell(3, 1);
for d = 1:3
    s = 1:stride(d):N(d)-patch_size(d)+1;
    if s(end) + patch_size(d) - 1 < N(d)
        s = [s, N(d) - patch_size(d) + 1];
    end
    starts{d} = s;
end

total_patches = numel(starts{1}) * numel(starts{2}) * numel(starts{3});
fprintf('  总 patch 数: %d\n', total_patches);

count = 0;
for i1 = starts{1}
    for i2 = starts{2}
        for i3 = starts{3}
            
            idx1 = i1:i1+patch_size(1)-1;
            idx2 = i2:i2+patch_size(2)-1;
            idx3 = i3:i3+patch_size(3)-1;
            
            patch_in = input_norm(idx1, idx2, idx3);
            
            % 跳过全零 patch
            if max(abs(patch_in(:))) < 1e-6
                continue;
            end
            
            % 网络推理 [1 x 1 x 64 x 64 x 64]
            patch_in_5d = reshape(patch_in, [1, 1, patch_size]);
            
            try
                patch_out_5d = predict(net, dlarray(single(patch_in_5d), 'BCSSS'));
                patch_out = double(extractdata(patch_out_5d));
                patch_out = reshape(patch_out, patch_size);
            catch
                % 兼容旧版 predict 接口
                patch_out = double(predict(net, single(patch_in_5d)));
                patch_out = reshape(patch_out, patch_size);
            end
            
            % 加权累加
            chi_accum(idx1, idx2, idx3) = chi_accum(idx1, idx2, idx3) + patch_out .* win;
            weight_accum(idx1, idx2, idx3) = weight_accum(idx1, idx2, idx3) + win;
            
            count = count + 1;
        end
    end
end

fprintf('  实际推理 patch 数: %d\n', count);

% 归一化
weight_accum(weight_accum < eps) = eps;
chi = chi_accum ./ weight_accum;

% 反归一化
chi = chi * norm_factor;
chi = chi .* Mask;

end

%% =========================================================================
function win = hanning_3d(patch_size)
% 3D Hanning 窗

w1 = hann(patch_size(1));
w2 = hann(patch_size(2));
w3 = hann(patch_size(3));

win = w1(:) .* w2(:).' ;
win = reshape(win, [patch_size(1), patch_size(2), 1]) .* reshape(w3, [1, 1, patch_size(3)]);

end