function chi = dl_xqsm(input_norm, Mask, N, voxel_size, norm_factor, cfg)
% dl_xqsm.m
% xQSM 推理模块
%
% 参考: Gao et al., "xQSM: quantitative susceptibility mapping with
%        octave convolutional and noise-regularized neural networks",
%        NMR in Biomedicine, 2021
%
% 特点:
%   - Octave Convolution: 将特征分为高频/低频两路
%     高频路径保留细节, 低频路径捕获全局结构
%   - 对 magic angle cone 附近的噪声放大有更好的抑制
%   - 输入: 归一化局部场 [64x64x64] patch
%   - 输出: susceptibility map [64x64x64] patch
%
% 部署: ONNX 或 Python bridge

chi = [];

onnx_path = cfg.deeplearning.xqsm_onnx;

%% 方式 A: ONNX
if exist(onnx_path, 'file')
    fprintf('  加载 xQSM ONNX 模型: %s\n', onnx_path);
    
    try
        net = importONNXNetwork(onnx_path, ...
            'OutputLayerType', 'regression', ...
            'InputDataFormats', 'BCSSS', ...
            'OutputDataFormats', 'BCSSS');
        
        fprintf('  xQSM ONNX 加载成功\n');
        chi = patch_inference_xqsm(net, input_norm, Mask, N, [64 64 64], norm_factor);
        return;
        
    catch ME
        fprintf('  xQSM ONNX 导入失败: %s\n', ME.message);
    end
end

%% 方式 B: Python bridge
if isempty(chi)
    chi = dl_python_bridge('xqsm', input_norm, Mask, N, voxel_size, norm_factor, cfg);
end

if ~isempty(chi)
    return;
end

%% 无可用模型
fprintf('  xQSM 模型文件不可用\n');
fprintf('  获取方式:\n');
fprintf('    1. 从 https://github.com/sunhongfu/deepMRI 下载 xQSM 权重\n');
fprintf('    2. 使用 torch.onnx.export 导出为 ONNX\n');
fprintf('    3. 放入: %s\n', cfg.dlModelDir);

end

%% =========================================================================
function chi = patch_inference_xqsm(net, input_norm, Mask, N, patch_size, norm_factor)
% xQSM patch-based 推理
% xQSM 使用与 QSMnet+ 相同的 patch 策略但网络内部不同

stride = patch_size / 2;

chi_accum = zeros(N);
weight_accum = zeros(N);

win = hanning_3d(patch_size);

starts = cell(3, 1);
for d = 1:3
    s = 1:stride(d):N(d)-patch_size(d)+1;
    if isempty(s)
        s = 1;
    end
    if s(end) + patch_size(d) - 1 < N(d)
        s = [s, N(d) - patch_size(d) + 1];
    end
    starts{d} = s;
end

count = 0;
for i1 = starts{1}
    for i2 = starts{2}
        for i3 = starts{3}
            
            idx1 = i1:min(i1+patch_size(1)-1, N(1));
            idx2 = i2:min(i2+patch_size(2)-1, N(2));
            idx3 = i3:min(i3+patch_size(3)-1, N(3));
            
            % 处理边界 padding
            patch_in = zeros(patch_size);
            actual_size = [numel(idx1), numel(idx2), numel(idx3)];
            patch_in(1:actual_size(1), 1:actual_size(2), 1:actual_size(3)) = ...
                input_norm(idx1, idx2, idx3);
            
            if max(abs(patch_in(:))) < 1e-6
                continue;
            end
            
            patch_in_5d = reshape(single(patch_in), [1, 1, patch_size]);
            
            try
                patch_out = double(predict(net, dlarray(patch_in_5d, 'BCSSS')));
                patch_out = extractdata(patch_out);
            catch
                patch_out = double(predict(net, patch_in_5d));
            end
            patch_out = reshape(patch_out, patch_size);
            
            % 只取实际大小部分
            win_crop = win(1:actual_size(1), 1:actual_size(2), 1:actual_size(3));
            out_crop = patch_out(1:actual_size(1), 1:actual_size(2), 1:actual_size(3));
            
            chi_accum(idx1, idx2, idx3) = chi_accum(idx1, idx2, idx3) + out_crop .* win_crop;
            weight_accum(idx1, idx2, idx3) = weight_accum(idx1, idx2, idx3) + win_crop;
            
            count = count + 1;
        end
    end
end

fprintf('  xQSM 推理 patch 数: %d\n', count);

weight_accum(weight_accum < eps) = eps;
chi = chi_accum ./ weight_accum;
chi = chi * norm_factor;
chi = chi .* Mask;

end

%% =========================================================================
function win = hanning_3d(patch_size)

w1 = hann(patch_size(1));
w2 = hann(patch_size(2));
w3 = hann(patch_size(3));

win = w1(:) .* w2(:).';
win = reshape(win, [patch_size(1), patch_size(2), 1]) .* reshape(w3, [1, 1, patch_size(3)]);

end