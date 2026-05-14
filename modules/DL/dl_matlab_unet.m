function chi = dl_matlab_unet(local_field, chi_ref, Mask, N, ~, cfg)
% dl_matlab_unet.m
% MATLAB 原生 3D U-Net QSM 重建
%
% 当外部预训练模型不可用时，使用 MATLAB Deep Learning Toolbox
% 构建轻量级 3D U-Net，在 QSM2016 数据上做 patch-based 训练+推理
%
% 策略:
%   - 从 local_field / chi_33 中提取 patch 对作为训练数据
%   - 训练小型 3D U-Net (encoder-decoder, 2层深度)
%   - 用训练好的网络对整个体积做推理
%
% 限制:
%   - 单受试者数据量有限，此方法为 DL pipeline 演示
%   - 实际应用需多受试者大规模训练集
%   - 需要 MATLAB R2019b+ 和 Deep Learning Toolbox
%
% 输入:
%   local_field - 局部场 (ppm)
%   chi_ref     - 参考 susceptibility (chi_33)
%   Mask        - 脑 mask
%   N           - 矩阵尺寸 [Nx Ny Nz]
%   voxel_size  - 体素尺寸 [dx dy dz] mm
%   cfg         - 配置结构体
%
% 输出:
%   chi         - 重建的 susceptibility map (ppm)

chi = [];

%% 检查工具箱可用性
if ~license('test', 'Neural_Network_Toolbox')
    fprintf('  Deep Learning Toolbox 不可用，跳过 MATLAB U-Net\n');
    return;
end

try
    convolution3dLayer(3, 8, 'Padding', 'same');
catch
    fprintf('  当前 MATLAB 版本不支持 3D 卷积层，跳过\n');
    return;
end

fprintf('  构建 MATLAB 原生 3D U-Net (轻量级演示版)\n');

%% 超参数
patch_size = [32 32 32];
n_train_patches = 300;
n_epochs = 30;
mini_batch = 4;
lr = 1e-3;
n_base_filters = 16;

%% 检查是否已有训练好的模型
model_save_path = fullfile(cfg.resultDir, 'matlab_unet3d_trained.mat');

if exist(model_save_path, 'file')
    fprintf('  发现已训练模型，直接加载: %s\n', model_save_path);
    S = load(model_save_path, 'net', 'norm_in', 'norm_out');
    net = S.net;
    norm_in = S.norm_in;
    norm_out = S.norm_out;
else
    %% 提取训练 patch 对
    fprintf('  提取训练 patch 对 (n=%d)...\n', n_train_patches);
    
    % 归一化因子
    vals_in = local_field(Mask);
    norm_in = prctile(abs(vals_in(isfinite(vals_in))), 99.5);
    if norm_in <= 0 || ~isfinite(norm_in); norm_in = 1; end
    
    vals_out = chi_ref(Mask);
    norm_out = prctile(abs(vals_out(isfinite(vals_out))), 99.5);
    if norm_out <= 0 || ~isfinite(norm_out); norm_out = 1; end
    
    in_norm = local_field / norm_in;
    tgt_norm = chi_ref / norm_out;
    
    [X_train, Y_train] = extract_training_patches( ...
        in_norm, tgt_norm, Mask, N, patch_size, n_train_patches);
    
    if isempty(X_train)
        fprintf('  无法提取足够训练 patch，跳过\n');
        return;
    end
    
    n_actual = size(X_train, 5);
    fprintf('  实际提取 %d 个训练 patch\n', n_actual);
    
    %% 构建网络
    fprintf('  构建 3D U-Net (base filters=%d)...\n', n_base_filters);
    lgraph = build_simple_3d_unet(patch_size, n_base_filters);
    
    %% 训练
    fprintf('  开始训练 (%d epochs, batch=%d, lr=%.1e)...\n', n_epochs, mini_batch, lr);
    
    options = trainingOptions('adam', ...
        'InitialLearnRate', lr, ...
        'MaxEpochs', n_epochs, ...
        'MiniBatchSize', mini_batch, ...
        'Shuffle', 'every-epoch', ...
        'Plots', 'none', ...
        'Verbose', true, ...
        'VerboseFrequency', 5, ...
        'ExecutionEnvironment', 'auto');
    
    % 数据格式: X [H W D C N], Y [H W D C N]
    % 已经是 [32 32 32 1 n_actual]
    
    try
        net = trainNetwork(X_train, Y_train, lgraph, options);
    catch ME
        fprintf('  训练失败: %s\n', ME.message);
        fprintf('  尝试使用 trainnet (R2023b+)...\n');
        try
            net = trainnet(X_train, Y_train, lgraph, "mse", options);
        catch ME2
            fprintf('  trainnet 也失败: %s\n', ME2.message);
            return;
        end
    end
    
    % 保存模型
    save(model_save_path, 'net', 'norm_in', 'norm_out', '-v7.3');
    fprintf('  模型已保存: %s\n', model_save_path);
end

%% 推理
fprintf('  执行 patch-based 推理...\n');

in_norm = local_field / norm_in;
in_norm(~Mask) = 0;

chi = patch_based_inference_unet(net, in_norm, Mask, N, patch_size, norm_out);

fprintf('  MATLAB U-Net 推理完成\n');

end

%% =========================================================================
% 提取训练 patch 对
% =========================================================================
function [X, Y] = extract_training_patches(in_norm, tgt_norm, Mask, N, patch_size, n_patches)
% 从体积中随机提取 mask 内有效的 patch 对
%
% 输出:
%   X - [H W D 1 n] single, 输入 patch
%   Y - [H W D 1 n] single, 目标 patch

valid_starts = zeros(0, 3);
max_attempts = n_patches * 20;

for attempt = 1:max_attempts
    s1 = randi(N(1) - patch_size(1) + 1);
    s2 = randi(N(2) - patch_size(2) + 1);
    s3 = randi(N(3) - patch_size(3) + 1);
    
    idx1 = s1:s1+patch_size(1)-1;
    idx2 = s2:s2+patch_size(2)-1;
    idx3 = s3:s3+patch_size(3)-1;
    
    mask_patch = Mask(idx1, idx2, idx3);
    
    % 要求 patch 内至少 50% 在 mask 内
    if sum(mask_patch(:)) / prod(patch_size) > 0.5
        valid_starts(end+1, :) = [s1 s2 s3]; %#ok<AGROW>
    end
    
    if size(valid_starts, 1) >= n_patches
        break;
    end
end

n_actual = size(valid_starts, 1);

if n_actual == 0
    X = [];
    Y = [];
    return;
end

X = zeros([patch_size, 1, n_actual], 'single');
Y = zeros([patch_size, 1, n_actual], 'single');

for i = 1:n_actual
    s = valid_starts(i, :);
    idx1 = s(1):s(1)+patch_size(1)-1;
    idx2 = s(2):s(2)+patch_size(2)-1;
    idx3 = s(3):s(3)+patch_size(3)-1;
    
    X(:,:,:,1,i) = single(in_norm(idx1, idx2, idx3));
    Y(:,:,:,1,i) = single(tgt_norm(idx1, idx2, idx3));
end

end

%% =========================================================================
% 构建简化 3D U-Net
% =========================================================================
function lgraph = build_simple_3d_unet(patch_size, nf)
% 构建 2 层深度的 3D U-Net
%
% 架构:
%   Input -> [Enc1: 2xConv nf] -> Pool -> [Enc2: 2xConv 2nf] -> Pool
%   -> [Bottleneck: 2xConv 4nf]
%   -> TransConv -> Concat(Enc2) -> [Dec2: 2xConv 2nf]
%   -> TransConv -> Concat(Enc1) -> [Dec1: 2xConv nf]
%   -> Conv 1x1x1 -> Output

input_size = [patch_size, 1];

layers = [
    image3dInputLayer(input_size, 'Name', 'input', 'Normalization', 'none')
    
    % === Encoder 1 ===
    convolution3dLayer(3, nf, 'Padding', 'same', 'Name', 'enc1_conv1')
    batchNormalizationLayer('Name', 'enc1_bn1')
    reluLayer('Name', 'enc1_relu1')
    convolution3dLayer(3, nf, 'Padding', 'same', 'Name', 'enc1_conv2')
    batchNormalizationLayer('Name', 'enc1_bn2')
    reluLayer('Name', 'enc1_relu2')
    ];

lgraph = layerGraph(layers);

% Pool 1
lgraph = addLayers(lgraph, maxPooling3dLayer(2, 'Stride', 2, 'Name', 'pool1'));
lgraph = connectLayers(lgraph, 'enc1_relu2', 'pool1');

% === Encoder 2 ===
enc2 = [
    convolution3dLayer(3, nf*2, 'Padding', 'same', 'Name', 'enc2_conv1')
    batchNormalizationLayer('Name', 'enc2_bn1')
    reluLayer('Name', 'enc2_relu1')
    convolution3dLayer(3, nf*2, 'Padding', 'same', 'Name', 'enc2_conv2')
    batchNormalizationLayer('Name', 'enc2_bn2')
    reluLayer('Name', 'enc2_relu2')
    ];
lgraph = addLayers(lgraph, enc2);
lgraph = connectLayers(lgraph, 'pool1', 'enc2_conv1');

% Pool 2
lgraph = addLayers(lgraph, maxPooling3dLayer(2, 'Stride', 2, 'Name', 'pool2'));
lgraph = connectLayers(lgraph, 'enc2_relu2', 'pool2');

% === Bottleneck ===
bottleneck = [
    convolution3dLayer(3, nf*4, 'Padding', 'same', 'Name', 'bn_conv1')
    batchNormalizationLayer('Name', 'bn_bn1')
    reluLayer('Name', 'bn_relu1')
    convolution3dLayer(3, nf*4, 'Padding', 'same', 'Name', 'bn_conv2')
    batchNormalizationLayer('Name', 'bn_bn2')
    reluLayer('Name', 'bn_relu2')
    ];
lgraph = addLayers(lgraph, bottleneck);
lgraph = connectLayers(lgraph, 'pool2', 'bn_conv1');

% === Decoder 2 ===
% TransConv up
lgraph = addLayers(lgraph, ...
    transposedConv3dLayer(2, nf*2, 'Stride', 2, 'Name', 'up2'));
lgraph = connectLayers(lgraph, 'bn_relu2', 'up2');

% Concatenation with enc2
lgraph = addLayers(lgraph, concatenationLayer(4, 2, 'Name', 'concat2'));
lgraph = connectLayers(lgraph, 'up2', 'concat2/in1');
lgraph = connectLayers(lgraph, 'enc2_relu2', 'concat2/in2');

dec2 = [
    convolution3dLayer(3, nf*2, 'Padding', 'same', 'Name', 'dec2_conv1')
    batchNormalizationLayer('Name', 'dec2_bn1')
    reluLayer('Name', 'dec2_relu1')
    convolution3dLayer(3, nf*2, 'Padding', 'same', 'Name', 'dec2_conv2')
    batchNormalizationLayer('Name', 'dec2_bn2')
    reluLayer('Name', 'dec2_relu2')
    ];
lgraph = addLayers(lgraph, dec2);
lgraph = connectLayers(lgraph, 'concat2', 'dec2_conv1');

% === Decoder 1 ===
lgraph = addLayers(lgraph, ...
    transposedConv3dLayer(2, nf, 'Stride', 2, 'Name', 'up1'));
lgraph = connectLayers(lgraph, 'dec2_relu2', 'up1');

lgraph = addLayers(lgraph, concatenationLayer(4, 2, 'Name', 'concat1'));
lgraph = connectLayers(lgraph, 'up1', 'concat1/in1');
lgraph = connectLayers(lgraph, 'enc1_relu2', 'concat1/in2');

dec1 = [
    convolution3dLayer(3, nf, 'Padding', 'same', 'Name', 'dec1_conv1')
    batchNormalizationLayer('Name', 'dec1_bn1')
    reluLayer('Name', 'dec1_relu1')
    convolution3dLayer(3, nf, 'Padding', 'same', 'Name', 'dec1_conv2')
    batchNormalizationLayer('Name', 'dec1_bn2')
    reluLayer('Name', 'dec1_relu2')
    ];
lgraph = addLayers(lgraph, dec1);
lgraph = connectLayers(lgraph, 'concat1', 'dec1_conv1');

% === Output ===
output_layers = [
    convolution3dLayer(1, 1, 'Name', 'output_conv')
    regressionLayer('Name', 'output')
    ];
lgraph = addLayers(lgraph, output_layers);
lgraph = connectLayers(lgraph, 'dec1_relu2', 'output_conv');

end

%% =========================================================================
% Patch-based 推理
% =========================================================================
function chi = patch_based_inference_unet(net, input_norm, Mask, N, patch_size, norm_out)
% 将体积切分为重叠 patch，逐个推理后加权拼接

stride = patch_size / 2;  % 50% 重叠

chi_accum = zeros(N);
weight_accum = zeros(N);

% 3D Hanning 窗
win = hanning_3d(patch_size);

% 计算起始位置
starts = cell(3, 1);
for d = 1:3
    s = 1:stride(d):(N(d) - patch_size(d) + 1);
    if isempty(s)
        s = 1;
    end
    % 确保覆盖末尾
    if s(end) + patch_size(d) - 1 < N(d)
        s = [s, N(d) - patch_size(d) + 1]; %#ok<AGROW>
    end
    starts{d} = unique(s);
end

total_patches = numel(starts{1}) * numel(starts{2}) * numel(starts{3});
fprintf('    推理 patch 总数: %d\n', total_patches);

count = 0;
for i1 = starts{1}
    for i2 = starts{2}
        for i3 = starts{3}
            
            idx1 = i1:i1+patch_size(1)-1;
            idx2 = i2:i2+patch_size(2)-1;
            idx3 = i3:i3+patch_size(3)-1;
            
            patch_in = input_norm(idx1, idx2, idx3);
            
            % 跳过几乎全零的 patch
            if max(abs(patch_in(:))) < 1e-6
                continue;
            end
            
            % [H W D 1 1] for predict
            patch_in_5d = reshape(single(patch_in), [patch_size, 1, 1]);
            
            try
                patch_out = predict(net, patch_in_5d);
                patch_out = double(squeeze(patch_out));
            catch
                % dlarray 方式
                try
                    patch_out = predict(net, dlarray(single(patch_in_5d), 'SSSBC'));
                    patch_out = double(extractdata(squeeze(patch_out)));
                catch
                    continue;
                end
            end
            
            if ~isequal(size(patch_out), patch_size)
                patch_out = reshape(patch_out, patch_size);
            end
            
            chi_accum(idx1, idx2, idx3) = chi_accum(idx1, idx2, idx3) + patch_out .* win;
            weight_accum(idx1, idx2, idx3) = weight_accum(idx1, idx2, idx3) + win;
            
            count = count + 1;
        end
    end
end

fprintf('    实际推理 patch: %d\n', count);

weight_accum(weight_accum < eps) = eps;
chi = chi_accum ./ weight_accum;

% 反归一化
chi = chi * norm_out;
chi = chi .* Mask;

end

%% =========================================================================
function win = hanning_3d(patch_size)
% 3D Hanning 窗用于 patch 拼接去伪影

w1 = hann(patch_size(1));
w2 = hann(patch_size(2));
w3 = hann(patch_size(3));

win = w1(:) .* w2(:).';
win = reshape(win, [patch_size(1), patch_size(2), 1]) .* reshape(w3, [1, 1, patch_size(3)]);

end