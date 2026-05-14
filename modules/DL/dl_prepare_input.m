function [input_norm, norm_factor] = dl_prepare_input(local_field, Mask)
% dl_prepare_input.m
% 深度学习模型的公共输入预处理
%
% QSMnet+ 和 xQSM 均期望输入为归一化的局部场:
%   input = local_field / norm_factor
%   其中 norm_factor 使得 mask 内数据的动态范围适合网络
%
% 常用归一化策略:
%   - QSMnet+: 除以 mask 内标准差的某个倍数 (论文中用 mean+3*std 截断)
%   - xQSM: 除以 P99.5 百分位数
%
% 这里使用兼容两者的策略: 除以 mask 内绝对值的 99.5 百分位

Mask = logical(Mask);
vals = local_field(Mask);
vals = vals(isfinite(vals));

p995 = prctile(abs(vals), 99.5);

if p995 <= 0 || ~isfinite(p995)
    norm_factor = 1;
else
    norm_factor = p995;
end

input_norm = local_field / norm_factor;

% 截断极端值
input_norm(input_norm > 3) = 3;
input_norm(input_norm < -3) = -3;

% mask 外置零
input_norm(~Mask) = 0;

fprintf('DL 输入预处理: norm_factor = %.6g ppm\n', norm_factor);
fprintf('  归一化后范围: [%.3f, %.3f]\n', min(input_norm(Mask)), max(input_norm(Mask)));

end