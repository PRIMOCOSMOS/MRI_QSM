function chi = dl_xqsm(input_norm, Mask, N, voxel_size, norm_factor, cfg)
% dl_xqsm.m
% xQSM 推理模块（强制 .pth + Python bridge）
%
% 说明:
% - 不使用 ONNX
% - 不使用 MATLAB .mat 网络
% - 通过 dl_python_bridge 调用独立 python 脚本

chi = [];

fprintf('  xQSM: 使用 .pth 权重 + 独立 Python bridge 脚本推理\n');

chi = dl_python_bridge('xqsm', input_norm, Mask, N, voxel_size, norm_factor, cfg);

if isempty(chi)
    error('xQSM 推理未返回结果。请检查 Python 环境、xQSM repo 路径和 checkpoint。');
end

chi = double(chi);
chi(~Mask) = 0;

end