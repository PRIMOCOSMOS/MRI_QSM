function local_field = bg_removal_pdf_medi(iFreq, Mask, matrix_size, voxel_size, B0_dir)
% MEDI toolbox README:
%   RDF = PDF(iFreq, N_std, Mask, matrix_size, voxel_size, B0_dir);
%
% 注意:
%   iFreq 为解缠总场，单位 rad/echo。
%   这里结果仅用于背景场去除对比，不作为最终 QSM 反演输入。

Mask = logical(Mask);
iFreq = double(iFreq);
iFreq(~Mask) = 0;

N_std = ones(matrix_size);

if exist('PDF', 'file') ~= 2
    error('未找到 MEDI toolbox PDF 函数。');
end

fprintf('  使用 MEDI toolbox PDF 标准接口:\n');
fprintf('    RDF = PDF(iFreq, N_std, Mask, matrix_size, voxel_size, B0_dir)\n');

local_field = PDF(iFreq, N_std, Mask, matrix_size, voxel_size, B0_dir);
local_field = double(local_field);
local_field(~Mask) = 0;

if ~is_valid_volume(local_field, Mask)
    error('MEDI PDF 返回结果无效或接近全 0。');
end

fprintf('  MEDI PDF 调用成功。\n');

end

%% =========================================================================
% LBV: 使用 MEDI README 标准接口
% =========================================================================

function tf = is_valid_volume(vol, Mask)

if isempty(vol)
    tf = false;
    return;
end

Mask = logical(Mask);

if ~isequal(size(vol), size(Mask))
    tf = false;
    return;
end

v = double(vol(Mask));
v = v(isfinite(v));

if isempty(v)
    tf = false;
    return;
end

tf = any(abs(v) > 1e-12) && std(v) > 1e-12;

end
