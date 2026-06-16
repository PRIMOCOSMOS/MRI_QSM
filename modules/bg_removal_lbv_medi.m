function [local_field, lbv_mask] = bg_removal_lbv_medi(iFreq, Mask, matrix_size, voxel_size, tol, peel)
% bg_removal_lbv_medi
%
% 修正版 LBV 背景场去除调用函数。
%
% 修复目标:
%   1. 优先通过 SEPIA 的 LBV wrapper 调用 LBV，因为 SEPIA 对 MEDI/LBV
%      的接口、depth、peel、mask 处理做了兼容封装。
%
%   2. 若 SEPIA wrapper 不可用，再尝试 MEDI toolbox 的 LBV。
%
%   3. MEDI README 中的最简调用是:
%          RDF = LBV(iFreq, Mask, matrix_size, voxel_size, tol);
%
%      但实际 MEDI toolbox 版本中 LBV 往往还支持:
%          RDF = LBV(iFreq, Mask, matrix_size, voxel_size, tol, depth, peel);
%
%      因此这里先尝试 7 参数接口，再尝试 5 参数接口。
%
%   4. 修正 "转换为逻辑值时，输出只能包含 true 或 false" 的常见诱因:
%      - iFreq 内存在 NaN/Inf
%      - Mask 类型或取值不符合 LBV 内部预期
%      - matrix_size / voxel_size 不是标准 double row vector
%      - SEPIA/MEDI 路径冲突导致调用到非预期 LBV
%
%   5. 只有在 SEPIA 和 MEDI LBV 都失败时，才使用内置 fallback。

%% ------------------------------------------------------------------------
% 输入规范化
% -------------------------------------------------------------------------
Mask = logical(Mask);

matrix_size = double(matrix_size(:).');
voxel_size  = double(voxel_size(:).');

if nargin < 5 || isempty(tol)
    tol = 0.005;
end

if nargin < 6 || isempty(peel)
    peel = 0;
end

tol  = double(tol);
peel = double(peel);

iFreq = double(iFreq);

% 非有限值会导致 MEDI/LBV 内部 logical 判断或多重网格求解异常
iFreq(~isfinite(iFreq)) = 0;
iFreq(~Mask) = 0;

% 输出 mask
lbv_mask = erode_mask_safe(Mask, peel);

% LBV depth：SEPIA 默认通常为 5；但对于当前数据，从日志看 FMG depth=4
% 这里使用保守自动估计，避免过深 multigrid 导致奇异/逻辑判断异常。
depth = max(1, floor(log2(min(matrix_size))) - 2);

fprintf('  LBV 输入检查:\n');
fprintf('    matrix_size = [%d %d %d]\n', matrix_size(1), matrix_size(2), matrix_size(3));
fprintf('    voxel_size  = [%.4g %.4g %.4g] mm\n', voxel_size(1), voxel_size(2), voxel_size(3));
fprintf('    tol         = %.4g\n', tol);
fprintf('    depth       = %d\n', depth);
fprintf('    peel        = %d\n', peel);

%% ------------------------------------------------------------------------
% 优先尝试 SEPIA LBV wrapper
% -------------------------------------------------------------------------
if exist('sepia_addpath', 'file') == 2
    try
        sepia_addpath;
        fprintf('  已调用 sepia_addpath。\n');
    catch ME
        fprintf('  sepia_addpath 调用失败: %s\n', ME.message);
    end
end

sepiaWrapperCandidates = { ...
    'Wrapper_BFR_LBV', ...
    'Wrapper_BFR_LBV_MEDI', ...
    'Wrapper_BFR_LBV_4sepia'};

for iw = 1:numel(sepiaWrapperCandidates)

    wrapperName = sepiaWrapperCandidates{iw};

    if exist(wrapperName, 'file') ~= 2
        continue;
    end

    fprintf('  尝试使用 SEPIA LBV wrapper: %s\n', wrapperName);

    try
        algorParam = struct();

        algorParam.bfr.method              = 'LBV';
        algorParam.bfr.tol                 = tol;
        algorParam.bfr.depth               = depth;
        algorParam.bfr.peel                = peel;
        algorParam.bfr.erode_radius        = 0;
        algorParam.bfr.erode_before_radius = 0;
        algorParam.bfr.refine_method       = 'None';
        algorParam.bfr.refine_order        = 0;

        headerAndExtraData = struct();
        headerAndExtraData.b0dir       = [0 0 1];
        headerAndExtraData.B0_dir      = [0 0 1];
        headerAndExtraData.matrixSize  = matrix_size;
        headerAndExtraData.matrix_size = matrix_size;
        headerAndExtraData.voxelSize   = voxel_size;
        headerAndExtraData.voxel_size  = voxel_size;

        wrapperFunc = str2func(wrapperName);

        try
            [local_field_tmp, lbv_mask_tmp] = wrapperFunc( ...
                iFreq, Mask, matrix_size, voxel_size, algorParam, headerAndExtraData);
        catch
            local_field_tmp = wrapperFunc( ...
                iFreq, Mask, matrix_size, voxel_size, algorParam, headerAndExtraData);
            lbv_mask_tmp = lbv_mask;
        end

        local_field_tmp = double(local_field_tmp);
        local_field_tmp(~isfinite(local_field_tmp)) = 0;

        if isempty(lbv_mask_tmp)
            lbv_mask_tmp = lbv_mask;
        end
        lbv_mask_tmp = logical(lbv_mask_tmp);

        local_field_tmp(~lbv_mask_tmp) = 0;

        if is_valid_volume(local_field_tmp, lbv_mask_tmp)
            local_field = local_field_tmp;
            lbv_mask = lbv_mask_tmp;

            fprintf('  SEPIA LBV wrapper 调用成功。\n');
            fprintf('  LBV 完成，mask 体素数: %d\n', nnz(lbv_mask));
            return;
        else
            fprintf('  SEPIA LBV wrapper 返回结果无效，继续尝试其他接口。\n');
        end

    catch ME
        fprintf('  SEPIA LBV wrapper 失败: %s\n', ME.message);
    end
end

%% ------------------------------------------------------------------------
% 尝试 MEDI toolbox LBV
% -------------------------------------------------------------------------
if exist('LBV', 'file') ~= 2
    fprintf('  未找到 MEDI toolbox LBV，使用内置 LBV fallback。\n');
    local_field = lbv_builtin(iFreq, Mask, voxel_size, tol, peel);
    local_field(~lbv_mask) = 0;
    fprintf('  LBV 完成，mask 体素数: %d\n', nnz(lbv_mask));
    return;
end

fprintf('  当前 MATLAB path 中的 LBV:\n');
try
    disp(which('LBV', '-all'));
catch
    disp(which('LBV'));
end

% MEDI LBV 内部有些版本对 mask 类型敏感，这里用 double 0/1 更稳妥
Mask_medi = double(Mask);

%% ------------------------------------------------------------------------
% MEDI LBV: 7 参数接口
% -------------------------------------------------------------------------
fprintf('  尝试 MEDI LBV 7 参数接口:\n');
fprintf('    RDF = LBV(iFreq, double(Mask), matrix_size, voxel_size, tol, depth, peel)\n');

try
    local_field = LBV(iFreq, Mask_medi, matrix_size, voxel_size, tol, depth, peel);
    local_field = double(local_field);
    local_field(~isfinite(local_field)) = 0;
    local_field(~lbv_mask) = 0;

    if is_valid_volume(local_field, lbv_mask)
        fprintf('  MEDI LBV 7 参数接口调用成功。\n');
        fprintf('  LBV 完成，mask 体素数: %d\n', nnz(lbv_mask));
        return;
    else
        fprintf('  MEDI LBV 7 参数接口返回结果无效。\n');
    end

catch ME
    fprintf('  MEDI LBV 7 参数接口失败: %s\n', ME.message);
end

%% ------------------------------------------------------------------------
% MEDI LBV: README 5 参数接口
% -------------------------------------------------------------------------
fprintf('  尝试 MEDI LBV README 5 参数接口:\n');
fprintf('    RDF = LBV(iFreq, double(Mask), matrix_size, voxel_size, tol)\n');

try
    local_field = LBV(iFreq, Mask_medi, matrix_size, voxel_size, tol);
    local_field = double(local_field);
    local_field(~isfinite(local_field)) = 0;
    local_field(~lbv_mask) = 0;

    if is_valid_volume(local_field, lbv_mask)
        fprintf('  MEDI LBV 5 参数接口调用成功。\n');
        fprintf('  LBV 完成，mask 体素数: %d\n', nnz(lbv_mask));
        return;
    else
        fprintf('  MEDI LBV 5 参数接口返回结果无效。\n');
    end

catch ME
    fprintf('  MEDI LBV 5 参数接口失败: %s\n', ME.message);
end

%% ------------------------------------------------------------------------
% 最终 fallback
% -------------------------------------------------------------------------
fprintf('  SEPIA/MEDI LBV 均失败，使用内置 LBV fallback。\n');

local_field = lbv_builtin(iFreq, Mask, voxel_size, tol, peel);
local_field(~isfinite(local_field)) = 0;
local_field(~lbv_mask) = 0;

if ~is_valid_volume(local_field, lbv_mask)
    warning('LBV fallback 结果仍无效，请检查 iFreq/Mask 是否存在严重异常。');
end

fprintf('  LBV 完成，mask 体素数: %d\n', nnz(lbv_mask));

end

%% =========================================================================
% V-SHARP 内置实现
% =========================================================================

function local_field = lbv_builtin(total_field, Mask, voxel_size, tol, peel)

Mask = logical(Mask);
N = size(Mask);

if nargin < 5 || isempty(peel)
    peel = 0;
end

lbv_mask = erode_mask_safe(Mask, peel);

[k1, k2, k3] = ndgrid(0:N(1)-1, 0:N(2)-1, 0:N(3)-1);

lap = 2 * ( ...
    (cos(2*pi*k1/N(1)) - 1) / voxel_size(1)^2 + ...
    (cos(2*pi*k2/N(2)) - 1) / voxel_size(2)^2 + ...
    (cos(2*pi*k3/N(3)) - 1) / voxel_size(3)^2 );

lap(abs(lap) < eps) = eps;

field = double(total_field);
field(~Mask) = 0;

lap_total = real(ifftn(lap .* fftn(field)));
rhs = lap_total .* double(lbv_mask);

local_field = real(ifftn(fftn(rhs) ./ lap));
local_field = local_field .* double(lbv_mask);

v = local_field(lbv_mask);
v = v(isfinite(v));
if ~isempty(v)
    local_field(lbv_mask) = local_field(lbv_mask) - mean(v);
end

local_field(~lbv_mask) = 0;

if nargin >= 4 && ~isempty(tol)
    local_field(abs(local_field) < tol * eps) = 0;
end

end

%% =========================================================================
% 辅助函数
% =========================================================================

function smv_kernel = create_smv_kernel(N, voxel_size, radius)

[y, x, z] = ndgrid( ...
    (-N(1)/2:N(1)/2-1) * voxel_size(1), ...
    (-N(2)/2:N(2)/2-1) * voxel_size(2), ...
    (-N(3)/2:N(3)/2-1) * voxel_size(3));

sphere = (x.^2 + y.^2 + z.^2) <= radius^2;
sphere = sphere / max(sum(sphere(:)), eps);

smv_kernel = fftn(fftshift(sphere));

end

function eroded = erode_mask_sphere(Mask, voxel_size, radius)

r_vox = ceil(radius ./ voxel_size);
r = max(1, min(r_vox));
eroded = erode_mask_safe(Mask, r);

end

function eroded = erode_mask_safe(Mask, r)

Mask = logical(Mask);

if nargin < 2 || isempty(r) || r <= 0
    eroded = Mask;
    return;
end

try
    se = strel('sphere', r);
    eroded = imerode(Mask, se);
catch
    [x, y, z] = ndgrid(-r:r, -r:r, -r:r);
    se = (x.^2 + y.^2 + z.^2) <= r^2;
    cnt = convn(double(Mask), double(se), 'same');
    eroded = cnt >= sum(se(:));
end

eroded = logical(eroded);

end

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
