function [local_field, eroded_mask] = bg_removal_vsharp(total_field, Mask, voxel_size, radii)

Mask = logical(Mask);
N = size(Mask);

radii = sort(radii(:).', 'descend');
radius = radii(1);

fprintf('  使用内置 V-SHARP，radius = %.2f mm。\n', radius);

smv_kernel = create_smv_kernel(N, voxel_size, radius);
eroded_mask = erode_mask_sphere(Mask, voxel_size, radius);

filtered = real(ifftn(fftn(double(total_field) .* double(Mask)) .* smv_kernel));
mask_filtered = real(ifftn(fftn(double(Mask)) .* smv_kernel));
mask_filtered(mask_filtered < 0.1) = 0.1;

local_field = (double(total_field) - filtered ./ mask_filtered) .* double(eroded_mask);
local_field(~eroded_mask) = 0;

fprintf('  V-SHARP 完成，侵蚀后 mask 体素数: %d\n', nnz(eroded_mask));

end

%% =========================================================================
% 内置 LBV fallback
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

