function kernel = create_dipole_kernel(N, spatial_res, B0_dir)
% create_dipole_kernel.m
% 创建与 MATLAB fftn/ifftn 顺序一致的 QSM k-space 偶极子核。
%
% D(k) = 1/3 - (k · B0_dir)^2 / |k|^2
%
% 关键修正:
%   之前版本使用 fftshift 后的 k-space kernel，但后续反演中直接与 fftn(vol)
%   相乘，这会造成 k-space 顺序不匹配。
%
%   本版本使用 ifftshift 构造 k 轴，使 kernel 与 fftn 输出顺序一致。
%
% 输入:
%   N           - [Nx Ny Nz]
%   spatial_res - [dx dy dz] mm
%   B0_dir      - 可选，B0方向，默认 [0 0 1]
%
% 输出:
%   kernel      - [Nx Ny Nz]，与 fftn 顺序一致

if nargin < 3 || isempty(B0_dir)
    B0_dir = [0 0 1];
end

N = double(N(:).');
spatial_res = double(spatial_res(:).');
B0_dir = double(B0_dir(:).');
B0_dir = B0_dir / max(norm(B0_dir), eps);

Nx = N(1);
Ny = N(2);
Nz = N(3);

dx = spatial_res(1);
dy = spatial_res(2);
dz = spatial_res(3);

% 与 fftn 输出顺序一致的频率坐标:
% 对偶数 N: [0, 1, ..., N/2-1, -N/2, ..., -1] / FOV
kx_vec = ifftshift((-floor(Nx/2):ceil(Nx/2)-1) / (Nx * dx));
ky_vec = ifftshift((-floor(Ny/2):ceil(Ny/2)-1) / (Ny * dy));
kz_vec = ifftshift((-floor(Nz/2):ceil(Nz/2)-1) / (Nz * dz));

[kx, ky, kz] = ndgrid(kx_vec, ky_vec, kz_vec);

k2 = kx.^2 + ky.^2 + kz.^2;
kdotB0 = kx * B0_dir(1) + ky * B0_dir(2) + kz * B0_dir(3);

kernel = zeros(Nx, Ny, Nz);

idx = k2 > 0;
kernel(idx) = 1/3 - (kdotB0(idx).^2 ./ k2(idx));

% DC 点设为 0，避免引入任意常数偏置
kernel(~idx) = 0;

% 防御性检查：如果 kernel 退化为常数，直接报错
if std(kernel(:)) < 1e-8
    error(['create_dipole_kernel: kernel 退化为近似常数。', ...
           '请检查 N、spatial_res、B0_dir。']);
end

end