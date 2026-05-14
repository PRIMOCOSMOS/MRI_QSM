function data = mod_phase_unwrap(data, cfg)
% mod_phase_unwrap.m
% 相位解缠模块
%
% QSM2016 数据集已提供 phs_unwrap 和 phs_tissue，
% 此模块主要用于验证和记录。
% 如果需要从 phs_wrap 重新解缠，可使用 Laplacian unwrapping。

fprintf('QSM2016 数据已提供解缠相位 (phs_unwrap) 和局部场 (phs_tissue)\n');
fprintf('phs_tissue 单位为 ppm，可直接用于偶极子反演\n');

% 验证: phs_tissue 是否合理
v = data.phs_tissue(data.Mask);
fprintf('phs_tissue 范围: [%.4f, %.4f] ppm\n', min(v), max(v));
fprintf('phs_tissue std : %.4f ppm\n', std(v));

% 如果需要从 phs_wrap 重新做 Laplacian unwrapping:
do_laplacian = isfield(cfg, 'phase_unwrap') && ...
    isfield(cfg.phase_unwrap, 'enable') && cfg.phase_unwrap.enable;
if do_laplacian
    fprintf('执行 Laplacian phase unwrapping...\n');
    data.phs_laplacian = laplacian_unwrap(data.phs_wrap, data.Mask, data.spatial_res);
end

end

function phs_unwrapped = laplacian_unwrap(phs_wrap, Mask, ~)
% Laplacian phase unwrapping
% 基于 Schofield & Zhu, Optics Letters 2003
%
% 原理: ∇²φ_unwrapped = Re{∇²[exp(iφ)] / exp(iφ)}

N = size(Mask);

% 构建 Laplacian 算子 (k-space)
[k1, k2, k3] = ndgrid(0:N(1)-1, 0:N(2)-1, 0:N(3)-1);
lap_kernel = -2 * (cos(2*pi*k1/N(1)) + cos(2*pi*k2/N(2)) + cos(2*pi*k3/N(3)) - 3);

% 避免除零
lap_kernel(lap_kernel == 0) = eps;

% 计算 Laplacian of wrapped phase (通过复数)
complex_phase = exp(1i * phs_wrap);
lap_phase = real(ifftn(lap_kernel .* fftn(complex_phase)) ./ (complex_phase + eps));

% 求解 Poisson 方程
phs_unwrapped = real(ifftn(fftn(lap_phase) ./ lap_kernel));
phs_unwrapped = phs_unwrapped .* Mask;

end