function result = snu_chisep_optimization_adapter(data, chi_total_ppm, R2star_Hz, localField_Hz, Mask, cfg, outDir)
% snu_chisep_optimization_adapter.m
% ============================================================================
% TRADITIONAL (convex-optimization) chi-separation, ONNX-free.
%
% Purpose: provide a transparent, dependency-light optimization-based
% χ-separation that can be compared head-to-head with the deep-learning
% χ-sepnet (snu_chisep_onnxruntime_adapter). It does NOT call any .onnx model
% and does NOT touch MATLAB's broken onnxmex, so it always runs.
%
% Model (Shin et al., Neuroimage 2021; same contrasts as MEDI+0 / iLSQR):
%   Let  x_pos = χ_para (>=0),  x_neg = |χ_dia| (>=0).
%   Field model (ppm):       d * (x_pos - x_neg) = local_field_ppm
%   Relaxation model (ppm):  (x_pos + x_neg)     = R2'/Dr  (= r2prime_ppm)
%   i.e. r2prime_ppm is ALREADY the susceptibility magnitude sum, so the data
%   term is ||(x_pos+x_neg) - r2prime_ppm||^2 (NO extra Dr factor).
%
% We solve, per source map, a regularized least squares (closed form via FFT,
% Tikhonov 'iLSQR-like'  OR  iterative L1 'MEDI-like'):
%   min_{x_pos,x_neg>=0}
%       || d*(x_pos - x_neg) - f ||_2^2
%     + w_r2 * || Dr*(x_pos + x_neg) - R2' ||_2^2
%     + lambda * R(x_pos, x_neg)
% with R = L2 (iLSQR-like, fast closed-ish form) or L1-TV (MEDI-like, ADMM).
%
% This is a faithful, self-contained implementation of the χ-separation
% forward model. It is meant as a reference/baseline for comparison, not a
% bit-exact reproduction of the SNU p-code.
%
% Signature identical to the other adapters so it plugs into
% mod_susceptibility_separation via cfg.sep.adapter_function.
%
% Relevant cfg fields:
%   cfg.sep.opt_method     'iLSQR' (L2, default) | 'MEDI' (L1-TV)
%   cfg.sep.opt_lambda     regularization weight (default 1e-2)
%   cfg.sep.opt_w_r2       weight of the R2' data term (default 1.0)
%   cfg.sep.opt_maxiter    iterations for MEDI/ADMM (default 100)
%   cfg.sep.snu_Dr         Dr (default 114)
%   cfg.sep.snu_local_field_mode  'forward_from_whqsm' | 'measured'
% ============================================================================

result = struct();
Mask = logical(Mask);
N = size(Mask);

method   = upper(char(get_cfg(cfg, {'sep','opt_method'}, 'iLSQR')));
lambda   = double(get_cfg(cfg, {'sep','opt_lambda'}, 1e-2));
w_r2     = double(get_cfg(cfg, {'sep','opt_w_r2'}, 1.0));
maxiter  = double(get_cfg(cfg, {'sep','opt_maxiter'}, 100));
Dr       = double(get_cfg(cfg, {'sep','snu_Dr'}, 114));

voxel_size = double(data.spatial_res(:).');
B0_dir = double(data.B0_dir(:).'); B0_dir = B0_dir ./ max(norm(B0_dir), eps);

% ---- local field (ppm) ----
localFieldMode = lower(char(get_cfg(cfg, {'sep','snu_local_field_mode'}, 'forward_from_whqsm')));
switch localFieldMode
    case {'forward_from_whqsm','forward','qsm_forward'}
        % forward field (ppm) directly from WH-QSM chi_total
        D = dipole_kernel(N, voxel_size, B0_dir);
        chi = double(chi_total_ppm); chi(~Mask) = 0;
        f_ppm = real(ifftn(D .* fftn(chi)));
    case {'measured','fieldmap','dicom'}
        % localField_Hz -> ppm
        CF = double(data.B0) * 42.576e6;
        f_ppm = double(localField_Hz) / CF * 1e6;
        D = dipole_kernel(N, voxel_size, B0_dir);
    otherwise
        error('Unknown cfg.sep.snu_local_field_mode: %s', localFieldMode);
end
f_ppm(~Mask) = 0;

% ---- R2' in ppm ----
R2star_Hz = double(R2star_Hz); R2star_Hz(~Mask) = 0;
r2_hz = [];
if isfield(data,'R2_Hz') && ~isempty(data.R2_Hz)
    r2_hz = double(data.R2_Hz); r2_hz(~Mask) = 0;
end
% Highest priority: an externally supplied R2' map already in ppm. This is used
% for FAIR comparison with onnx, so opt uses the SAME R2' that R2PRIMEnet
% produced (instead of raw R2*, which would inflate opt by the R2 baseline).
ext_r2prime_ppm = get_cfg(cfg, {'sep','opt_r2prime_ppm'}, []);
r2prime_source = '';
if ~isempty(ext_r2prime_ppm) && isequal(size(ext_r2prime_ppm), size(Mask))
    r2prime_ppm = double(ext_r2prime_ppm);
    r2prime_source = 'external_ppm (shared with onnx, fair)';
elseif ~isempty(r2_hz)
    r2prime_hz = max(R2star_Hz - r2_hz, 0);
    r2prime_ppm = r2prime_hz / Dr;
    r2prime_source = 'measured R2''=R2*-R2';
else
    % no SE R2 and no external R2': fall back to raw R2* (pseudo). This
    % OVER-estimates R2' by the R2 baseline (~0.13 ppm) and inflates opt.
    % Prefer passing cfg.sep.opt_r2prime_ppm for a fair comparison.
    r2prime_ppm = R2star_Hz / Dr;
    r2prime_source = 'pseudo R2* (INFLATED; no R2 and no external R2'')';
end
r2prime_ppm(~Mask) = 0;

fprintf('\nTraditional (optimization) chi-separation\n');
fprintf('  method     : %s\n', method);
fprintf('  lambda     : %.4g , w_r2 : %.4g\n', lambda, w_r2);
fprintf('  Dr         : %g , local field: %s\n', Dr, localFieldMode);
fprintf('  spatial_res: [%.4g %.4g %.4g] mm\n', voxel_size(1), voxel_size(2), voxel_size(3));
fprintf('  R2'' source : %s\n', r2prime_source);
% Diagnostic stats of the R2' input that drives the magnitude (x_pos+x_neg).
rr = r2prime_ppm(Mask); rr = rr(isfinite(rr));
if ~isempty(rr)
    fprintf('  R2''(ppm) in-mask: median=%.4f p99=%.4f  (= expected x_pos+x_neg level)\n', ...
        median(rr), prctile(rr,99));
end

% ---- solve ----
switch method
    case 'ILSQR'   % L2 Tikhonov, closed-form in k-space per the 2x2 system
        [x_pos, x_neg] = solve_l2(D, f_ppm, r2prime_ppm, Dr, w_r2, lambda, Mask);
    case 'MEDI'    % L1-TV via ADMM
        [x_pos, x_neg] = solve_l1_tv(D, f_ppm, r2prime_ppm, Dr, w_r2, lambda, maxiter, voxel_size, Mask);
    otherwise
        error('Unknown cfg.sep.opt_method: %s (use iLSQR or MEDI)', method);
end

x_pos = max(x_pos, 0); x_neg = max(x_neg, 0);
x_pos(~Mask) = 0; x_neg(~Mask) = 0;

% Pipeline convention: chi_para>=0, chi_dia<=0
chi_para = x_pos;
chi_dia  = -x_neg;
x_tot    = chi_para - x_neg;

result.method     = sprintf('Optimization_%s', method);
result.backend    = 'matlab_convex_optimization';
result.chi_para   = chi_para;
result.chi_dia    = chi_dia;
result.x_tot_raw  = x_tot;
result.x_para_raw = x_pos;
result.x_dia_raw  = x_neg;
result.qsm_map    = double(chi_total_ppm) .* Mask;   % reference QSM (WH-QSM)
result.r2p_map    = r2prime_ppm;
result.Dr         = Dr;
result.lambda     = lambda;
result.w_r2       = w_r2;
result.opt_method = method;
result.pseudo_r2prime = isempty(r2_hz) && isempty(ext_r2prime_ppm);
result.r2prime_source = r2prime_source;
result.local_field_mode = localFieldMode;

save(fullfile(outDir, sprintf('chisep_optimization_%s_raw.mat', method)), ...
    'x_pos','x_neg','chi_para','chi_dia','x_tot','result','-v7.3');
end

%% ========================================================================
function [x_pos, x_neg] = solve_l2(D, f, r2p, Dr, w_r2, lambda, Mask)
% Closed-form L2 solution. Work in k-space for the field term, image space for
% the (local) R2' term and the Tikhonov regularizer.
% Decision variables: x_pos, x_neg. Use change of variables:
%   s = x_pos - x_neg   (susceptibility "signed", couples to field via D)
%   t = x_pos + x_neg   (magnitude sum, couples to R2')
% Then x_pos=(s+t)/2, x_neg=(t-s)/2.
% Objective decouples:
%   s: min || D s - f ||^2 + lambda || s ||^2        -> Wiener/Tikhonov
%   t: min w_r2 || t - r2p ||^2 + lambda || t ||^2   -> scalar per voxel
% NOTE: r2p (== r2prime_ppm) is already (x_pos+x_neg) in ppm; no Dr here.
Fk = fftn(f);
Dk = D;
S = (conj(Dk).*Fk) ./ (abs(Dk).^2 + lambda);
s = real(ifftn(S));
% t closed form per voxel
t = (w_r2 .* r2p) ./ (w_r2 + lambda);
x_pos = 0.5*(s + t);
x_neg = 0.5*(t - s);
x_pos(~Mask)=0; x_neg(~Mask)=0;
end

%% ========================================================================
function [x_pos, x_neg] = solve_l1_tv(D, f, r2p, Dr, w_r2, lambda, maxiter, vox, Mask)
% MEDI-like: L1 total-variation regularization on both source maps via ADMM.
% Same s/t split as L2 for the data terms; TV applied to x_pos and x_neg.
% This is a compact ADMM; for a baseline comparison it is sufficient.
N = size(f);
mu = 0.05;                      % ADMM penalty
% gradient operators (forward diff) in k-space
[Ex,Ey,Ez] = grad_kernels(N, vox);
E2 = abs(Ex).^2 + abs(Ey).^2 + abs(Ez).^2;
Dk = D;

% init from L2
[x_pos, x_neg] = solve_l2(D, f, r2p, Dr, w_r2, lambda*1, Mask);

% ADMM variables for TV of s = x_pos - x_neg and t = x_pos + x_neg
s = x_pos - x_neg; t = x_pos + x_neg;
zx=zeros(N); zy=zeros(N); zz=zeros(N); ux=zeros(N); uy=zeros(N); uz=zeros(N); %#ok<NASGU>

Fk = fftn(f);
for it = 1:maxiter
    % ---- s update (field term + TV) ----
    rhs = conj(Dk).*Fk + mu*(div_k(Ex,Ey,Ez, fftn(zx-ux), fftn(zy-uy), fftn(zz-uz)));
    Sden = abs(Dk).^2 + mu*E2 + lambda;
    s = real(ifftn(rhs ./ Sden));
    % ---- TV shrinkage on grad(s) ----
    gx = real(ifftn(Ex.*fftn(s))); gy = real(ifftn(Ey.*fftn(s))); gz = real(ifftn(Ez.*fftn(s)));
    [zx,zy,zz] = shrink3(gx+ux, gy+uy, gz+uz, lambda/mu);
    ux = ux + gx - zx; uy = uy + gy - zy; uz = uz + gz - zz;
    % ---- t update (R2' term, closed form; r2p already = x_pos+x_neg ppm) ----
    t = (w_r2 .* r2p) ./ (w_r2 + lambda);
end
x_pos = 0.5*(s + t);
x_neg = 0.5*(t - s);
x_pos(~Mask)=0; x_neg(~Mask)=0;
end

%% ---- helpers ----
function D = dipole_kernel(N, vox, B0_dir)
if exist('create_dipole_kernel','file') == 2
    D = create_dipole_kernel(N, vox, B0_dir);
    return;
end
kx = ifftshift((-floor(N(1)/2):ceil(N(1)/2)-1)/(N(1)*vox(1)));
ky = ifftshift((-floor(N(2)/2):ceil(N(2)/2)-1)/(N(2)*vox(2)));
kz = ifftshift((-floor(N(3)/2):ceil(N(3)/2)-1)/(N(3)*vox(3)));
[KX,KY,KZ]=ndgrid(kx,ky,kz);
k2 = KX.^2+KY.^2+KZ.^2; kdot = KX*B0_dir(1)+KY*B0_dir(2)+KZ*B0_dir(3);
D = zeros(N); idx = k2>0; D(idx) = 1/3 - (kdot(idx).^2 ./ k2(idx));
end

function [Ex,Ey,Ez] = grad_kernels(N, vox)
% forward difference in k-space
[kx,ky,kz] = ndgrid(0:N(1)-1, 0:N(2)-1, 0:N(3)-1);
Ex = (exp(2i*pi*kx/N(1)) - 1) / vox(1);
Ey = (exp(2i*pi*ky/N(2)) - 1) / vox(2);
Ez = (exp(2i*pi*kz/N(3)) - 1) / vox(3);
end

function out = div_k(Ex,Ey,Ez, Zx,Zy,Zz)
out = conj(Ex).*Zx + conj(Ey).*Zy + conj(Ez).*Zz;
end

function [zx,zy,zz] = shrink3(gx,gy,gz,thr)
mag = sqrt(gx.^2+gy.^2+gz.^2) + eps;
sc = max(mag - thr, 0) ./ mag;
zx = gx.*sc; zy = gy.*sc; zz = gz.*sc;
end

function v = get_cfg(cfg, pathCells, default)
v = default;
try
    s = cfg;
    for i = 1:numel(pathCells)
        if isfield(s, pathCells{i}), s = s.(pathCells{i}); else, return; end
    end
    if ~isempty(s), v = s; end
catch
    v = default;
end
end
