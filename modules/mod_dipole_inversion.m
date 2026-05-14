function [qsm_results, qsm_names] = mod_dipole_inversion(local_field, data, cfg)
% mod_dipole_inversion.m
% 偶极子反演模块
%
% 方法:
%   1. TKD
%   2. Closed-form L2
%   3. iLSQR
%   4. MEDI / MEDI-like structural prior
%   5. WH-QSM fallback
%
% 关键修正:
%   1. create_dipole_kernel 已修正为与 fftn 顺序一致，避免 kernel 退化/错位。
%
%   2. QSM2016 phs_tissue 是 ppm 局部场，而 MEDI toolbox README 中 RDF 单位是 rad/echo。
%      MEDI 内部通常使用:
%          field_ppm = RDF / (2*pi*delta_TE*CF) * 1e6
%
%      因此若希望 field_ppm = local_field_ppm，应设置:
%          CF = 1e6
%          delta_TE = 1
%          RDF = local_field_ppm * 2*pi
%
%      之前设置 CF = 1 且 RDF = ppm，会使 MEDI 内部场放大约 1e6/(2*pi)，
%      导致 MEDI 结果非红即蓝、数值爆炸。
%
%   3. MEDI lambda:
%      cfg 中的 lambda 例如 0.01~3 是针对本 pipeline ppm 反演的尺度。
%      MEDI toolbox README 推荐 lambda 约 1000。
%      因此传给 MEDI toolbox 时使用 lambda_medi = lambda * 1000。
%
%   4. 若 MEDI toolbox 输出仍异常，则使用内置结构先验 ADMM-TV fallback。

N = data.N;
Mask = logical(data.Mask);
voxel_size = data.spatial_res(:).';

local_field = double(local_field);
local_field(~Mask) = 0;

if ~is_valid_volume(local_field, Mask)
    warning('输入 local_field 无效，改用 data.phs_tissue。');
    local_field = double(data.phs_tissue);
    local_field(~Mask) = 0;
end

qsm_results = [];
qsm_names = {};

%% 创建偶极子核
kernel = create_dipole_kernel(N, voxel_size, [0 0 1]);

fprintf('Dipole kernel 统计: min=%.4f, max=%.4f, std=%.4f\n', ...
    min(kernel(:)), max(kernel(:)), std(kernel(:)));

if std(kernel(:)) < 1e-8
    error('Dipole kernel 退化为常数，请检查 create_dipole_kernel.m。');
end

%% ========================================================================
% 方法 1: TKD
% ========================================================================
fprintf('--- TKD (Thresholded K-space Division) ---\n');

thre = cfg.inversion.tkd_threshold;

kernel_inv = zeros(N);
idx = abs(kernel) > thre;
kernel_inv(idx) = 1 ./ kernel(idx);

chi_tkd = real(ifftn(fftn(local_field) .* kernel_inv));
chi_tkd = remove_mask_mean(chi_tkd, Mask);
chi_tkd = chi_tkd .* double(Mask);

qsm_results = cat(4, qsm_results, chi_tkd);
qsm_names{end+1, 1} = 'TKD';

print_volume_summary('TKD', chi_tkd, Mask);

%% ========================================================================
% 方法 2: Closed-form L2
% ========================================================================
fprintf('--- Closed-form L2 ---\n');

reg_param = cfg.inversion.cfl2_reg;

[k1, k2, k3] = ndgrid(0:N(1)-1, 0:N(2)-1, 0:N(3)-1);
E1 = 1 - exp(2i * pi * k1 / N(1));
E2 = 1 - exp(2i * pi * k2 / N(2));
E3 = 1 - exp(2i * pi * k3 / N(3));
EtE = abs(E1).^2 + abs(E2).^2 + abs(E3).^2;
DtD = abs(kernel).^2;

denom = DtD + reg_param * EtE;
denom(denom < eps) = eps;

chi_L2 = real(ifftn(conj(kernel) .* fftn(local_field) ./ denom));
chi_L2 = remove_mask_mean(chi_L2, Mask);
chi_L2 = chi_L2 .* double(Mask);

qsm_results = cat(4, qsm_results, chi_L2);
qsm_names{end+1, 1} = 'CFL2';

print_volume_summary('Closed-form L2', chi_L2, Mask);

%% ========================================================================
% 方法 3: iLSQR
% ========================================================================
fprintf('--- iLSQR (iterative LSQR) ---\n');

chi_ilsqr = inversion_ilsqr(local_field, kernel, Mask, ...
    cfg.inversion.ilsqr_tol, cfg.inversion.ilsqr_maxiter);

qsm_results = cat(4, qsm_results, chi_ilsqr);
qsm_names{end+1, 1} = 'iLSQR';

print_volume_summary('iLSQR', chi_ilsqr, Mask);

%% ========================================================================
% 方法 4: MEDI
% ========================================================================
fprintf('--- MEDI / MEDI-like structural prior inversion ---\n');

lambdaList = cfg.inversion.medi_lambdas;

for ii = 1:numel(lambdaList)
    lambda = lambdaList(ii);
    fprintf(' MEDI lambda label = %.4g\n', lambda);

    chi_medi = run_medi_with_structural( ...
        local_field, data.magn, data.mp_rage, Mask, voxel_size, kernel, ...
        lambda, cfg.resultDir, cfg.inversion.medi_use_structural);

    chi_medi = real(double(chi_medi));
    chi_medi(~Mask) = 0;

    if ~is_plausible_qsm(chi_medi, Mask)
        fprintf('  MEDI toolbox 输出异常，使用内置结构先验 ADMM-TV fallback。\n');
        chi_medi = internal_medi_structural_admm( ...
            local_field, kernel, Mask, data.mp_rage, lambda);
    end

    chi_medi = remove_mask_mean(chi_medi, Mask);
    chi_medi = chi_medi .* double(Mask);

    qsm_results = cat(4, qsm_results, chi_medi);
    qsm_names{end+1, 1} = sprintf('MEDI_L%.4g', lambda);

    print_volume_summary(sprintf('MEDI lambda=%.4g', lambda), chi_medi, Mask);
end

%% ========================================================================
% 方法 5: WH-QSM fallback
% ========================================================================
fprintf('--- WH-QSM / Weak-harmonic-inspired TV inversion ---\n');

chi_whqsm = inversion_whqsm_stable(data, local_field, kernel, Mask, voxel_size);

if ~isempty(chi_whqsm) && is_plausible_qsm(chi_whqsm, Mask)
    chi_whqsm = remove_mask_mean(chi_whqsm, Mask);
    chi_whqsm = chi_whqsm .* double(Mask);

    qsm_results = cat(4, qsm_results, chi_whqsm);
    qsm_names{end+1, 1} = 'WH-QSM';

    print_volume_summary('WH-QSM', chi_whqsm, Mask);
else
    warning('WH-QSM 结果无效，已跳过。');
end

%% 保存
save(fullfile(cfg.resultDir, 'dipole_inversion_results.mat'), ...
    'qsm_results', 'qsm_names', 'kernel', '-v7.3');

fprintf('偶极子反演完成，共 %d 种方法。\n', size(qsm_results, 4));

end

%% =========================================================================
% iLSQR
% =========================================================================
function chi = inversion_ilsqr(local_field, kernel, Mask, tol, maxiter)

N = size(Mask);
Mask = logical(Mask);

fprintf('  iLSQR: %d mask voxels, tol=%.2e, maxiter=%d\n', nnz(Mask), tol, maxiter);

mu = 5e-3;

DtD = abs(kernel).^2;

[k1, k2, k3] = ndgrid(0:N(1)-1, 0:N(2)-1, 0:N(3)-1);
E1 = 1 - exp(2i * pi * k1 / N(1));
E2 = 1 - exp(2i * pi * k2 / N(2));
E3 = 1 - exp(2i * pi * k3 / N(3));
EtE = abs(E1).^2 + abs(E2).^2 + abs(E3).^2;

Dtb = real(ifftn(conj(kernel) .* fftn(local_field)));

chi_vol = zeros(N);
r = Dtb .* double(Mask);
p = r;
rsold = sum(r(:).^2);
rs0 = rsold;

for iter = 1:maxiter
    Ap = real(ifftn((DtD + mu * EtE) .* fftn(p)));
    Ap = Ap .* double(Mask);

    alpha = rsold / max(sum(p(:) .* Ap(:)), eps);

    chi_vol = chi_vol + alpha * p;
    r = r - alpha * Ap;

    rsnew = sum(r(:).^2);

    if sqrt(rsnew) < tol * sqrt(max(rs0, eps))
        fprintf('  iLSQR 收敛于第 %d 次迭代。\n', iter);
        break;
    end

    p = r + (rsnew / max(rsold, eps)) * p;
    rsold = rsnew;
end

if iter == maxiter
    fprintf('  iLSQR 达到最大迭代次数 %d。\n', maxiter);
end

chi = remove_mask_mean(chi_vol, Mask);
chi = chi .* double(Mask);

end

%% =========================================================================
% MEDI toolbox 调用
% =========================================================================
function chi_medi = run_medi_with_structural( ...
    local_field_ppm, magn, mp_rage, Mask, voxel_size, kernel, lambda_label, outDir, use_structural)

Mask = logical(Mask);
matrix_size = size(Mask);

lambdaTag = strrep(sprintf('%.4g', lambda_label), '.', 'p');
caseDir = fullfile(outDir, ['MEDI_structural_lambda_' lambdaTag]);

if ~exist(caseDir, 'dir')
    mkdir(caseDir);
end

cleanup_medi_case_dir(caseDir);

% -------------------------------------------------------------------------
% QSM2016 ppm -> MEDI README rad/echo
% -------------------------------------------------------------------------
% MEDI README 变量单位:
%   RDF: rad/echo
%   CF: Hz
%   delta_TE: sec
%
% 构造人工参数使得 MEDI 内部换算后 field_ppm = local_field_ppm:
%   field_ppm = RDF / (2*pi*delta_TE*CF) * 1e6
%   令 CF=1e6, delta_TE=1, RDF=local_field_ppm*2*pi
RDF = double(local_field_ppm) * 2*pi;
RDF(~Mask) = 0;

iFreq = RDF;

CF = 1e6;
delta_TE = 1;
TE = 1;
B0_dir = [0 0 1];

% MEDI lambda 映射到 toolbox 推荐尺度
if lambda_label < 10
    lambda_medi = lambda_label * 1000;
else
    lambda_medi = lambda_label;
end

fprintf('  MEDI 单位转换: RDF = local_field_ppm * 2*pi, CF=1e6, delta_TE=1。\n');
fprintf('  MEDI toolbox lambda = %.4g\n', lambda_medi);

if ~is_valid_volume(RDF, Mask)
    warning('MEDI 输入 RDF 无效，使用 fallback。');
    chi_medi = internal_medi_structural_admm(local_field_ppm, kernel, Mask, mp_rage, lambda_label);
    return;
end

% -------------------------------------------------------------------------
% morphology prior
% -------------------------------------------------------------------------
if use_structural && ~isempty(mp_rage) && any(mp_rage(:) > 0)
    fprintf('  使用 mp_rage 作为 MEDI morphology prior。\n');
    iMag = normalize_image(mp_rage, Mask);
else
    fprintf('  使用 magnitude 作为 MEDI morphology prior。\n');
    iMag = normalize_image(magn, Mask);
end

N_std = ones(matrix_size);

save(fullfile(caseDir, 'RDF.mat'), ...
    'RDF', 'iFreq', 'iMag', 'N_std', 'Mask', ...
    'matrix_size', 'voxel_size', 'delta_TE', 'TE', ...
    'CF', 'B0_dir');

chi_medi = [];

if exist('MEDI_L1', 'file') == 2
    oldDir = pwd;
    cleanObj = onCleanup(@() cd(oldDir)); %#ok<NASGU>
    cd(caseDir);

    try
        fprintf('  调用 MEDI_L1 toolbox。\n');

        try
            chi_medi = MEDI_L1('lambda', lambda_medi, 'merit', 'smv', 5);
        catch
            chi_medi = MEDI_L1('lambda', lambda_medi);
        end

        if isempty(chi_medi)
            chi_medi = load_medi_output_file(caseDir);
        end

        chi_medi = double(real(chi_medi));
        chi_medi(~Mask) = 0;

        if is_plausible_qsm(chi_medi, Mask)
            fprintf('  MEDI_L1 toolbox 成功。\n');
            return;
        else
            fprintf('  MEDI_L1 toolbox 输出异常，将使用 fallback。\n');
        end

    catch ME
        fprintf('  MEDI_L1 调用失败: %s\n', ME.message);
        fprintf('  将使用 fallback。\n');
    end
else
    fprintf('  未找到 MEDI_L1，使用 fallback。\n');
end

chi_medi = internal_medi_structural_admm(local_field_ppm, kernel, Mask, mp_rage, lambda_label);

end

%% =========================================================================
% 内置 MEDI-like structural ADMM-TV
% =========================================================================
function chi = internal_medi_structural_admm(local_field, kernel, Mask, structural, lambda_label)

Mask = logical(Mask);
N = size(Mask);

b = double(local_field);
b(~Mask) = 0;
b = remove_mask_mean(b, Mask);

% ppm 尺度下的稳定 TV 权重
alpha_tv = max(min(lambda_label * 5e-4, 2e-3), 1e-5);
rho = 5e-3;
max_iter = 80;
tol = 1e-4;

fprintf('  内置 MEDI-like ADMM-TV: alpha=%.3e, rho=%.3e\n', alpha_tv, rho);

DtD = abs(kernel).^2;

[k1, k2, k3] = ndgrid(0:N(1)-1, 0:N(2)-1, 0:N(3)-1);
E1 = 1 - exp(2i * pi * k1 / N(1));
E2 = 1 - exp(2i * pi * k2 / N(2));
E3 = 1 - exp(2i * pi * k3 / N(3));
EtE = abs(E1).^2 + abs(E2).^2 + abs(E3).^2;

Dtb = real(ifftn(conj(kernel) .* fftn(b)));

Ws = build_structural_weight(structural, Mask);

chi = zeros(N);
zx = zeros(N); zy = zeros(N); zz = zeros(N);
ux = zeros(N); uy = zeros(N); uz = zeros(N);

denom = DtD + rho * EtE;
denom(denom < 1e-6) = 1e-6;

for iter = 1:max_iter
    div_zu = compute_divergence(zx - ux, zy - uy, zz - uz);
    rhs = Dtb + rho * div_zu;

    chi_old = chi;

    chi = real(ifftn(fftn(rhs) ./ denom));
    chi = chi .* double(Mask);
    chi = remove_mask_mean(chi, Mask);

    [gx, gy, gz] = compute_gradient(chi);

    th = alpha_tv * Ws / rho;

    zx = shrink(gx + ux, th);
    zy = shrink(gy + uy, th);
    zz = shrink(gz + uz, th);

    ux = ux + gx - zx;
    uy = uy + gy - zy;
    uz = uz + gz - zz;

    relchg = norm(chi(:) - chi_old(:)) / max(norm(chi_old(:)), eps);

    if iter > 5 && relchg < tol
        fprintf('  内置 MEDI-like ADMM-TV 收敛于第 %d 次迭代。\n', iter);
        break;
    end
end

chi = winsorize_qsm(chi, Mask, 0.5);
chi = chi .* double(Mask);

end

%% =========================================================================
% WH-QSM stable implementation
% =========================================================================
function chi = inversion_whqsm_stable(varargin)
% inversion_whqsm_stable
%
% 兼容两种调用方式:
%   A) 旧接口: chi = inversion_whqsm_stable(data, local_field, kernel, Mask, voxel_size)
%   B) 新接口: chi = inversion_whqsm_stable(local_field, data, voxel_size)
%
% 目标:
%   使用 SEPIA 的 QSMMacroIOWrapper + FANSI 进行 WH-QSM，
%   并稳定定位输出 QSM 文件。

chi = [];

%% ------------------------------------------------------------------------
% 0) 解析输入参数（兼容旧/新接口）
% -------------------------------------------------------------------------
if nargin == 5
    % 旧接口
    data = varargin{1};
    local_field = varargin{2};
    % varargin{3} = kernel (未使用)
    Mask_in = varargin{4};
    voxel_size = varargin{5};
elseif nargin == 3
    % 新接口
    local_field = varargin{1};
    data = varargin{2};
    voxel_size = varargin{3};
    if isfield(data, 'Mask')
        Mask_in = data.Mask;
    else
        error('inversion_whqsm_stable: data.Mask 不存在。');
    end
else
    error(['inversion_whqsm_stable 输入参数数量错误。' newline ...
           '支持: (data, local_field, kernel, Mask, voxel_size) 或 (local_field, data, voxel_size)']);
end

Mask = logical(Mask_in);
matrixSize = size(Mask);
voxel_size = double(voxel_size(:).');

fprintf('WH-QSM: 使用 SEPIA QSMMacroIOWrapper 调用 FANSI (weak-harmonic)。\n');

%% ------------------------------------------------------------------------
% 1) 准备输入局部场
% -------------------------------------------------------------------------
b_ppm = double(local_field);
b_ppm(~Mask) = 0;

if ~is_valid_volume(b_ppm, Mask)
    fprintf('local_field 无效，改用 data.phs_tissue。\n');
    b_ppm = double(data.phs_tissue);
    b_ppm(~Mask) = 0;
end

b_ppm = remove_mask_mean(b_ppm, Mask);

%% ------------------------------------------------------------------------
% 2) 添加 SEPIA 路径
% -------------------------------------------------------------------------
sepiaRoot = 'D:\MRI_PRO\MRILAB_X\sepia';
if exist(sepiaRoot, 'dir') ~= 7
    error('WH-QSM: SEPIA 路径不存在: %s', sepiaRoot);
end

addpath(sepiaRoot);
addpath(genpath(sepiaRoot));
fprintf('已添加 SEPIA 路径: %s\n', sepiaRoot);

if exist('sepia_addpath', 'file') ~= 2
    error('WH-QSM: 未找到 sepia_addpath。');
end

try
    sepia_addpath;
    fprintf('已调用 sepia_addpath。\n');
catch ME
    error('WH-QSM: sepia_addpath 调用失败: %s', ME.message);
end

if exist('QSMMacroIOWrapper', 'file') ~= 2
    error('WH-QSM: 未找到 QSMMacroIOWrapper。请确认 SEPIA 安装完整并在路径中。');
end

%% ------------------------------------------------------------------------
% 3) ppm -> Hz
% -------------------------------------------------------------------------
gyro = 42.57747892;
if exist('sepia_universal_variables', 'file') == 2
    try
        sepia_universal_variables;
        if ~(exist('gyro', 'var') && isnumeric(gyro) && isfinite(gyro) && gyro > 0)
            gyro = 42.57747892;
        end
    catch
        gyro = 42.57747892;
    end
end

B0 = 3;
if isfield(data, 'B0') && isnumeric(data.B0) && isfinite(data.B0) && data.B0 > 0
    B0 = double(data.B0);
elseif isfield(data, 'b0') && isnumeric(data.b0) && isfinite(data.b0) && data.b0 > 0
    B0 = double(data.b0);
end

localField_Hz = b_ppm * B0 * gyro;
localField_Hz(~Mask) = 0;

fprintf('输入给 SEPIA 的 local field 单位: Hz\n');
fprintf('B0 = %.4g T, gyro = %.10g\n', B0, gyro);

%% ------------------------------------------------------------------------
% 4) FANSI 参数
% -------------------------------------------------------------------------
algorParam = struct();
algorParam.general.isBET = 0;
algorParam.general.isInvert = 0;

algorParam.qsm.reference_tissue = 'None';
algorParam.qsm.method = 'FANSI';
algorParam.qsm.tol = 1e-4;
algorParam.qsm.maxiter = 100;
algorParam.qsm.lambda = 5e-4;
algorParam.qsm.alpha1 = 5e-4;
algorParam.qsm.mu1 = 5e-5;
algorParam.qsm.mu = 5e-5;
algorParam.qsm.mu2 = 1.0;
algorParam.qsm.solver = 'Nonlinear';
algorParam.qsm.constraint = 'TV';
algorParam.qsm.gradient_mode = 'none';
algorParam.qsm.isWeakHarmonic = true;
algorParam.qsm.beta = 150;
algorParam.qsm.muh = 5;
algorParam.qsm.isGPU = false;

%% ------------------------------------------------------------------------
% 5) QSMMacroIOWrapper 文件接口
% -------------------------------------------------------------------------
if exist('niftiwrite', 'file') ~= 2
    error('WH-QSM: 当前 MATLAB 无 niftiwrite，无法执行 QSMMacroIOWrapper 文件接口。');
end

tmpDir = tempname;
mkdir(tmpDir);

localFieldFile  = fullfile(tmpDir, 'Sepia_localfield.nii');
maskFile        = fullfile(tmpDir, 'Sepia_mask.nii');
magFile         = fullfile(tmpDir, 'Sepia_mag.nii');
headerFile      = fullfile(tmpDir, 'Sepia_header.mat');
output_basename = fullfile(tmpDir, 'Sepia');

niftiwrite(single(localField_Hz), localFieldFile);
niftiwrite(uint8(Mask), maskFile);

if isfield(data, 'magn') && ~isempty(data.magn)
    mag = double(data.magn);
    mag(~Mask) = 0;
else
    mag = double(Mask);
end
niftiwrite(single(mag), magFile);

header = struct();
header.matrixSize = matrixSize;
header.matrix_size = matrixSize;
header.voxelSize = voxel_size;
header.voxel_size = voxel_size;
header.b0dir = [0 0 1];
header.B0_dir = [0 0 1];
header.b0 = B0;
header.B0 = B0;
header.TE = 0.025;
header.delta_TE = 0.025;
header.CF = B0 * gyro;

matrix_size = matrixSize; %#ok<NASGU>
voxelSize = voxel_size; %#ok<NASGU>
voxel_size_ = voxel_size; %#ok<NASGU>
B0_dir = [0 0 1]; %#ok<NASGU>
b0dir = [0 0 1]; %#ok<NASGU>
b0 = B0; %#ok<NASGU>
TE = 0.025; %#ok<NASGU>
delta_TE = 0.025; %#ok<NASGU>
CF = B0 * gyro; %#ok<NASGU>

save(headerFile, ...
    'header', ...
    'matrix_size', 'matrixSize', ...
    'voxelSize', 'voxel_size', 'voxel_size_', ...
    'B0_dir', 'b0dir', ...
    'b0', 'B0', ...
    'TE', 'delta_TE', 'CF');

input = struct();
input(1).name = localFieldFile;
input(2).name = magFile;
input(3).name = '';
input(4).name = headerFile;
mask_filename = maskFile;

fprintf('调用 QSMMacroIOWrapper...\n');
QSMMacroIOWrapper(input, output_basename, mask_filename, algorParam);

qsmPath = resolve_sepia_qsm_output(output_basename);
if isempty(qsmPath)
    niiList = list_nifti_files(tmpDir);
    error(['WH-QSM: QSMMacroIOWrapper 已完成，但未定位到 QSM 输出文件。' newline ...
           'output_basename: ' output_basename newline ...
           'tmpDir NIfTI 列表: ' strjoin(niiList, ', ')]);
end

fprintf('读取 SEPIA 输出: %s\n', qsmPath);
chi_tmp = double(niftiread(qsmPath));
chi_tmp = squeeze(chi_tmp);
chi_tmp(~Mask) = 0;

if ~is_plausible_qsm(chi_tmp, Mask)
    error('WH-QSM: 已读取输出文件，但结果数值异常（is_plausible_qsm 未通过）。');
end

chi = chi_tmp;
fprintf('WH-QSM 完成。\n');

end

%% =========================================================================
% MEDI output loader
% =========================================================================
function QSM = load_medi_output_file(caseDir)

patterns = {'QSM*.mat', '*QSM*.mat', '*MEDI*.mat', 'results*.mat'};
files = [];

for i = 1:numel(patterns)
    files = [files; dir(fullfile(caseDir, patterns{i}))]; %#ok<AGROW>
end

files = files(~strcmpi({files.name}, 'RDF.mat'));

if isempty(files)
    error('未找到 MEDI 输出文件。');
end

[~, idx] = sort([files.datenum], 'descend');
S = load(fullfile(caseDir, files(idx(1)).name));

if isfield(S, 'QSM')
    QSM = S.QSM;
elseif isfield(S, 'chi_medi')
    QSM = S.chi_medi;
elseif isfield(S, 'x')
    QSM = S.x;
else
    names = fieldnames(S);
    QSM = [];
    for j = 1:numel(names)
        v = S.(names{j});
        if isnumeric(v) && ndims(v) == 3
            QSM = v;
            return;
        end
    end

    error('无法从 MEDI 输出中识别 QSM 体积。');
end

end

%% =========================================================================
% 辅助函数
% =========================================================================
function cleanup_medi_case_dir(caseDir)

if ~exist(caseDir, 'dir')
    return;
end

patterns = {'QSM*.mat', '*QSM*.mat', '*MEDI*.mat', 'results*.mat', 'RDF.mat'};
for i = 1:numel(patterns)
    files = dir(fullfile(caseDir, patterns{i}));
    for j = 1:numel(files)
        try
            delete(fullfile(caseDir, files(j).name));
        catch
        end
    end
end

end

function img = normalize_image(img, Mask)

Mask = logical(Mask);
img = double(img);
img(~Mask) = 0;

vals = img(Mask);
vals = vals(isfinite(vals) & vals > 0);

if isempty(vals)
    img = double(Mask);
    return;
end

p995 = prctile(vals, 99.5);
img = img / max(p995, eps);
img = min(max(img, 0), 1);
img(~Mask) = 0;

end

function Ws = build_structural_weight(structural, Mask)

Mask = logical(Mask);

if isempty(structural) || ~any(structural(:))
    Ws = double(Mask);
    return;
end

s = normalize_image(structural, Mask);

[gx, gy, gz] = compute_gradient(s);
gmag = sqrt(gx.^2 + gy.^2 + gz.^2);

vals = gmag(Mask);
vals = vals(isfinite(vals));

if isempty(vals) || max(vals) <= 0
    Ws = double(Mask);
    return;
end

scale = prctile(vals, 90);
scale = max(scale, eps);

Ws = exp(-gmag / scale);
Ws = min(max(Ws, 0.05), 1);
Ws(~Mask) = 0;

end

function [gx, gy, gz] = compute_gradient(vol)

gx = circshift(vol, -1, 1) - vol;
gy = circshift(vol, -1, 2) - vol;
gz = circshift(vol, -1, 3) - vol;

end

function div = compute_divergence(px, py, pz)

div = px - circshift(px, 1, 1) + ...
      py - circshift(py, 1, 2) + ...
      pz - circshift(pz, 1, 3);

end

function y = shrink(x, t)

y = sign(x) .* max(abs(x) - t, 0);

end

function vol = remove_mask_mean(vol, Mask)

Mask = logical(Mask);
v = vol(Mask);
v = v(isfinite(v));

if ~isempty(v)
    vol(Mask) = vol(Mask) - mean(v);
end

vol(~Mask) = 0;

end

function vol = winsorize_qsm(vol, Mask, absLimit)

Mask = logical(Mask);
v = vol(Mask);
v = v(isfinite(v));

if isempty(v)
    return;
end

p999 = prctile(abs(v), 99.9);
lim = min(max(p999, 0.15), absLimit);

vol(vol > lim) = lim;
vol(vol < -lim) = -lim;
vol(~Mask) = 0;

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

function tf = is_plausible_qsm(vol, Mask)

if ~is_valid_volume(vol, Mask)
    tf = false;
    return;
end

Mask = logical(Mask);
v = double(vol(Mask));
v = v(isfinite(v));

p999 = prctile(abs(v), 99.9);
s = std(v);

% 脑 QSM 通常在 sub-ppm 范围。
% 若 p99.9 超过 2 ppm，基本说明单位或反演爆炸。
tf = p999 < 2 && s < 1;

end

function qsmPath = resolve_sepia_qsm_output(output_basename)
% resolve_sepia_qsm_output
%
% 兼容不同 SEPIA 版本的 QSM 输出命名:
%   <base>_QSM.nii(.gz)
%   <base>_qsm.nii(.gz)
%   <base>_Chi.nii(.gz)
%   <base>_chi.nii(.gz)
%   <base>_Chimap.nii(.gz)   <-- 关键修复
%
% 若上述固定候选都找不到，则递归扫描目录并按关键词评分。

qsmPath = '';

[outDir, base, ~] = fileparts(output_basename);

% -------------------------------------------------------------------------
% 1) 固定候选名（优先）
% -------------------------------------------------------------------------
cands = { ...
    fullfile(outDir, [base '_QSM.nii.gz']), ...
    fullfile(outDir, [base '_QSM.nii']), ...
    fullfile(outDir, [base '_qsm.nii.gz']), ...
    fullfile(outDir, [base '_qsm.nii']), ...
    fullfile(outDir, [base '_Chi.nii.gz']), ...
    fullfile(outDir, [base '_Chi.nii']), ...
    fullfile(outDir, [base '_chi.nii.gz']), ...
    fullfile(outDir, [base '_chi.nii']), ...
    fullfile(outDir, [base '_Chimap.nii.gz']), ...  % 修复点
    fullfile(outDir, [base '_Chimap.nii']), ...
    fullfile(outDir, [base '_chimap.nii.gz']), ...
    fullfile(outDir, [base '_chimap.nii']) ...
    };

for i = 1:numel(cands)
    if exist(cands{i}, 'file') == 2
        qsmPath = cands{i};
        return;
    end
end

% -------------------------------------------------------------------------
% 2) 回退扫描（兼容更多命名）
% -------------------------------------------------------------------------
d1 = dir(fullfile(outDir, '*.nii'));
d2 = dir(fullfile(outDir, '*.nii.gz'));
files = [d1; d2];

if isempty(files)
    return;
end

bestScore = -Inf;
bestPath = '';

for i = 1:numel(files)
    fpath = fullfile(files(i).folder, files(i).name);
    lname = lower(files(i).name);

    score = 0;

    % 强关键词
    if contains(lname, 'qsm')
        score = score + 10;
    end
    if contains(lname, 'chi')
        score = score + 8;
    end
    if contains(lname, 'chimap')
        score = score + 12;
    end

    % 与 basename 相关
    if contains(lower(fpath), lower(base))
        score = score + 5;
    end

    % 排除输入文件
    if contains(lname, 'localfield') || contains(lname, 'mask') || contains(lname, 'mag') || contains(lname, 'header')
        score = score - 20;
    end

    if score > bestScore
        bestScore = score;
        bestPath = fpath;
    end
end

if bestScore > 0
    qsmPath = bestPath;
end

end

function names = list_nifti_files(rootDir)
% list_nifti_files
%
% 返回目录内 NIfTI 文件名列表，用于报错诊断。

names = {};

try
    d1 = dir(fullfile(rootDir, '*.nii'));
    d2 = dir(fullfile(rootDir, '*.nii.gz'));
    names = [{d1.name}, {d2.name}];
catch
    names = {};
end

if isempty(names)
    names = {'<none>'};
end

end