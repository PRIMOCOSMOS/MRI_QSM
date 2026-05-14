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
% WH-QSM stable fallback
% =========================================================================
function chi = inversion_whqsm_stable(data, local_field, kernel, Mask, voxel_size) %#ok<INUSD>
% inversion_whqsm_stable
%
% 仅通过 SEPIA 提供的 FANSI 封装调用 weak-harmonic FANSI。
%
% 重要:
%   1. 不直接调用 FANSI toolbox 内部函数。
%   2. 不手动生成 dipole_kernel_angulated.m。
%   3. 不在本函数内额外引入 FANSI 依赖。
%   4. 依赖 SEPIA 自己的 sepia_addpath / QSMMacroIOWrapper / Wrapper_QSM_FANSI。
%
% 如果 SEPIA 内部没有正确配置 FANSI，函数会跳过 WH-QSM，返回 []。

chi = [];

Mask = logical(Mask);
matrixSize = size(Mask);
voxel_size = double(voxel_size(:).');

fprintf('  WH-QSM: 使用 SEPIA 封装调用 FANSI weak-harmonic regularisation。\n');

%% ------------------------------------------------------------------------
% 1. 准备输入局部场
% -------------------------------------------------------------------------
b_ppm = double(local_field);
b_ppm(~Mask) = 0;

if ~is_valid_volume(b_ppm, Mask)
    fprintf('  local_field 无效，改用 data.phs_tissue。\n');
    b_ppm = double(data.phs_tissue);
    b_ppm(~Mask) = 0;
end

b_ppm = remove_mask_mean(b_ppm, Mask);

%% ------------------------------------------------------------------------
% 2. 添加 SEPIA 路径
% -------------------------------------------------------------------------
sepiaRoot = 'D:\MRI_PRO\MRILAB_X\sepia';

if exist(sepiaRoot, 'dir') ~= 7
    fprintf('  SEPIA 路径不存在: %s\n', sepiaRoot);
    fprintf('  跳过 WH-QSM/FANSI。\n');
    return;
end

addpath(sepiaRoot);
addpath(genpath(sepiaRoot));

fprintf('  已添加 SEPIA 路径: %s\n', sepiaRoot);

if exist('sepia_addpath', 'file') == 2
    try
        sepia_addpath;
        fprintf('  已调用 sepia_addpath，由 SEPIA 管理算法依赖路径。\n');
    catch ME
        fprintf('  sepia_addpath 调用失败: %s\n', ME.message);
        fprintf('  跳过 WH-QSM/FANSI。\n');
        return;
    end
else
    fprintf('  未找到 sepia_addpath。\n');
    fprintf('  跳过 WH-QSM/FANSI。\n');
    return;
end

%% ------------------------------------------------------------------------
% 3. 使用 SEPIA 的物理单位约定
% -------------------------------------------------------------------------
% SEPIA QSM standalone 通常接收 local field，单位 Hz，
% 并根据 header 中的 B0 和 gyro 转换输出为 ppm。
%
% QSM2016 的 phs_tissue/local_field 已经是 ppm。
% 因此这里先将 ppm 转成 SEPIA 期望的 Hz:
%
%   localField_Hz = localField_ppm * B0 * gyro
%
% 其中 gyro 使用 SEPIA 自己的 sepia_universal_variables 中定义的值。
% 若无法读取，则使用常用值 42.57747892 Hz/ppm/T。

gyro = 42.57747892;

if exist('sepia_universal_variables', 'file') == 2
    try
        sepia_universal_variables;
        if exist('gyro', 'var') && isnumeric(gyro) && isfinite(gyro) && gyro > 0
            fprintf('  使用 SEPIA gyro = %.10g\n', gyro);
        else
            gyro = 42.57747892;
        end
    catch
        gyro = 42.57747892;
    end
end

% QSM Challenge 2016 通常按 3T 数据处理。
% 若 data 中包含 B0 信息则优先使用。
B0 = 3;

if isfield(data, 'B0') && isnumeric(data.B0) && isfinite(data.B0) && data.B0 > 0
    B0 = double(data.B0);
elseif isfield(data, 'b0') && isnumeric(data.b0) && isfinite(data.b0) && data.b0 > 0
    B0 = double(data.b0);
end

localField_Hz = b_ppm * B0 * gyro;
localField_Hz(~Mask) = 0;

fprintf('  输入给 SEPIA/FANSI 的 local field 单位: Hz\n');
fprintf('  B0 = %.4g T, gyro = %.10g\n', B0, gyro);

%% ------------------------------------------------------------------------
% 4. 构造 SEPIA FANSI 参数
% -------------------------------------------------------------------------
algorParam = struct();

algorParam.general.isBET    = 0;
algorParam.general.isInvert = 0;

algorParam.qsm.reference_tissue = 'None';
algorParam.qsm.method           = 'FANSI';

algorParam.qsm.tol              = 1e-4;
algorParam.qsm.maxiter          = 100;

% SEPIA/FANSI 参数
algorParam.qsm.lambda           = 5e-4;
algorParam.qsm.alpha1           = 5e-4;  % 兼容不同 SEPIA 版本
algorParam.qsm.mu1              = 5e-3;
algorParam.qsm.mu               = 5e-3;  % 兼容不同 SEPIA 版本
algorParam.qsm.mu2              = 1e-2;

algorParam.qsm.solver           = 'Nonlinear';
algorParam.qsm.constraint       = 'TV';
algorParam.qsm.gradient_mode    = 'none';

% weak harmonic regularisation
algorParam.qsm.isWeakHarmonic   = true;
algorParam.qsm.beta             = 1e-3;
algorParam.qsm.muh              = 1e-2;

algorParam.qsm.isGPU            = false;

%% ------------------------------------------------------------------------
% 5. 首选 SEPIA 高层 QSM macro wrapper
% -------------------------------------------------------------------------
if exist('QSMMacroIOWrapper', 'file') == 2

    fprintf('  使用 SEPIA 高层接口 QSMMacroIOWrapper 调用 FANSI。\n');

    tmpDir = tempname;
    mkdir(tmpDir);

    localFieldFile = fullfile(tmpDir, 'sepia_localfield.nii');
    maskFile       = fullfile(tmpDir, 'sepia_mask.nii');
    magFile        = fullfile(tmpDir, 'sepia_mag.nii');
    headerFile     = fullfile(tmpDir, 'sepia_header.mat');

    output_basename = fullfile(tmpDir, 'sepia_fansi_wh');

    try
        if exist('niftiwrite', 'file') ~= 2
            fprintf('  当前 MATLAB 无 niftiwrite，无法使用 QSMMacroIOWrapper 文件接口。\n');
            fprintf('  尝试使用 SEPIA 低层 wrapper。\n');
        else
            niftiwrite(single(localField_Hz), localFieldFile);
            niftiwrite(uint8(Mask), maskFile);

            if isfield(data, 'magn') && ~isempty(data.magn)
                mag = double(data.magn);
                mag(~Mask) = 0;
            else
                mag = double(Mask);
            end
            niftiwrite(single(mag), magFile);

            % SEPIA header
            header = struct();
            header.matrixSize  = matrixSize;
            header.matrix_size = matrixSize;
            header.voxelSize   = voxel_size;
            header.voxel_size  = voxel_size;
            header.b0dir       = [0 0 1];
            header.B0_dir      = [0 0 1];
            header.b0          = B0;
            header.B0          = B0;
            header.TE          = 1;
            header.delta_TE    = 1;
            header.CF          = B0 * gyro;

            matrix_size = matrixSize; %#ok<NASGU>
            voxelSize   = voxel_size; %#ok<NASGU>
            voxel_size_ = voxel_size; %#ok<NASGU>
            B0_dir      = [0 0 1]; %#ok<NASGU>
            b0dir       = [0 0 1]; %#ok<NASGU>
            b0          = B0; %#ok<NASGU>
            TE          = 1; %#ok<NASGU>
            delta_TE    = 1; %#ok<NASGU>
            CF          = B0 * gyro; %#ok<NASGU>

            save(headerFile, ...
                'header', ...
                'matrix_size', ...
                'matrixSize', ...
                'voxelSize', ...
                'voxel_size', ...
                'voxel_size_', ...
                'B0_dir', ...
                'b0dir', ...
                'b0', ...
                'B0', ...
                'TE', ...
                'delta_TE', ...
                'CF');

            input = struct();
            input(1).name = localFieldFile;
            input(2).name = magFile;
            input(3).name = '';
            input(4).name = headerFile;

            mask_filename = maskFile;

            QSMMacroIOWrapper(input, output_basename, mask_filename, algorParam);

            qsmFiles = [ ...
                dir(fullfile(tmpDir, '*QSM*.nii')); ...
                dir(fullfile(tmpDir, '*QSM*.nii.gz')); ...
                dir(fullfile(tmpDir, '**', '*QSM*.nii')); ...
                dir(fullfile(tmpDir, '**', '*QSM*.nii.gz'))];

            if isempty(qsmFiles)
                fprintf('  QSMMacroIOWrapper 完成但未找到 QSM NIfTI 输出。\n');
                fprintf('  跳过 WH-QSM/FANSI。\n');
                chi = [];
                return;
            end

            [~, idxNewest] = max([qsmFiles.datenum]);
            qsmPath = fullfile(qsmFiles(idxNewest).folder, qsmFiles(idxNewest).name);

            fprintf('  读取 SEPIA FANSI 输出: %s\n', qsmPath);

            chi_tmp = double(niftiread(qsmPath));
            chi_tmp = squeeze(chi_tmp);
            chi_tmp(~Mask) = 0;

            if is_plausible_qsm(chi_tmp, Mask)
                chi = chi_tmp;
                fprintf('  SEPIA QSMMacroIOWrapper + FANSI 调用成功。\n');
                return;
            else
                fprintf('  SEPIA FANSI 输出异常。\n');
                fprintf('  跳过 WH-QSM/FANSI。\n');
                chi = [];
                return;
            end
        end

    catch ME
        fprintf('  QSMMacroIOWrapper 调用失败: %s\n', ME.message);
        fprintf('  不手动处理 FANSI 内部依赖，继续尝试 SEPIA 低层 wrapper。\n');
    end
else
    fprintf('  未找到 QSMMacroIOWrapper，尝试 SEPIA 低层 FANSI wrapper。\n');
end

%% ------------------------------------------------------------------------
% 6. 备选: 使用 SEPIA 低层 FANSI wrapper
% -------------------------------------------------------------------------
wrapperCandidates = { ...
    'Wrapper_QSM_FANSI', ...
    'Wrapper_QSM_FANSI_v3', ...
    'Wrapper_QSM_FANSI_V3'};

wrapperName = '';

for i = 1:numel(wrapperCandidates)
    if exist(wrapperCandidates{i}, 'file') == 2
        wrapperName = wrapperCandidates{i};
        break;
    end
end

if isempty(wrapperName)
    fprintf('  未找到 SEPIA FANSI wrapper。\n');
    fprintf('  跳过 WH-QSM/FANSI。\n');
    chi = [];
    return;
end

fprintf('  使用 SEPIA 低层 wrapper: %s\n', wrapperName);

headerAndExtraData = struct();

headerAndExtraData.b0dir       = [0 0 1];
headerAndExtraData.B0_dir      = [0 0 1];

headerAndExtraData.b0          = B0;
headerAndExtraData.B0          = B0;

headerAndExtraData.voxelSize   = voxel_size;
headerAndExtraData.voxel_size  = voxel_size;

headerAndExtraData.matrixSize  = matrixSize;
headerAndExtraData.matrix_size = matrixSize;

if isfield(data, 'magn') && ~isempty(data.magn)
    mag = double(data.magn);
    mag(~Mask) = 0;
else
    mag = double(Mask);
end

headerAndExtraData.magn    = mag;
headerAndExtraData.iMag    = mag;
headerAndExtraData.weights = double(Mask);
headerAndExtraData.N_std   = ones(matrixSize);

headerAndExtraData.TE       = 1;
headerAndExtraData.delta_TE = 1;
headerAndExtraData.CF       = B0 * gyro;

try
    wrapperFunc = str2func(wrapperName);

    chi_tmp = wrapperFunc( ...
        localField_Hz, ...
        Mask, ...
        matrixSize, ...
        voxel_size, ...
        algorParam, ...
        headerAndExtraData);

    chi_tmp = double(real(chi_tmp));
    chi_tmp(~Mask) = 0;

    if is_plausible_qsm(chi_tmp, Mask)
        chi = chi_tmp;
        fprintf('  SEPIA FANSI wrapper 调用成功。\n');
        return;
    else
        fprintf('  SEPIA FANSI wrapper 输出异常。\n');
        fprintf('  跳过 WH-QSM/FANSI。\n');
        chi = [];
        return;
    end

catch ME
    fprintf('  SEPIA FANSI wrapper 调用失败: %s\n', ME.message);
    fprintf('  不手动引入 FANSI 内部依赖。\n');
    fprintf('  请确认 SEPIA 已正确配置 FANSI_HOME。\n');
    fprintf('  跳过 WH-QSM/FANSI。\n');
    chi = [];
    return;
end

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