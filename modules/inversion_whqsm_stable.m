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

function vol = remove_mask_mean(vol, Mask)

Mask = logical(Mask);
v = vol(Mask);
v = v(isfinite(v));

if ~isempty(v)
    vol(Mask) = vol(Mask) - mean(v);
end

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

