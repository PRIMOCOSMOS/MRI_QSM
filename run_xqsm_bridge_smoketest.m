function run_xqsm_bridge_smoketest()
% run_xqsm_bridge_smoketest.m
% xQSM Python bridge 冒烟测试（不跑全 pipeline）
%
% 目标:
% 1) 验证配置是否可加载
% 2) 验证 xQSM checkpoint / repo / bridge 脚本路径
% 3) 用 data/phs_tissue + data/msk 跑一次 xQSM bridge
% 4) 输出一个 NIfTI 和一个 MAT，便于人工检查
%
% 运行方式:
%   在项目根目录 MATLAB 命令行执行:
%       run_xqsm_bridge_smoketest

clc;
fprintf('\n================ xQSM Bridge Smoke Test ================\n');

repoRoot = fileparts(mfilename('fullpath'));

% 加载项目路径
addpath(fullfile(repoRoot, 'config'));
addpath(fullfile(repoRoot, 'modules'));
addpath(fullfile(repoRoot, 'modules', 'DL'));
addpath(fullfile(repoRoot, 'Utils_self'));

% -------------------------------------------------------------------------
% 1) 读取配置
% -------------------------------------------------------------------------
cfg = pipeline_config();

fprintf('RootDir         : %s\n', cfg.rootDir);
fprintf('DataDir         : %s\n', cfg.dataDir);
fprintf('ResultDir       : %s\n', cfg.resultDir);
fprintf('xQSM checkpoint : %s\n', cfg.deeplearning.xqsm_pth);
fprintf('xQSM repo root  : %s\n', cfg.deeplearning.xqsm_repo_root);
fprintf('Bridge script   : %s\n', cfg.deeplearning.xqsm_bridge_script);

assert(exist(cfg.dataDir, 'dir') == 7, 'DataDir 不存在: %s', cfg.dataDir);
assert(exist(cfg.resultDir, 'dir') == 7, 'ResultDir 不存在: %s', cfg.resultDir);
assert(exist(cfg.deeplearning.xqsm_pth, 'file') == 2, ...
    'xQSM checkpoint 不存在: %s', cfg.deeplearning.xqsm_pth);
assert(exist(cfg.deeplearning.xqsm_repo_root, 'dir') == 7, ...
    'xQSM repo root 不存在: %s', cfg.deeplearning.xqsm_repo_root);
assert(exist(cfg.deeplearning.xqsm_bridge_script, 'file') == 2, ...
    'bridge script 不存在: %s', cfg.deeplearning.xqsm_bridge_script);
assert(exist(fullfile(cfg.deeplearning.xqsm_repo_root, 'python', 'xQSM.py'), 'file') == 2, ...
    'xQSM.py 不存在于 repo: %s', fullfile(cfg.deeplearning.xqsm_repo_root, 'python', 'xQSM.py'));

% -------------------------------------------------------------------------
% 2) 读入最小测试数据（官方 challenge 的 phs_tissue + msk）
% -------------------------------------------------------------------------
phs_tissue = load_var_local(cfg.dataDir, 'phs_tissue');
msk = load_var_local(cfg.dataDir, 'msk');

field = double(phs_tissue);
Mask = logical(msk);

assert(ndims(field) == 3, 'phs_tissue 必须是 3D');
assert(isequal(size(field), size(Mask)), 'phs_tissue 与 msk 尺寸不一致');

field(~Mask) = 0;
N = size(field);

fprintf('Input shape     : [%d %d %d]\n', N(1), N(2), N(3));
fprintf('Mask voxels     : %d\n', nnz(Mask));
fprintf('Field range(ppm): [%.6f, %.6f]\n', min(field(Mask)), max(field(Mask)));

% -------------------------------------------------------------------------
% 3) 预处理并调用 xQSM bridge
% -------------------------------------------------------------------------
[input_norm, norm_factor] = dl_prepare_input(field, Mask);
fprintf('norm_factor     : %.6g\n', norm_factor);

% 这里 voxel_size 对当前 bridge 不是必须参数，填 [1 1 1] 即可
voxel_size = [1 1 1];

tic;
chi = dl_python_bridge('xqsm', input_norm, Mask, N, voxel_size, norm_factor, cfg);
t_elapsed = toc;

assert(~isempty(chi), 'xQSM bridge 返回空结果');
assert(isequal(size(chi), N), '输出尺寸不匹配');
assert(all(isfinite(chi(Mask))), '输出含 NaN/Inf');

fprintf('Inference time  : %.2f sec\n', t_elapsed);
fprintf('Output range    : [%.6f, %.6f]\n', min(chi(Mask)), max(chi(Mask)));
fprintf('Output std      : %.6f\n', std(chi(Mask)));

% -------------------------------------------------------------------------
% 4) 保存输出
% -------------------------------------------------------------------------
outMat = fullfile(cfg.resultDir, 'smoketest_xqsm_output.mat');
save(outMat, 'chi', 'field', 'Mask', 'norm_factor', '-v7.3');
fprintf('Saved MAT       : %s\n', outMat);

% 可选 NIfTI 输出（如果有 NIfTI toolbox）
outNii = fullfile(cfg.resultDir, 'smoketest_xqsm_output.nii');
try
    nii = make_nii(single(chi), [1 1 1]);
    save_nii(nii, outNii);
    fprintf('Saved NIfTI     : %s\n', outNii);
catch ME
    fprintf('NIfTI 未保存(可忽略): %s\n', ME.message);
end

fprintf('================ Smoke Test Passed ================\n\n');

end

%% =========================================================================
function v = load_var_local(folder, varname)
f1 = fullfile(folder, [varname '.mat']);
f2 = fullfile(folder, varname);

if exist(f1, 'file')
    S = load(f1);
elseif exist(f2, 'file')
    S = load(f2);
else
    error('找不到变量文件: %s 或 %s', f1, f2);
end

if isfield(S, varname)
    v = S.(varname);
else
    names = fieldnames(S);
    if numel(names) == 1
        v = S.(names{1});
    else
        error('文件 "%s" 中存在多个变量且无同名字段', varname);
    end
end
end