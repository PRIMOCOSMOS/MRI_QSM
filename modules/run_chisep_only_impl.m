function run_chisep_only_impl(whichSubject)
% RUN_CHISEP_ONLY.m
% ============================================================================
% Run only the susceptibility source separation stage using existing WH-QSM
% outputs. This avoids re-running DICOM loading and WH-QSM.
%
% Usage from project root:
%   RUN_CHISEP_ONLY              % process normal_* and elderly_* if present
%   RUN_CHISEP_ONLY('normal')
%   RUN_CHISEP_ONLY('elderly')
%   RUN_CHISEP_ONLY('D:\...\normal_swi_subj1')
% ============================================================================

if nargin < 1, whichSubject = 'all'; end

repoRoot = fileparts(fileparts(mfilename('fullpath')));
adapterDir = fullfile(repoRoot, 'MRI_QSM_dicom_adapter');
addpath(repoRoot, '-begin');
addpath(adapterDir, '-begin');
addpath(fullfile(repoRoot, 'modules'), '-begin');
addpath(fullfile(repoRoot, 'Utils_self'), '-begin');

P = whqsm_local_paths();
addpath(P.projectRoot, '-begin');
addpath(P.adapterDir, '-begin');
addpath(P.modulesDir, '-begin');
addpath(P.utilsDir, '-begin');
if isfield(P,'sepiaRoot') && exist(P.sepiaRoot,'dir') == 7
    addpath(P.sepiaRoot, '-begin'); addpath(genpath(P.sepiaRoot), '-begin');
end
if isfield(P,'mediRoot') && exist(P.mediRoot,'dir') == 7
    addpath(P.mediRoot, '-begin'); addpath(genpath(P.mediRoot), '-begin');
end
if isfield(P,'chiSepRoot') && exist(P.chiSepRoot,'dir') == 7
    addpath(P.chiSepRoot, '-begin'); addpath(genpath(P.chiSepRoot), '-begin');
end
% Put project modules first again so importONNXNetwork shim wins.
addpath(P.modulesDir, '-begin');
clear functions; clear mex; rehash;

outRoot = fullfile(P.dataRoot, '_qsm_comparison_results');
if exist(outRoot, 'dir') ~= 7
    error('QSM comparison output root not found: %s. Run WH-QSM first.', outRoot);
end

subjectDirs = resolve_subject_dirs(outRoot, whichSubject);
if isempty(subjectDirs)
    error('No matching subject result directories found under %s', outRoot);
end

fprintf('\n============================================================\n');
fprintf(' CHI-SEPARATION ONLY ENTRY\n');
fprintf('============================================================\n');
fprintf('Subjects: %d\n', numel(subjectDirs));
fprintf('chiSepRoot: %s\n', P.chiSepRoot);
fprintf('adapter   : %s\n', P.chiSepAdapterFunction);

for i = 1:numel(subjectDirs)
    subDir = subjectDirs{i};
    fprintf('\n------------------------------------------------------------\n');
    fprintf('Subject result dir: %s\n', subDir);
    try
        [data, chi, cfg, label] = load_existing_whqsm_subject(subDir, P);
    catch ME
        warning('被试加载失败，跳过: %s\n  原因: %s', subDir, ME.message);
        if contains(lower(ME.message),'corrupt') || contains(ME.message,'损坏') ...
                || contains(lower(ME.message),'cannot read')
            fprintf(['  提示: whqsm_*_complete.mat 可能损坏(常因 WH-QSM 中断或磁盘写入未完成)。\n' ...
                     '        请对该被试重跑 WH-QSM 生成完整结果文件。\n']);
        end
        continue;
    end

    try
        cfg = configure_sep_cfg(cfg, P, subDir);
        runCompare = isfield(P,'chisepRunMethodCompare') && P.chisepRunMethodCompare;
        if runCompare
            % Method comparison mode: deep-learning chi-sepnet vs optimization.
            cmp = mod_chisep_method_comparison(data, chi, cfg); %#ok<NASGU>
            save(fullfile(subDir, ['chisep_compare_' label '.mat']), 'cmp', 'cfg', '-v7.3');
            fprintf('Saved chi-separation method-comparison: %s\n', ...
                fullfile(subDir, ['chisep_compare_' label '.mat']));
        else
            % Single-method separation (uses cfg.sep.adapter_function).
            sep_results = mod_susceptibility_separation(data, chi, cfg); %#ok<NASGU>
            save(fullfile(subDir, ['chisep_only_' label '.mat']), 'sep_results', 'cfg', '-v7.3');
            fprintf('Saved chi-separation-only result marker: %s\n', ...
                fullfile(subDir, ['chisep_only_' label '.mat']));
        end
    catch ME
        warning('被试处理失败，跳过继续: %s\n  原因: %s', subDir, ME.message);
        continue;
    end
end

fprintf('\n✅ CHI-SEPARATION ONLY completed.\n');
end

%% =========================================================================
function subjectDirs = resolve_subject_dirs(outRoot, whichSubject)
whichSubject = char(whichSubject);
if exist(whichSubject, 'dir') == 7
    subjectDirs = {whichSubject};
    return;
end
allDirs = [dir(fullfile(outRoot, 'normal_*')); dir(fullfile(outRoot, 'elderly_*'))];
allDirs = allDirs([allDirs.isdir]);
subjectDirs = {};
for i = 1:numel(allDirs)
    name = allDirs(i).name;
    if strcmpi(whichSubject, 'all') || contains(lower(name), lower(whichSubject))
        subjectDirs{end+1} = fullfile(allDirs(i).folder, name); %#ok<AGROW>
    end
end
end

function [data, chi, cfg, label] = load_existing_whqsm_subject(subDir, P)
[~, name] = fileparts(subDir);
if startsWith(lower(name), 'normal_')
    label = 'normal';
elseif startsWith(lower(name), 'elderly_')
    label = 'elderly';
else
    label = 'subject';
end

completeFile = fullfile(subDir, ['whqsm_' label '_complete.mat']);
if exist(completeFile, 'file') ~= 2
    % fallback: find any complete mat
    d = dir(fullfile(subDir, 'whqsm_*_complete.mat'));
    if isempty(d)
        error('No whqsm_*_complete.mat found in %s', subDir);
    end
    completeFile = fullfile(d(1).folder, d(1).name);
end
% Robust load: if complete.mat is corrupt/truncated, fall back to the
% companion chi_<grp>.mat (+ qsm2016_format/data_full.mat) to reconstruct.
S = [];
loadErr = '';
try
    S = load(completeFile);
catch ME
    loadErr = ME.message;
    warning('complete.mat 无法读取(可能损坏): %s\n  尝试从备用文件恢复...', completeFile);
end

if isempty(S) || ~isfield(S, 'data') || ~isfield(S, 'chi')
    [S, recMsg] = recover_subject_from_parts(subDir, label);
    if isempty(S)
        error(['无法加载被试且恢复失败: %s\n  complete.mat 错误: %s\n  恢复信息: %s\n' ...
               '  建议: 对该被试重跑 WH-QSM(现已使用原子+校验保存，可避免再次损坏)。'], ...
               subDir, loadErr, recMsg);
    end
    fprintf('  ✅ 已从备用文件恢复该被试 (chi_%s.mat + data_full.mat)。\n', label);
end

data = S.data;
chi = S.chi;
if isfield(S, 'cfg')
    cfg = S.cfg;
else
    cfg = struct();
end

% If old complete file lacks R2star_Hz, try qsm2016_format data_full.
if ~isfield(data, 'R2star_Hz') || isempty(data.R2star_Hz)
    df = fullfile(subDir, 'qsm2016_format', 'data_full.mat');
    if exist(df, 'file') == 2
        T = load(df);
        if isfield(T, 'data') && isfield(T.data, 'R2star_Hz')
            data.R2star_Hz = T.data.R2star_Hz;
            data.R2star_s0 = T.data.R2star_s0;
            data.R2star_fit_residual = T.data.R2star_fit_residual;
        end
    end
end
if ~isfield(data, 'R2star_Hz') || isempty(data.R2star_Hz)
    error(['Existing WH-QSM data do not contain R2star_Hz. ' ...
           'Run the updated DICOM loader/WH-QSM once, or provide R2star_Hz.mat. Subject: %s'], subDir);
end
end

function cfg = configure_sep_cfg(cfg, P, subDir)
if ~isfield(cfg, 'resultDir') || isempty(cfg.resultDir)
    cfg.resultDir = fullfile(subDir, 'results');
end
if ~exist(cfg.resultDir, 'dir'), mkdir(cfg.resultDir); end
cfg.sep.enable = true;
cfg.sep.method = P.suscepSepMethod;
cfg.sep.chiSepRoot = P.chiSepRoot;
cfg.sep.adapter_function = P.chiSepAdapterFunction;
cfg.sep.allow_exploratory_fallback = P.allowExploratorySeparationFallback;
cfg.sep.r2star_to_chi_abs_HzPerPpm = P.r2starToChiAbsHzPerPpm;
cfg.sep.snu_local_field_mode = P.snuLocalFieldMode;
cfg.sep.snu_resgen = P.snuResgen;
cfg.sep.snu_HaveR2Prime = P.snuHaveR2Prime;
cfg.sep.snu_is_scaling = P.snuIsScaling;
cfg.sep.snu_scaling_factor = P.snuScalingFactor;
cfg.sep.snu_interp_method = P.snuInterpMethod;
cfg.sep.snu_sinc_window_size = P.snuSincWindowSize;
cfg.sep.snu_sinc_window_type = P.snuSincWindowType;
cfg.sep.snu_Dr = P.snuDr;

% ----- ONNX Runtime chi-separation 桥接配置（绕过 onnxmex） -----
cfg.sep.onnx_python_executable = get_field_default(P, 'onnxPythonExecutable', '');
cfg.sep.onnx_bridge_script     = get_field_default(P, 'onnxBridgeScript', '');
cfg.sep.onnx_qsm_model         = get_field_default(P, 'onnxQsmModel', '');
cfg.sep.onnx_xsep_model        = get_field_default(P, 'onnxXsepModel', '');
cfg.sep.onnx_r2prime_model     = get_field_default(P, 'onnxR2primeModel', '');
cfg.sep.onnx_norm_factor       = get_field_default(P, 'onnxNormFactor', '');
cfg.sep.onnx_pipeline          = get_field_default(P, 'onnxPipeline', 'auto');
cfg.sep.onnx_qsm_source        = get_field_default(P, 'onnxQsmSource', 'qsmnet');
cfg.sep.onnx_field_unit        = get_field_default(P, 'onnxFieldUnit', 'Hz');
cfg.sep.onnx_device            = get_field_default(P, 'onnxDevice', 'auto');
cfg.sep.onnx_resgen            = get_field_default(P, 'onnxResgen', 'auto');
cfg.sep.onnx_r2_map            = get_field_default(P, 'onnxR2Map', []);
% method-comparison / optimization fields
cfg.sep.compare_methods        = get_field_default(P, 'chisepCompareMethods', {'onnx','opt'});
cfg.sep.opt_method             = get_field_default(P, 'optMethod', 'iLSQR');
cfg.sep.opt_lambda             = get_field_default(P, 'optLambda', 1e-2);
cfg.sep.opt_w_r2               = get_field_default(P, 'optWr2', 1.0);
cfg.sep.opt_maxiter            = get_field_default(P, 'optMaxIter', 100);
cfg.sep.roi_label_file         = get_field_default(P, 'roiLabelFile', '');
end

function v = get_field_default(S, name, default)
if isfield(S, name) && ~isempty(S.(name))
    v = S.(name);
else
    v = default;
end
end

function [S, msg] = recover_subject_from_parts(subDir, label)
% Reconstruct a usable {data, chi, cfg} struct from companion files when
% whqsm_*_complete.mat is corrupt:
%   - chi_<label>.mat        -> chi (+ sep_results)
%   - qsm2016_format/data_full.mat -> data (with R2star_Hz etc.)
S = []; msg = '';
chiFile = fullfile(subDir, ['chi_' label '.mat']);
dataFull = fullfile(subDir, 'qsm2016_format', 'data_full.mat');

chi = [];
try
    if exist(chiFile,'file') == 2
        C = load(chiFile);
        if isfield(C,'chi'), chi = C.chi; end
    end
catch ME
    msg = [msg sprintf('chi_%s.mat 读取失败: %s; ', label, ME.message)];
end

data = [];
try
    if exist(dataFull,'file') == 2
        D = load(dataFull);
        if isfield(D,'data'), data = D.data; end
    end
catch ME
    msg = [msg sprintf('data_full.mat 读取失败: %s; ', ME.message)];
end

if isempty(chi)
    msg = [msg '未能获得 chi（chi_*.mat 缺失或损坏）; ']; return;
end
if isempty(data) || ~isfield(data,'Mask')
    msg = [msg '未能获得 data/Mask（data_full.mat 缺失或损坏）; ']; return;
end

% If chi grid mismatches data.Mask, abort (cannot safely proceed).
if ~isequal(size(chi), size(data.Mask))
    msg = [msg sprintf('chi 尺寸 %s 与 data.Mask %s 不一致; ', ...
        mat2str(size(chi)), mat2str(size(data.Mask)))]; return;
end

S = struct('data', data, 'chi', chi, 'cfg', struct());
msg = 'recovered from chi_*.mat + data_full.mat';
end
