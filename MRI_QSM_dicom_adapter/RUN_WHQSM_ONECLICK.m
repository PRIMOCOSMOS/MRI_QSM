function RUN_WHQSM_ONECLICK()
% RUN_WHQSM_ONECLICK.m
% ============================================================================
% 一键运行两个真实被试 WH-QSM-only Pipeline。
%
% 使用方式：
%   1) 打开 MATLAB
%   2) cd 到 MRI_QSM/MRI_QSM_dicom_adapter
%   3) 命令行输入：RUN_WHQSM_ONECLICK
%
% 所有固化路径在 whqsm_local_paths.m 中维护。
% ============================================================================

clc;
fprintf('\n');
fprintf('╔══════════════════════════════════════════════════════════════╗\n');
fprintf('║  ONE-CLICK REAL-DATA WH-QSM PIPELINE                        ║\n');
fprintf('║  fixed paths → DICOM multi-echo fit → SEPIA/FANSI WH-QSM    ║\n');
fprintf('╚══════════════════════════════════════════════════════════════╝\n\n');

% -------------------------------------------------------------------------
% 1. Load fixed paths
% -------------------------------------------------------------------------
thisDir = fileparts(mfilename('fullpath'));
addpath(thisDir);
P = whqsm_local_paths();

fprintf('[1/5] Fixed paths\n');
fprintf('  projectRoot : %s\n', P.projectRoot);
fprintf('  adapterDir  : %s\n', P.adapterDir);
fprintf('  dataRoot    : %s\n', P.dataRoot);
fprintf('  sepiaRoot   : %s\n', P.sepiaRoot);
if isfield(P, 'mediRoot'), fprintf('  mediRoot    : %s\n', P.mediRoot); end
fprintf('  keepWork    : %d\n', P.keepSepiaWork);
fprintf('  maskMethod  : %s\n', P.maskMethod);
fprintf('  maskErodeMm : %.3g\n', P.maskErodeMm);
fprintf('  maskThrFact : %.3g\n', P.maskThresholdFactor);
fprintf('  phaseQMask  : %d | pct=%.1f | MADx=%.2f | maxRad=%.3g | maxRel=%.3g | minKeep=%.2f\n', ...
    get_p(P,'usePhaseQualityMask',true), get_p(P,'phaseResidualPercentile',97.5), ...
    get_p(P,'phaseResidualMadFactor',4.0), get_p(P,'phaseResidualMaxRad',0.35), ...
    get_p(P,'phaseRelativeResidualMax',0.50), get_p(P,'phaseQualityMinKeepFraction',0.85));
fprintf('  edgeRefine  : %d | band=%.2f mm | pct=%.1f | absMaxHz=%.1f | postBfrErode=%.2f mm\n', ...
    get_p(P,'useR2starEdgeRefine',true), get_p(P,'r2starEdgeBandMm',2.0), ...
    get_p(P,'r2starEdgePercentile',97.5), get_p(P,'r2starEdgeAbsMaxHz',80), ...
    get_p(P,'postBfrErodeMm',1.0));
fprintf('  twoPassQSM  : %d | useLastEcho=%d | otsuFactor=%.2f | shell=%.2f mm\n', ...
    get_p(P,'useTwoPassQSM',true), get_p(P,'twoPassMaskUseLastEcho',true), ...
    get_p(P,'twoPassMaskOtsuFactor',1.3), get_p(P,'twoPassMaskShellMm',3.0));
fprintf('  runChiSep   : %d\n', P.runSusceptibilitySeparation);
fprintf('  chiSepRoot  : %s\n', P.chiSepRoot);
fprintf('  whqsmIter   : %d\n', P.whqsmMaxIter);
fprintf('  whqsmTol    : %.3g\n', P.whqsmTol);
fprintf('  whqsmLambda : %.3g\n', P.whqsmLambda);

% -------------------------------------------------------------------------
% 2. Deterministic path setup
% -------------------------------------------------------------------------
fprintf('\n[2/5] MATLAB path setup\n');
if isfield(P, 'resetMatlabPath') && P.resetMatlabPath
    restoredefaultpath;
    rehash toolboxcache;
    fprintf('  restoredefaultpath done.\n');
end

addpath(P.projectRoot, '-begin');
addpath(P.adapterDir, '-begin');
addpath(fullfile(P.projectRoot, 'modules'), '-begin');
addpath(fullfile(P.projectRoot, 'Utils_self'), '-begin');

if exist(P.sepiaRoot, 'dir') == 7
    addpath(P.sepiaRoot, '-begin');
    addpath(genpath(P.sepiaRoot), '-begin');
else
    error('SEPIA root does not exist: %s', P.sepiaRoot);
end

if isfield(P, 'mediRoot') && ~isempty(P.mediRoot) && exist(P.mediRoot, 'dir') == 7
    addpath(P.mediRoot, '-begin');
    addpath(genpath(P.mediRoot), '-begin');
    fprintf('  MEDI path added for BET/mask utilities.\n');
else
    fprintf('  MEDI root not found or not configured; BET fallback may be unavailable.\n');
end

if isfield(P, 'chiSepRoot') && ~isempty(P.chiSepRoot) && exist(P.chiSepRoot, 'dir') == 7
    addpath(P.chiSepRoot, '-begin');
    addpath(genpath(P.chiSepRoot), '-begin');
    fprintf('  chi-separation toolbox path added.\n');
else
    fprintf('  chi-separation toolbox root not found; separation will export inputs and skip unless fallback is enabled.\n');
end

if exist('sepia_addpath', 'file') == 2
    try
        sepia_addpath;
        fprintf('  sepia_addpath OK.\n');
    catch ME
        warning('sepia_addpath failed, continuing with genpath: %s', ME.message);
    end
end
rehash;
fprintf('  Path setup done.\n');

% -------------------------------------------------------------------------
% 3. Preflight checks and logging
% -------------------------------------------------------------------------
fprintf('\n[3/5] Preflight checks\n');
assert_dir(P.projectRoot, 'projectRoot');
assert_dir(P.adapterDir, 'adapterDir');
assert_dir(P.dataRoot, 'dataRoot');
assert_dir(P.sepiaRoot, 'sepiaRoot');
if isfield(P, 'mediRoot') && ~isempty(P.mediRoot) && exist(P.mediRoot, 'dir') == 7
    fprintf('  ✅ %-24s %s\n', 'mediRoot', P.mediRoot);
else
    fprintf('  ⚠️ %-24s %s\n', 'mediRoot not found', getfield_or_local(P, 'mediRoot', '<empty>'));
end
assert_file(fullfile(P.adapterDir, 'run_whqsm_comparison.m'), 'run_whqsm_comparison.m');
assert_file(fullfile(P.adapterDir, 'dicom_loader_subject.m'), 'dicom_loader_subject.m');
assert_file(fullfile(P.projectRoot, 'modules', 'mod_whqsm_reconstruction.m'), 'mod_whqsm_reconstruction.m');
assert_func('dicominfo');
assert_func('dicomread');
assert_func('niftiwrite');
assert_func('niftiread');
assert_func('QSMMacroIOWrapper');

outRoot = fullfile(P.dataRoot, '_qsm_comparison_results');
if ~exist(outRoot, 'dir'), mkdir(outRoot); end
if isfield(P, 'logDir') && ~isempty(P.logDir)
    logDir = P.logDir;
else
    logDir = fullfile(outRoot, 'logs');
end
if ~exist(logDir, 'dir'), mkdir(logDir); end
logFile = fullfile(logDir, ['RUN_WHQSM_ONECLICK_' datestr(now, 'yyyymmdd_HHMMSS') '.log']);
fprintf('  logFile: %s\n', logFile);

diary(logFile);
cleanupObj = onCleanup(@() diary('off')); %#ok<NASGU>

fprintf('\n========== ONE-CLICK WH-QSM LOG START ==========%s\n', datestr(now));
fprintf('projectRoot = %s\n', P.projectRoot);
fprintf('dataRoot    = %s\n', P.dataRoot);
fprintf('sepiaRoot   = %s\n', P.sepiaRoot);
if isfield(P, 'mediRoot'), fprintf('mediRoot    = %s\n', P.mediRoot); end
fprintf('maskMethod  = %s\n', P.maskMethod);
fprintf('maskErodeMm = %.3g\n', P.maskErodeMm);
fprintf('maskThresholdFactor = %.3g\n', P.maskThresholdFactor);
fprintf('usePhaseQualityMask = %d | phaseResidualPercentile = %.1f | phaseResidualMadFactor = %.2f | phaseResidualMaxRad = %.3g | phaseRelativeResidualMax = %.3g | phaseQualityMinKeepFraction = %.2f\n', ...
    get_p(P,'usePhaseQualityMask',true), get_p(P,'phaseResidualPercentile',97.5), ...
    get_p(P,'phaseResidualMadFactor',4.0), get_p(P,'phaseResidualMaxRad',0.35), ...
    get_p(P,'phaseRelativeResidualMax',0.50), get_p(P,'phaseQualityMinKeepFraction',0.85));
fprintf('useR2starEdgeRefine = %d | r2starEdgeBandMm = %.2f | r2starEdgePercentile = %.1f | r2starEdgeAbsMaxHz = %.1f | postBfrErodeMm = %.2f\n', ...
    get_p(P,'useR2starEdgeRefine',true), get_p(P,'r2starEdgeBandMm',2.0), ...
    get_p(P,'r2starEdgePercentile',97.5), get_p(P,'r2starEdgeAbsMaxHz',80), get_p(P,'postBfrErodeMm',1.0));
fprintf('useTwoPassQSM = %d | twoPassMaskUseLastEcho = %d | twoPassMaskOtsuFactor = %.2f | twoPassMaskShellMm = %.2f\n', ...
    get_p(P,'useTwoPassQSM',true), get_p(P,'twoPassMaskUseLastEcho',true), ...
    get_p(P,'twoPassMaskOtsuFactor',1.3), get_p(P,'twoPassMaskShellMm',3.0));
fprintf('runSusceptibilitySeparation = %d\n', P.runSusceptibilitySeparation);
fprintf('chiSepRoot = %s\n', P.chiSepRoot);
fprintf('whqsmMaxIter = %d\n', P.whqsmMaxIter);
fprintf('whqsmTol     = %.3g\n', P.whqsmTol);
fprintf('whqsmLambda  = %.3g\n', P.whqsmLambda);
fprintf('MATLAB      = %s\n', version);
fprintf('QSMMacroIOWrapper = %s\n', which('QSMMacroIOWrapper'));

% -------------------------------------------------------------------------
% 4. Subject discovery preview
% -------------------------------------------------------------------------
fprintf('\n[4/5] Subject discovery preview\n');
subjects = discover_subjects(P.dataRoot);
nNormal = sum(strcmp({subjects.group}, 'NORMAL'));
nElderly = sum(strcmp({subjects.group}, 'ELDERLY'));
fprintf('  NORMAL=%d, ELDERLY=%d\n', nNormal, nElderly);
if nNormal < 1 || nElderly < 1
    error('One-click run needs at least one NORMAL and one ELDERLY subject. Please check folder names or DICOM age metadata.');
end

% -------------------------------------------------------------------------
% 5. Run WH-QSM-only pipeline
% -------------------------------------------------------------------------
fprintf('\n[5/5] Running WH-QSM-only pipeline\n');
run_whqsm_comparison(P.dataRoot, P.projectRoot, ...
    'sepia_root', P.sepiaRoot, ...
    'keep_sepia_work', P.keepSepiaWork, ...
    'mask_method', P.maskMethod, ...
    'mask_erode_mm', P.maskErodeMm, ...
    'mask_threshold_factor', P.maskThresholdFactor, ...
    'bet_fractional_threshold', P.betFractionalThreshold, ...
    'bet_vertical_gradient', P.betVerticalGradient, ...
    'use_phase_quality_mask', get_p(P,'usePhaseQualityMask',true), ...
    'phase_residual_percentile', get_p(P,'phaseResidualPercentile',97.5), ...
    'phase_residual_mad_factor', get_p(P,'phaseResidualMadFactor',4.0), ...
    'phase_residual_max_rad', get_p(P,'phaseResidualMaxRad',0.35), ...
    'phase_relative_residual_max', get_p(P,'phaseRelativeResidualMax',0.50), ...
    'phase_quality_min_keep_fraction', get_p(P,'phaseQualityMinKeepFraction',0.85), ...
    'use_r2star_edge_refine', get_p(P,'useR2starEdgeRefine',true), ...
    'r2star_edge_band_mm', get_p(P,'r2starEdgeBandMm',2.0), ...
    'r2star_edge_percentile', get_p(P,'r2starEdgePercentile',97.5), ...
    'r2star_edge_abs_max_hz', get_p(P,'r2starEdgeAbsMaxHz',80), ...
    'post_bfr_erode_mm', get_p(P,'postBfrErodeMm',1.0), ...
    'use_two_pass_qsm', get_p(P,'useTwoPassQSM',true), ...
    'two_pass_mask_use_last_echo', get_p(P,'twoPassMaskUseLastEcho',true), ...
    'two_pass_mask_otsu_factor', get_p(P,'twoPassMaskOtsuFactor',1.3), ...
    'two_pass_mask_shell_mm', get_p(P,'twoPassMaskShellMm',3.0), ...
    'run_susceptibility_separation', P.runSusceptibilitySeparation, ...
    'chi_sep_root', P.chiSepRoot, ...
    'chi_sep_adapter_function', P.chiSepAdapterFunction, ...
    'suscep_sep_method', P.suscepSepMethod, ...
    'allow_exploratory_separation_fallback', P.allowExploratorySeparationFallback, ...
    'r2star_to_chi_abs_HzPerPpm', P.r2starToChiAbsHzPerPpm, ...
    'snu_local_field_mode', P.snuLocalFieldMode, ...
    'snu_resgen', P.snuResgen, ...
    'snu_HaveR2Prime', P.snuHaveR2Prime, ...
    'snu_is_scaling', P.snuIsScaling, ...
    'snu_scaling_factor', P.snuScalingFactor, ...
    'snu_interp_method', P.snuInterpMethod, ...
    'snu_sinc_window_size', P.snuSincWindowSize, ...
    'snu_sinc_window_type', P.snuSincWindowType, ...
    'snu_Dr', P.snuDr, ...
    'onnx_python_executable', get_p(P,'onnxPythonExecutable',''), ...
    'onnx_bridge_script', get_p(P,'onnxBridgeScript',''), ...
    'onnx_qsm_model', get_p(P,'onnxQsmModel',''), ...
    'onnx_xsep_model', get_p(P,'onnxXsepModel',''), ...
    'onnx_r2prime_model', get_p(P,'onnxR2primeModel',''), ...
    'onnx_norm_factor', get_p(P,'onnxNormFactor',''), ...
    'onnx_pipeline', get_p(P,'onnxPipeline','auto'), ...
    'onnx_qsm_source', get_p(P,'onnxQsmSource','qsmnet'), ...
    'onnx_field_unit', get_p(P,'onnxFieldUnit','Hz'), ...
    'onnx_device', get_p(P,'onnxDevice','auto'), ...
    'onnx_resgen', get_p(P,'onnxResgen','auto'), ...
    'onnx_r2_map', get_p(P,'onnxR2Map',[]), ...
    'whqsm_maxiter', P.whqsmMaxIter, ...
    'whqsm_tol', P.whqsmTol, ...
    'whqsm_lambda', P.whqsmLambda, ...
    'whqsm_beta', P.whqsmBeta, ...
    'whqsm_muh', P.whqsmMuh);

fprintf('\n========== ONE-CLICK WH-QSM LOG END ==========%s\n', datestr(now));
fprintf('\n✅ One-click WH-QSM completed. Output root:\n  %s\n', outRoot);
fprintf('Log file:\n  %s\n\n', logFile);

end

%% =========================================================================
function assert_dir(p, label)
if exist(p, 'dir') ~= 7
    error('%s does not exist: %s', label, p);
end
fprintf('  ✅ %-24s %s\n', label, p);
end

function assert_file(p, label)
if exist(p, 'file') ~= 2
    error('%s missing: %s', label, p);
end
fprintf('  ✅ %-24s %s\n', label, p);
end

function assert_func(name)
if exist(name, 'file') ~= 2 && exist(name, 'builtin') ~= 5
    error('Required function not found on MATLAB path: %s', name);
end
fprintf('  ✅ %-24s %s\n', name, which(name));
end

function v = getfield_or_local(s, name, default)
if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
    v = s.(name);
else
    v = default;
end
end

function v = get_p(s, name, default)
% Alias used for optional ONNX-bridge fields in whqsm_local_paths.m.
if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
    v = s.(name);
else
    v = default;
end
end
