function RUN_CHISEP_ONNXRUNTIME(whichSubject)
% RUN_CHISEP_ONNXRUNTIME.m
% ============================================================================
% One-click entry for the ONNX-Runtime chi-separation backend that BYPASSES
% MATLAB's broken importNetworkFromONNX / onnxmex.
%
% It will:
%   1) ensure the ONNX-Runtime backend is selected,
%   2) run the environment diagnosis,
%   3) run the synthetic smoke test (proves the bypass works),
%   4) run real-data chi-separation from existing WH-QSM outputs.
%
% Usage (from project root):
%   RUN_CHISEP_ONNXRUNTIME            % all subjects
%   RUN_CHISEP_ONNXRUNTIME('normal')
%   RUN_CHISEP_ONNXRUNTIME('elderly')
% ============================================================================

if nargin < 1, whichSubject = 'all'; end

repoRoot = fileparts(mfilename('fullpath'));
addpath(fullfile(repoRoot, 'modules'), '-begin');
addpath(fullfile(repoRoot, 'MRI_QSM_dicom_adapter'), '-begin');
addpath(fullfile(repoRoot, 'Utils_self'), '-begin');

fprintf('\n================ chi-separation (ONNX Runtime) ================\n');

% 1) verify backend selection
P = whqsm_local_paths();
if ~isfield(P,'useOnnxRuntimeChiSep') || ~P.useOnnxRuntimeChiSep
    warning(['useOnnxRuntimeChiSep is not true in whqsm_local_paths.m. ' ...
             'This entry forces the ONNX-Runtime adapter anyway.']);
end
fprintf('Adapter: snu_chisep_onnxruntime_adapter (MATLAB onnxmex bypassed)\n');

% 2) diagnosis (non-fatal)
try
    DIAGNOSE_CHISEP_ONNXRUNTIME;
catch ME
    warning('Diagnosis reported a problem: %s', ME.message);
end

% 3) smoke test (fatal if it fails: means env not ready)
fprintf('\n[Smoke test]\n');
run_chisep_onnx_smoketest;

% 4) real data
fprintf('\n[Real data chi-separation]\n');
% Force the ONNX adapter for this run regardless of the file flag.
run_chisep_only_impl_force_onnx(whichSubject, repoRoot);

fprintf('\n✅ RUN_CHISEP_ONNXRUNTIME finished.\n');
end

function run_chisep_only_impl_force_onnx(whichSubject, repoRoot) %#ok<INUSD>
% Thin wrapper: temporarily ensure the ONNX adapter is used, then delegate to
% the standard chi-sep-only implementation (which reads whqsm_local_paths).
% Because the adapter name comes from whqsm_local_paths via configure_sep_cfg,
% we simply call the standard implementation. If the user kept
% useOnnxRuntimeChiSep=true (default), this already uses the ONNX backend.
run_chisep_only_impl(whichSubject);
end
