function run_chisep_onnx_smoketest()
% run_chisep_onnx_smoketest.m
% ============================================================================
% End-to-end smoke test for the chi-separation ONNX-Runtime bridge.
% It does NOT need full WH-QSM outputs. It creates a tiny synthetic volume,
% writes it to a temp .mat, calls infer_chisep_from_mat.py with the REAL .onnx
% models + norm_factor.mat, and checks the output.
%
% Purpose: prove that MATLAB's broken onnxmex is fully bypassed and the
% Python(onnxruntime) path works on your machine before running real data.
%
% Run from project root:
%   run_chisep_onnx_smoketest
% ============================================================================

clc;
fprintf('\n========== chi-separation ONNX bridge smoke test ==========\n');

repoRoot = fileparts(mfilename('fullpath'));
addpath(fullfile(repoRoot, 'modules'), '-begin');
addpath(fullfile(repoRoot, 'MRI_QSM_dicom_adapter'), '-begin');

P = whqsm_local_paths();

% Resolve config
py     = pick(P,'onnxPythonExecutable','');
bridge = pick(P,'onnxBridgeScript', fullfile(repoRoot,'modules','DL','python','infer_chisep_from_mat.py'));
md     = fullfile(P.chiSepRoot,'models');
qsm    = first_exist(pick(P,'onnxQsmModel',''),  {fullfile(md,'240904_QSMnet.onnx'),fullfile(md,'QSMnet.onnx')});
xsep   = first_exist(pick(P,'onnxXsepModel',''), {fullfile(md,'chi_sepnet.onnx'),fullfile(md,'xsepnet.onnx')});
r2p    = first_exist(pick(P,'onnxR2primeModel',''), {fullfile(md,'R2PRIMEnet.onnx')});
nf     = first_exist(pick(P,'onnxNormFactor',''), {fullfile(md,'norm_factor.mat')});
pipeline = pick(P,'onnxPipeline','auto');

need = {py 'Python'; bridge 'bridge script'; qsm 'QSMnet onnx'; xsep 'chi-sepnet onnx'; nf 'norm_factor.mat'};
for i = 1:size(need,1)
    if isempty(need{i,1}) || exist(need{i,1},'file') ~= 2
        error('Missing %s: %s\nFix paths in whqsm_local_paths.m and re-run.', need{i,2}, need{i,1});
    end
end

% Build tiny synthetic input (48^3 so crop_img_16x keeps 48 = 3*16).
N = [48 48 48];
[xx,yy,zz] = ndgrid(linspace(-1,1,N(1)),linspace(-1,1,N(2)),linspace(-1,1,N(3)));
r = sqrt(xx.^2+yy.^2+zz.^2);
mask = double(r < 0.85);
local_field_hz = 5*exp(-(r/0.4).^2) .* mask;     % small Hz field
r2star_hz = (15 + 25*exp(-(r/0.3).^2)) .* mask;  % R2* ~ 15..40 /s
% include an R2 map so the higher-quality r2' pipeline runs if pipeline=auto/r2p
r2_hz = (10 + 5*exp(-(r/0.5).^2)) .* mask;

tmpDir = fullfile(repoRoot, 'output');
if exist(tmpDir,'dir') ~= 7, mkdir(tmpDir); end
in_mat  = fullfile(tmpDir, 'smoketest_chisep_input.mat');
out_mat = fullfile(tmpDir, 'smoketest_chisep_output.mat');
if exist(out_mat,'file')==2, delete(out_mat); end
save(in_mat, 'mask','local_field_hz','r2star_hz','r2_hz','-v7');

cmd = sprintf(['"%s" "%s" --input_mat "%s" --output_mat "%s" ' ...
    '--qsm_onnx "%s" --xsep_onnx "%s" --norm_factor "%s" ' ...
    '--pipeline %s --field_unit Hz --CF 123177385 --Dr 114 --delta_TE 0.0056 --device auto'], ...
    py, bridge, in_mat, out_mat, qsm, xsep, nf, pipeline);
if ~isempty(r2p) && exist(r2p,'file')==2
    cmd = sprintf('%s --r2prime_onnx "%s"', cmd, r2p);
end
fprintf('Command:\n  %s\n\n', cmd);

t = tic;
[st, out] = system(cmd);
fprintf('%s\n', out);
dt = toc(t);

if st ~= 0
    error('Bridge returned exit code %d. See output above.', st);
end
if exist(out_mat,'file') ~= 2
    error('Bridge produced no output file: %s', out_mat);
end

S = load(out_mat);
req = {'x_para','x_dia','x_tot','qsm_map','r2prime_map','mask_out'};
for i=1:numel(req)
    assert(isfield(S,req{i}), 'Output missing field: %s', req{i});
end

fprintf('--- RESULT ---\n');
fprintf('pipeline used   : %s\n', char(S.pipeline));
fprintf('inference time  : %.2f s\n', dt);
fprintf('output size     : %s\n', mat2str(size(S.x_para)));
m = S.mask_out > 0;
fprintf('chi_para  range : [%.5f, %.5f]\n', min(S.x_para(m)), max(S.x_para(m)));
fprintf('chi_dia   range : [%.5f, %.5f]\n', min(S.x_dia(m)),  max(S.x_dia(m)));
assert(all(isfinite(S.x_para(:))) && all(isfinite(S.x_dia(:))), 'Output has NaN/Inf');
assert(all(S.x_para(:) >= -1e-6) && all(S.x_dia(:) >= -1e-6), ...
    'chi_para/chi_dia must be non-negative after zero-truncation');

fprintf('\n✅ Smoke test PASSED. MATLAB onnxmex fully bypassed via onnxruntime.\n');
fprintf('   Next: run real data with RUN_CHISEP_ONLY (uses the same bridge).\n\n');
end

function v = pick(s,name,default)
if isstruct(s)&&isfield(s,name)&&~isempty(s.(name)), v=s.(name); else, v=default; end
end

function p = first_exist(preferred, cands)
p = '';
if ~isempty(preferred)&&exist(preferred,'file')==2, p=preferred; return; end
for i=1:numel(cands), if exist(cands{i},'file')==2, p=cands{i}; return; end, end
end
