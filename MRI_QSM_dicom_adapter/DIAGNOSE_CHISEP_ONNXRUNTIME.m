function DIAGNOSE_CHISEP_ONNXRUNTIME()
% DIAGNOSE_CHISEP_ONNXRUNTIME.m
% ============================================================================
% Check that the ONNX-Runtime chi-separation bridge can run, WITHOUT touching
% MATLAB's broken importNetworkFromONNX / onnxmex.
%
% It verifies:
%   1) Python executable exists and runs
%   2) numpy / scipy / onnxruntime are importable in that Python
%   3) The bridge script, .onnx models and norm_factor.mat exist
%   4) onnxruntime can actually open each .onnx and report I/O specs
%
% Usage (from MRI_QSM_dicom_adapter):
%   DIAGNOSE_CHISEP_ONNXRUNTIME
% ============================================================================

fprintf('\n========== chi-separation ONNX-Runtime diagnosis ==========\n');

P = whqsm_local_paths();

py = '';
if isfield(P,'onnxPythonExecutable'), py = char(P.onnxPythonExecutable); end
if isempty(py) || exist(py,'file') ~= 2
    % try pyenv / where python
    try, pe = pyenv; if exist(char(pe.Executable),'file')==2, py = char(pe.Executable); end, catch, end
end
fprintf('[1] Python executable\n    %s\n', py);
if isempty(py) || exist(py,'file') ~= 2
    fprintf('    ✗ Not found. Set P.onnxPythonExecutable in whqsm_local_paths.m\n');
    return;
end

% 2) Check python packages
fprintf('[2] Python packages (numpy/scipy/onnxruntime)\n');
chk = ['import importlib,sys;' ...
       'mods=["numpy","scipy","onnxruntime"];' ...
       'res=[];' ...
       'res=[(m,(importlib.import_module(m).__version__ if importlib.util.find_spec(m) else None)) for m in mods];' ...
       'print("\n".join("  %s: %s"%(m,v) for m,v in res));' ...
       'sys.exit(0 if all(v for _,v in res) else 3)'];
cmd = sprintf('"%s" -c "%s"', py, chk);
[st, out] = system(cmd);
fprintf('%s\n', out);
if st ~= 0
    fprintf('    ✗ Missing package(s). Install with:\n');
    fprintf('      "%s" -m pip install numpy scipy onnxruntime\n', py);
    fprintf('      (GPU: onnxruntime-gpu)\n');
    return;
end

% 3) Files
fprintf('[3] Required files\n');
bridge = getf(P,'onnxBridgeScript', fullfile(P.modulesDir,'DL','python','infer_chisep_from_mat.py'));
modelsDir = fullfile(P.chiSepRoot, 'models');
qsm  = first_exist(getf(P,'onnxQsmModel',''),  {fullfile(modelsDir,'240904_QSMnet.onnx'), fullfile(modelsDir,'QSMnet.onnx')});
xsep = first_exist(getf(P,'onnxXsepModel',''), {fullfile(modelsDir,'chi_sepnet.onnx'), fullfile(modelsDir,'xsepnet.onnx')});
r2p  = first_exist(getf(P,'onnxR2primeModel',''), {fullfile(modelsDir,'R2PRIMEnet.onnx')});
nf   = first_exist(getf(P,'onnxNormFactor',''), {fullfile(modelsDir,'norm_factor.mat')});
report_file('bridge script', bridge);
report_file('QSMnet onnx',   qsm);
report_file('chi-sepnet onnx', xsep);
report_file('R2PRIMEnet onnx (r2s only)', r2p, true);
report_file('norm_factor.mat', nf);

% 4) Let python open each onnx and print IO
fprintf('[4] onnxruntime can open the models? (I/O specs)\n');
inspect = fileparts(mfilename('fullpath'));
% Write a tiny inline python inspector
tmpPy = fullfile(tempdir, 'inspect_onnx_tmp.py');
fid = fopen(tmpPy,'w');
fprintf(fid, '%s\n', 'import sys, onnxruntime as ort');
fprintf(fid, '%s\n', 'for p in sys.argv[1:]:');
fprintf(fid, '%s\n', '    if not p: continue');
fprintf(fid, '%s\n', '    try:');
fprintf(fid, '%s\n', '        s=ort.InferenceSession(p,providers=["CPUExecutionProvider"])');
fprintf(fid, '%s\n', '        ins=[(i.name,i.shape) for i in s.get_inputs()]');
fprintf(fid, '%s\n', '        outs=[(o.name,o.shape) for o in s.get_outputs()]');
fprintf(fid, '%s\n', '        print("  OK %s"%p); print("     in :",ins); print("     out:",outs)');
fprintf(fid, '%s\n', '    except Exception as e:');
fprintf(fid, '%s\n', '        print("  FAIL %s -> %s"%(p,e))');
fclose(fid);
cmd2 = sprintf('"%s" "%s" "%s" "%s" "%s"', py, tmpPy, qsm, xsep, r2p);
[~, out2] = system(cmd2);
fprintf('%s\n', out2);
if exist(tmpPy,'file')==2, delete(tmpPy); end

fprintf('========== diagnosis done ==========\n');
fprintf('If all green, run: RUN_CHISEP_ONNXRUNTIME\n\n');
end

function report_file(label, p, optional)
if nargin < 3, optional = false; end
if ~isempty(p) && exist(p,'file')==2
    fprintf('    ✓ %-28s %s\n', label, p);
elseif optional
    fprintf('    - %-28s (not set / not needed)\n', label);
else
    fprintf('    ✗ %-28s MISSING\n', label);
end
end

function p = first_exist(preferred, cands)
p = '';
if ~isempty(preferred) && exist(preferred,'file')==2, p = preferred; return; end
for i=1:numel(cands)
    if exist(cands{i},'file')==2, p = cands{i}; return; end
end
end

function v = getf(s,name,default)
if isstruct(s)&&isfield(s,name)&&~isempty(s.(name)), v=s.(name); else, v=default; end
end
