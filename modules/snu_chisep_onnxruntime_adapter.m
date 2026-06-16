function result = snu_chisep_onnxruntime_adapter(data, chi_total_ppm, R2star_Hz, localField_Hz, Mask, cfg, outDir)
% snu_chisep_onnxruntime_adapter.m
% ============================================================================
% Drop-in replacement for snu_chisep_v121_adapter that BYPASSES MATLAB's broken
% ONNX importer (importNetworkFromONNX / onnxmex.mexw64 invalid).
%
% Instead of calling SNU v1.2.x p-code (which imports ONNX inside MATLAB and
% crashes when onnxmex is broken), this adapter:
%   1) writes the chi-separation inputs to a temp .mat,
%   2) calls a standalone Python script that loads the SAME .onnx models with
%      onnxruntime and reproduces the official chi_sepnet inference,
%   3) reads the results back into MATLAB.
%
% Signature is identical to snu_chisep_v121_adapter so you only need to switch
% cfg.sep.adapter_function = 'snu_chisep_onnxruntime_adapter'.
%
% Required cfg fields (see whqsm_local_paths.m for defaults):
%   cfg.sep.chiSepRoot                 root of Chisep_Toolbox_v1.2.x (for models/)
%   cfg.sep.onnx_python_executable     python.exe with numpy/scipy/onnxruntime
%   cfg.sep.onnx_bridge_script         path to infer_chisep_from_mat.py
%   cfg.sep.onnx_qsm_model             QSMnet .onnx
%   cfg.sep.onnx_xsep_model            chi-sepnet .onnx
%   cfg.sep.onnx_r2prime_model         R2PRIMEnet .onnx (only for r2s pipeline)
%   cfg.sep.onnx_norm_factor           norm_factor.mat (toolbox)
%   cfg.sep.snu_Dr                     Dr (default 114)
%   cfg.sep.snu_local_field_mode       'forward_from_whqsm' | 'measured'
%   cfg.sep.onnx_pipeline              'auto' | 'r2p' | 'r2s'   (default 'auto')
%   cfg.sep.onnx_field_unit            'Hz' | 'radian' | 'ppm'  (default 'Hz')
%   cfg.sep.onnx_device                'auto' | 'cpu' | 'cuda'  (default 'auto')
% ============================================================================

result = struct();
Mask = logical(Mask);
matrix_size = size(Mask);

% -------------------------------------------------------------------------
% 1) Resolve & validate configuration
% -------------------------------------------------------------------------
chiSepRoot = get_cfg(cfg, {'sep','chiSepRoot'}, '');
Dr         = double(get_cfg(cfg, {'sep','snu_Dr'}, 114));
pipeline   = lower(char(get_cfg(cfg, {'sep','onnx_pipeline'}, 'auto')));
fieldUnit  = char(get_cfg(cfg, {'sep','onnx_field_unit'}, 'Hz'));
device     = char(get_cfg(cfg, {'sep','onnx_device'}, 'auto'));
% QSM source for chi-sepnet input:
%   'qsmnet'   -> run QSMnet onnx (official default)
%   'external' -> use the WH-QSM chi_total_ppm passed into this adapter
%                 (unifies QSM source with your main pipeline)
qsmSource  = lower(char(get_cfg(cfg, {'sep','onnx_qsm_source'}, 'qsmnet')));

% Default model/script locations derived from project + toolbox if not set.
thisDir   = fileparts(mfilename('fullpath'));         % .../modules
defScript = fullfile(thisDir, 'DL', 'python', 'infer_chisep_from_mat.py');

bridgeScript = char(get_cfg(cfg, {'sep','onnx_bridge_script'}, defScript));
qsmModel     = char(get_cfg(cfg, {'sep','onnx_qsm_model'}, ''));
xsepModel    = char(get_cfg(cfg, {'sep','onnx_xsep_model'}, ''));
r2pModel     = char(get_cfg(cfg, {'sep','onnx_r2prime_model'}, ''));
normFactor   = char(get_cfg(cfg, {'sep','onnx_norm_factor'}, ''));

% Auto-discover models inside <chiSepRoot>/models if not explicitly set.
if ~isempty(chiSepRoot) && exist(chiSepRoot,'dir') == 7
    modelsDir = fullfile(chiSepRoot, 'models');
    if isempty(qsmModel)
        qsmModel = pick_first_existing({ ...
            fullfile(modelsDir, '240904_QSMnet.onnx'), ...
            fullfile(modelsDir, 'QSMnet.onnx'), ...
            fullfile(modelsDir, 'QSMmodel.onnx')});
    end
    if isempty(xsepModel)
        xsepModel = pick_first_existing({ ...
            fullfile(modelsDir, '240904_xsepnet.onnx'), ...
            fullfile(modelsDir, 'chi_sepnet.onnx'), ...
            fullfile(modelsDir, 'chi-sepnet.onnx'), ...
            fullfile(modelsDir, 'xsepnet.onnx'), ...
            fullfile(modelsDir, 'chi_sepnet_R2p.onnx'), ...
            fullfile(modelsDir, 'chi_sepnet_R2s.onnx')});
    end
    if isempty(r2pModel)
        r2pModel = pick_first_existing({ ...
            fullfile(modelsDir, '240531_R2PRIMEnet.onnx'), ...   % 3T
            fullfile(modelsDir, 'R2PNET_7T.onnx'), ...           % 7T (仅 7T 数据)
            fullfile(modelsDir, 'R2PRIMEnet.onnx'), ...
            fullfile(modelsDir, 'R2primenet.onnx')});
    end
    if isempty(normFactor)
        normFactor = pick_first_existing({ ...
            fullfile(modelsDir, 'norm_factor.mat'), ...
            fullfile(modelsDir, 'xsepnet_norm_factor.mat')});
    end
end

assert_file(bridgeScript, 'onnx_bridge_script (infer_chisep_from_mat.py)');
assert_file(xsepModel,    'onnx_xsep_model (chi-sepnet .onnx)');
assert_file(normFactor,   'onnx_norm_factor (norm_factor.mat)');
% QSMnet only required when generating QSM internally.
if strcmp(qsmSource, 'qsmnet')
    assert_file(qsmModel, 'onnx_qsm_model (QSMnet .onnx)');
end

% -------------------------------------------------------------------------
% 2) Build local field map (same logic as snu_chisep_v121_adapter)
% -------------------------------------------------------------------------
localFieldMode = lower(char(get_cfg(cfg, {'sep','snu_local_field_mode'}, 'forward_from_whqsm')));
switch localFieldMode
    case {'forward_from_whqsm','forward','qsm_forward'}
        local_field_hz = forward_field_from_chi(chi_total_ppm, data, Mask);
        % forward field is in Hz already (see helper)
        usedUnit = 'Hz';
    case {'measured','fieldmap','dicom'}
        local_field_hz = double(localField_Hz);
        usedUnit = fieldUnit;   % whatever the caller declares localField to be
    otherwise
        error('Unknown cfg.sep.snu_local_field_mode: %s', localFieldMode);
end
local_field_hz(~Mask) = 0;

R2star_Hz = double(R2star_Hz);
R2star_Hz(~Mask) = 0;

% Optional R2 map (from SE) enables the higher-quality r2'-pipeline.
r2_hz = [];
r2candidate = get_cfg(cfg, {'sep','onnx_r2_map'}, []);   % can be a variable or path
if ischar(r2candidate) || isstring(r2candidate)
    if exist(char(r2candidate),'file') == 2
        T = load(char(r2candidate));
        fn = fieldnames(T);
        r2_hz = double(T.(fn{1}));
    end
elseif ~isempty(r2candidate)
    r2_hz = double(r2candidate);
elseif isfield(data, 'R2_Hz') && ~isempty(data.R2_Hz)
    r2_hz = double(data.R2_Hz);
end
if ~isempty(r2_hz)
    r2_hz(~Mask) = 0;
end

% Decide pipeline if auto
effPipeline = pipeline;
if strcmp(effPipeline, 'auto')
    if ~isempty(r2_hz), effPipeline = 'r2p'; else, effPipeline = 'r2s'; end
end
if strcmp(effPipeline, 'r2s')
    assert_file(r2pModel, 'onnx_r2prime_model (R2PRIMEnet .onnx) required for r2s pipeline');
end

% -------------------------------------------------------------------------
% 3) Resolve python executable (reuse same strategy as dl_python_bridge)
% -------------------------------------------------------------------------
python_exe = resolve_python_executable(cfg);

% -------------------------------------------------------------------------
% 4) Write input mat, call python, read output
% -------------------------------------------------------------------------
if ~exist(outDir,'dir'), mkdir(outDir); end
input_mat  = fullfile(outDir, 'temp_chisep_onnx_input.mat');
output_mat = fullfile(outDir, 'temp_chisep_onnx_output.mat');
if exist(output_mat,'file')==2, delete(output_mat); end

% Save under the canonical variable names the Python bridge expects.
mask = double(Mask);                        %#ok<NASGU>
local_field_hz = double(local_field_hz);     % already in Hz
r2star_hz = double(R2star_Hz);               %#ok<NASGU>
% When using the WH-QSM result as QSM source, pass chi_total_ppm as external_qsm.
external_qsm = double(chi_total_ppm);        %#ok<NASGU>
saveVars = {'mask', 'local_field_hz', 'r2star_hz'};
if ~isempty(r2_hz)
    r2_hz = double(r2_hz); %#ok<NASGU>
    saveVars{end+1} = 'r2_hz';
end
if strcmp(qsmSource, 'external')
    saveVars{end+1} = 'external_qsm';
end
save(input_mat, saveVars{:}, '-v7');

CF = double(data.B0) * 42.576e6;            % Larmor freq Hz, like v121 adapter

% QSMnet path: pass a valid file even when external (script ignores it then).
qsmArg = qsmModel;
if isempty(qsmArg), qsmArg = xsepModel; end  % harmless placeholder if missing

vsz = double(data.spatial_res(:).');
if numel(vsz) ~= 3 || any(~isfinite(vsz)) || any(vsz <= 0)
    vsz = [1 1 1];
end
resgenMode = char(get_cfg(cfg, {'sep','onnx_resgen'}, 'auto'));  % auto|on|off

cmd = sprintf(['"%s" "%s" --input_mat "%s" --output_mat "%s" ' ...
    '--qsm_onnx "%s" --xsep_onnx "%s" --norm_factor "%s" ' ...
    '--qsm_source %s --pipeline %s --field_unit %s ' ...
    '--CF %.10g --Dr %.10g --delta_TE %.10g --device %s ' ...
    '--voxel_size %.6g,%.6g,%.6g --resgen %s'], ...
    python_exe, bridgeScript, input_mat, output_mat, ...
    qsmArg, xsepModel, normFactor, ...
    qsmSource, effPipeline, usedUnit, CF, Dr, get_delta_te(data), device, ...
    vsz(1), vsz(2), vsz(3), resgenMode);
if strcmp(effPipeline, 'r2s')
    cmd = sprintf('%s --r2prime_onnx "%s"', cmd, r2pModel);
end

fprintf('\nSNU chi-separation via ONNX Runtime (MATLAB onnxmex bypassed)\n');
fprintf('  python      : %s\n', python_exe);
fprintf('  bridge      : %s\n', bridgeScript);
fprintf('  QSMnet      : %s\n', qsmModel);
fprintf('  chi-sepnet  : %s\n', xsepModel);
if strcmp(effPipeline,'r2s'), fprintf('  R2PRIMEnet  : %s\n', r2pModel); end
fprintf('  norm_factor : %s\n', normFactor);
fprintf('  pipeline    : %s  | field unit: %s | CF=%.6g Hz | Dr=%g\n', ...
    effPipeline, usedUnit, CF, Dr);
fprintf('  data.spatial_res (voxel mm) : [%.4g %.4g %.4g]  -> passed to resgen\n', ...
    vsz(1), vsz(2), vsz(3));
fprintf('  matrix size : [%d %d %d]\n', matrix_size(1), matrix_size(2), matrix_size(3));
fprintf('  command     :\n    %s\n', cmd);

[status, out] = system(cmd);
fprintf('%s\n', out);

if status ~= 0
    cleanup(input_mat, output_mat);
    if status == 9009
        error(['Python not found (exit 9009). Set ' ...
               'cfg.sep.onnx_python_executable or cfg.deeplearning.python_executable ' ...
               'to a python.exe that has numpy/scipy/onnxruntime installed.']);
    end
    error('chi-separation ONNX bridge failed (exit=%d).', status);
end
if exist(output_mat,'file') ~= 2
    cleanup(input_mat, output_mat);
    error('ONNX bridge finished but produced no output: %s', output_mat);
end

S = load(output_mat);
req = {'x_para','x_dia','x_tot','qsm_map','r2prime_map','mask_out'};
for i = 1:numel(req)
    if ~isfield(S, req{i})
        cleanup(input_mat, output_mat);
        error('ONNX bridge output missing field: %s', req{i});
    end
end

x_para = double(S.x_para);
x_dia  = double(S.x_dia);
x_tot  = double(S.x_tot);
qsm_map = double(S.qsm_map);
r2p_map = double(S.r2prime_map);
mask_out = logical(S.mask_out);

% -------------------------------------------------------------------------
% 5) Safety net: the Python bridge already restores outputs to the original
%    grid (it center crop/pads to the network's fixed input size and inverts).
%    pad_center_to is therefore normally a no-op; kept to guarantee the size
%    matches matrix_size for any edge case.
%    Pad back symmetrically with zeros so output matches the input grid.
% -------------------------------------------------------------------------
x_para  = pad_center_to(x_para,  matrix_size);
x_dia   = pad_center_to(x_dia,   matrix_size);
x_tot   = pad_center_to(x_tot,   matrix_size);
qsm_map = pad_center_to(qsm_map, matrix_size);
r2p_map = pad_center_to(r2p_map, matrix_size);

% Pipeline convention: chi_para >= 0, chi_dia <= 0 (store dia as negative).
chi_para = max(x_para, 0);
chi_dia  = -max(x_dia, 0);
chi_para(~Mask) = 0;
chi_dia(~Mask)  = 0;
x_tot(~Mask)    = 0;
qsm_map(~Mask)  = 0;
r2p_map(~Mask)  = 0;

result.method        = sprintf('SNU_ChiSepnet_ONNXRuntime_%s', upper(effPipeline));
result.chi_para      = chi_para;
result.chi_dia       = chi_dia;
result.x_tot_raw     = x_tot;
result.x_para_raw    = x_para;
result.x_dia_raw     = x_dia;
result.qsm_map       = qsm_map;
result.r2p_map       = r2p_map;
result.local_field_mode = localFieldMode;
result.CF            = CF;
result.Dr            = Dr;
result.pipeline      = effPipeline;
result.field_unit    = usedUnit;
result.qsm_source    = qsmSource;
result.backend       = 'onnxruntime';
result.bridge_script = bridgeScript;
result.models        = struct('qsm', qsmModel, 'xsep', xsepModel, ...
                              'r2prime', r2pModel, 'norm_factor', normFactor);

save(fullfile(outDir, 'snu_chisep_onnxruntime_raw_outputs.mat'), ...
    'x_para','x_dia','x_tot','qsm_map','r2p_map','chi_para','chi_dia','result','-v7.3');

cleanup(input_mat, output_mat);
end


%% ========================================================================
function dte = get_delta_te(data)
% Estimate echo spacing (s) for radian->ppm conversion. Only used if
% field_unit=='radian'; harmless otherwise.
dte = 0.0056;
try
    if isfield(data,'echo_times_sec') && numel(data.echo_times_sec) >= 2
        te = double(data.echo_times_sec(:));
        dte = median(diff(te));
    elseif isfield(data,'echo_times_ms') && numel(data.echo_times_ms) >= 2
        te = double(data.echo_times_ms(:))/1000;
        dte = median(diff(te));
    end
catch
end
if ~(isfinite(dte) && dte > 0), dte = 0.0056; end
end

function out = pad_center_to(vol, target)
% Symmetric zero-pad (or center-crop) vol to target size.
vol = double(vol);
sz = size(vol);
if numel(sz) < 3, sz(end+1:3) = 1; end
out = zeros(target);
src_idx = cell(1,3); dst_idx = cell(1,3);
for d = 1:3
    s = sz(d); t = target(d);
    if s == t
        src_idx{d} = 1:s; dst_idx{d} = 1:t;
    elseif s < t            % pad
        off = floor((t - s)/2);
        src_idx{d} = 1:s; dst_idx{d} = (1:s) + off;
    else                    % crop
        off = floor((s - t)/2);
        src_idx{d} = (1:t) + off; dst_idx{d} = 1:t;
    end
end
out(dst_idx{1}, dst_idx{2}, dst_idx{3}) = vol(src_idx{1}, src_idx{2}, src_idx{3});
end

function p = pick_first_existing(cands)
p = '';
for i = 1:numel(cands)
    if exist(cands{i}, 'file') == 2
        p = cands{i}; return;
    end
end
end

function assert_file(p, what)
if isempty(p) || exist(p,'file') ~= 2
    error('Required file for chi-separation ONNX bridge not found: %s\n  (%s)', ...
        what, p);
end
end

function field_hz = forward_field_from_chi(chi_ppm, data, Mask)
N = size(Mask);
voxel_size = double(data.spatial_res(:).');
B0_dir = double(data.B0_dir(:).');
B0_dir = B0_dir ./ max(norm(B0_dir), eps);
if exist('create_dipole_kernel','file') == 2
    D = create_dipole_kernel(N, voxel_size, B0_dir);
else
    D = local_dipole_kernel(N, voxel_size, B0_dir);
end
chi_ppm = double(chi_ppm); chi_ppm(~Mask) = 0;
field_ppm = real(ifftn(D .* fftn(chi_ppm)));
field_ppm(~Mask) = 0;
field_hz = field_ppm * (double(data.B0) * 42.576);
end

function D = local_dipole_kernel(N, voxel_size, B0_dir)
N = double(N(:).'); voxel_size = double(voxel_size(:).'); B0_dir = double(B0_dir(:).');
B0_dir = B0_dir ./ max(norm(B0_dir), eps);
kx = ifftshift((-floor(N(1)/2):ceil(N(1)/2)-1) / (N(1)*voxel_size(1)));
ky = ifftshift((-floor(N(2)/2):ceil(N(2)/2)-1) / (N(2)*voxel_size(2)));
kz = ifftshift((-floor(N(3)/2):ceil(N(3)/2)-1) / (N(3)*voxel_size(3)));
[KX,KY,KZ] = ndgrid(kx,ky,kz);
k2 = KX.^2 + KY.^2 + KZ.^2;
kdot = KX*B0_dir(1) + KY*B0_dir(2) + KZ*B0_dir(3);
D = zeros(N); idx = k2 > 0;
D(idx) = 1/3 - (kdot(idx).^2 ./ k2(idx));
end

function python_exe = resolve_python_executable(cfg)
python_exe = '';
% A) sep-specific override
cand = get_cfg(cfg, {'sep','onnx_python_executable'}, '');
if ~isempty(cand) && exist(char(cand),'file')==2, python_exe = char(cand); return; end
% B) reuse deeplearning python
cand = get_cfg(cfg, {'deeplearning','python_executable'}, '');
if ~isempty(cand) && exist(char(cand),'file')==2, python_exe = char(cand); return; end
% C) pyenv
try
    pe = pyenv;
    if ~isempty(pe.Executable) && exist(char(pe.Executable),'file')==2
        python_exe = char(pe.Executable); return;
    end
catch
end
% D) system lookup
if ispc
    [st,outp] = system('where python');
    if st==0
        lines = regexp(strtrim(outp), '\r?\n', 'split');
        for i=1:numel(lines)
            p = strtrim(lines{i});
            if ~isempty(p) && exist(p,'file')==2, python_exe = p; return; end
        end
    end
else
    for c = {'which python3','which python'}
        [st,outp] = system(c{1});
        if st==0
            p = strtrim(outp);
            if exist(p,'file')==2, python_exe = p; return; end
        end
    end
end
error(['No usable Python found. Set cfg.sep.onnx_python_executable to a ' ...
       'python.exe with numpy/scipy/onnxruntime installed.']);
end

function cleanup(varargin)
for i = 1:numel(varargin)
    f = varargin{i};
    if exist(f,'file')==2, try, delete(f); catch, end, end
end
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
