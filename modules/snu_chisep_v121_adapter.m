function result = snu_chisep_v121_adapter(data, chi_total_ppm, R2star_Hz, localField_Hz, Mask, cfg, outDir)
% snu_chisep_v121_adapter.m
% ============================================================================
% Adapter for SNU-LIST Chisep_Toolbox_v1.2.1 based on the official
% Chisep_script.m supplied by the user.
%
% The official script does NOT call chi_sepnet_general directly in v1.2.
% Instead, for RunOptions.Chisep = 'Chi-sepnet':
%
%   if RunOptions.resgen
%       [x_para,x_dia,x_tot,qsm_map,r2p_map] = chi_sepnet_general_new_wResolGen(
%           home_directory, local_field_hz, map, mask_brain_new, Dr,
%           B0dir, CF, VoxelSize, HaveR2Prime, B0_strength,
%           is_scaling, scaling_factor, interp_method, sinc_window_size,
%           sinc_window_type);
%   else
%       [x_para,x_dia,x_tot,qsm_map,r2p_map] = chi_sepnet_general_sinc(
%           home_directory, local_field_hz, map, mask_brain_new, Dr,
%           B0dir, CF, VoxelSize, HaveR2Prime, B0_strength,
%           is_scaling, scaling_factor, interp_method, sinc_window_size,
%           sinc_window_type);
%   end
%
% This adapter follows that official API. For GRE-only data, map = R2* and
% HaveR2Prime = false.
% ============================================================================

result = struct();

chiSepRoot = get_cfg(cfg, {'sep','chiSepRoot'}, '');
if isempty(chiSepRoot) || exist(chiSepRoot, 'dir') ~= 7
    error('SNU chi-separation root does not exist: %s', chiSepRoot);
end
addpath(chiSepRoot, '-begin');
addpath(genpath(chiSepRoot), '-begin');

% Very important: SNU v1.2.x p-code calls the deprecated MATLAB function
% importONNXNetwork. On recent MATLAB releases this may resolve to an old
% support-package wrapper whose onnxmex dependency is broken. Put this project
% module directory back at the very front so our compatibility shim
% modules/importONNXNetwork.m is used first.
projectModuleDir = fileparts(mfilename('fullpath'));
addpath(projectModuleDir, '-begin');
configure_onnx_runtime_path();
clear importONNXNetwork;
clear mex;
rehash;
fprintf('  importONNXNetwork resolves to: %s\n', which('importONNXNetwork'));

Mask = logical(Mask);
matrix_size = size(Mask);
voxel_size = double(data.spatial_res(:).');
B0dir = double(data.B0_dir(:).');
B0dir = B0dir ./ max(norm(B0dir), eps);
B0_strength = double(data.B0);

% SNU script comments define CF as Larmor frequency in Hz:
%   B0_strength = CF / 42.58e6
% Therefore use Hz, not Hz/ppm.
CF = B0_strength * 42.576e6;

% GRE-only: use R2* feature map.
map = double(R2star_Hz);
map(~Mask) = 0;
HaveR2Prime = logical(get_cfg(cfg, {'sep','snu_HaveR2Prime'}, false));

% SNU official Chi-sepnet Dr default.
Dr = double(get_cfg(cfg, {'sep','snu_Dr'}, 114));

% Official script options.
resgen = logical(get_cfg(cfg, {'sep','snu_resgen'}, false));
is_scaling = logical(get_cfg(cfg, {'sep','snu_is_scaling'}, false));
scaling_factor = double(get_cfg(cfg, {'sep','snu_scaling_factor'}, 0.19));
interp_method = char(get_cfg(cfg, {'sep','snu_interp_method'}, 'sinc'));
sinc_window_size = double(get_cfg(cfg, {'sep','snu_sinc_window_size'}, 15));
sinc_window_type = char(get_cfg(cfg, {'sep','snu_sinc_window_type'}, 'hann'));

% local_field_hz expected by SNU is local/tissue field in Hz after BFR. Our
% WH-QSM-only branch has no explicit V-SHARP/PDF/LBV local-field output.
% Default: use WH-QSM forward field as a model-consistent local-field surrogate.
% Alternative: cfg.sep.snu_local_field_mode='measured' passes DICOM phase-fit field.
localFieldMode = lower(char(get_cfg(cfg, {'sep','snu_local_field_mode'}, 'forward_from_whqsm')));
switch localFieldMode
    case {'forward_from_whqsm','forward','qsm_forward'}
        local_field_hz = forward_field_from_chi(chi_total_ppm, data, Mask);
    case {'measured','fieldmap','dicom'}
        local_field_hz = double(localField_Hz);
    otherwise
        error('Unknown cfg.sep.snu_local_field_mode: %s', localFieldMode);
end
local_field_hz(~Mask) = 0;
mask_brain_new = double(Mask);
home_directory = ensure_trailing_filesep(chiSepRoot);

fprintf('\nSNU Chisep v1.2.1 adapter using official Chisep_script API\n');
fprintf('  chiSepRoot     : %s\n', chiSepRoot);
fprintf('  local field    : %s\n', localFieldMode);
fprintf('  feature map    : R2* Hz, HaveR2Prime=%d\n', HaveR2Prime);
fprintf('  CF             : %.6g Hz\n', CF);
fprintf('  B0_strength    : %.6g T\n', B0_strength);
fprintf('  Dr             : %.6g\n', Dr);
fprintf('  resgen         : %d\n', resgen);
fprintf('  is_scaling     : %d, scaling_factor=%.6g\n', is_scaling, scaling_factor);
fprintf('  interp         : %s, sinc window=%g/%s\n', interp_method, sinc_window_size, sinc_window_type);
fprintf('  voxel_size     : [%.6g %.6g %.6g] mm\n', voxel_size(1), voxel_size(2), voxel_size(3));

% Save exact SNU inputs for reproducibility.
snuInputFile = fullfile(outDir, 'snu_chisep_v121_inputs.mat');
B0_direction = B0dir; %#ok<NASGU>
B0dir_save = B0dir; %#ok<NASGU>
local_field_hz_save = local_field_hz; %#ok<NASGU>
map_save = map; %#ok<NASGU>
mask_brain_new_save = mask_brain_new; %#ok<NASGU>
save(snuInputFile, 'home_directory', 'local_field_hz_save', 'map_save', ...
    'mask_brain_new_save', 'Dr', 'B0_direction', 'B0dir_save', 'CF', ...
    'voxel_size', 'matrix_size', 'HaveR2Prime', 'B0_strength', ...
    'is_scaling', 'scaling_factor', 'interp_method', 'sinc_window_size', ...
    'sinc_window_type', 'resgen', 'localFieldMode', '-v7.3');

try
    if resgen
        if function_exists_on_path('chi_sepnet_general_new_wResolGen')
            fprintf('  Calling chi_sepnet_general_new_wResolGen\n');
            [x_para, x_dia, x_tot, qsm_map, r2p_map] = chi_sepnet_general_new_wResolGen( ...
                home_directory, local_field_hz, map, mask_brain_new, Dr, ...
                B0dir, CF, voxel_size, HaveR2Prime, B0_strength, ...
                is_scaling, scaling_factor, interp_method, sinc_window_size, sinc_window_type);
            call_signature = 'chi_sepnet_general_new_wResolGen_official_15args';
        else
            error('resgen=true but chi_sepnet_general_new_wResolGen not found on path.');
        end
    else
        if function_exists_on_path('chi_sepnet_general_sinc')
            fprintf('  Calling chi_sepnet_general_sinc\n');
            [x_para, x_dia, x_tot, qsm_map, r2p_map] = chi_sepnet_general_sinc( ...
                home_directory, local_field_hz, map, mask_brain_new, Dr, ...
                B0dir, CF, voxel_size, HaveR2Prime, B0_strength, ...
                is_scaling, scaling_factor, interp_method, sinc_window_size, sinc_window_type);
            call_signature = 'chi_sepnet_general_sinc_official_15args';
        else
            error('chi_sepnet_general_sinc not found on path.');
        end
    end
catch ME
    error(['SNU Chisep official API failed: %s\n' ...
           'Exact inputs were saved to: %s\n' ...
           'Check ONNX/model dependencies and required helper toolboxes.'], ME.message, snuInputFile);
end

x_para = double(squeeze(x_para));
x_dia  = double(squeeze(x_dia));
x_tot  = double(squeeze(x_tot));
qsm_map = double(squeeze(qsm_map));
r2p_map = double(squeeze(r2p_map));

if ~isequal(size(x_para), matrix_size) || ~isequal(size(x_dia), matrix_size)
    error('SNU chi-separation output size mismatch. x_para=%s x_dia=%s expected=%s', ...
        mat2str(size(x_para)), mat2str(size(x_dia)), mat2str(matrix_size));
end

% Official script treats x_para and x_dia as non-negative source magnitudes and
% x_tot as x_para - x_dia. Pipeline convention stores chi_dia <= 0.
chi_para = max(x_para, 0);
chi_dia  = -max(x_dia, 0);
chi_para(~Mask) = 0;
chi_dia(~Mask) = 0;
x_tot(~Mask) = 0;
qsm_map(~Mask) = 0;
r2p_map(~Mask) = 0;

result.method = 'SNU_Chisep_v1.2.1_official_ChiSepnet_R2star';
result.chi_para = chi_para;
result.chi_dia = chi_dia;
result.x_tot_raw = x_tot;
result.x_para_raw = x_para;
result.x_dia_raw = x_dia;
result.qsm_map = qsm_map;
result.r2p_map = r2p_map;
result.local_field_mode = localFieldMode;
result.CF = CF;
result.Dr = Dr;
result.HaveR2Prime = HaveR2Prime;
result.resgen = resgen;
result.is_scaling = is_scaling;
result.scaling_factor = scaling_factor;
result.interp_method = interp_method;
result.sinc_window_size = sinc_window_size;
result.sinc_window_type = sinc_window_type;
result.input_file = snuInputFile;
result.call_signature = call_signature;
result.toolbox_function = which(call_signature_function_name(call_signature));

save(fullfile(outDir, 'snu_chisep_v121_raw_outputs.mat'), ...
    'x_para', 'x_dia', 'x_tot', 'qsm_map', 'r2p_map', 'chi_para', 'chi_dia', 'result', '-v7.3');
end

%% =========================================================================
function name = call_signature_function_name(signature)
if contains(signature, 'new_wResolGen')
    name = 'chi_sepnet_general_new_wResolGen';
else
    name = 'chi_sepnet_general_sinc';
end
end

function configure_onnx_runtime_path()
% The ONNX support package may keep dependent DLLs in subfolders that are not
% visible to the Windows loader when a MEX is called from encrypted toolbox
% code. Add every DLL-containing ONNX support folder to PATH before SNU imports
% ONNX models.
roots = {};
for nm = {'importONNXNetwork','importONNXLayers','importNetworkFromONNX'}
    try
        p = which(nm{1});
        if ~isempty(p)
            roots{end+1} = fileparts(p); %#ok<AGROW>
        end
    catch
    end
end
roots = unique(roots);
allDllDirs = {};
for r = 1:numel(roots)
    try
        dlls = dir(fullfile(roots{r}, '**', '*.dll'));
        for i = 1:numel(dlls)
            allDllDirs{end+1} = dlls(i).folder; %#ok<AGROW>
        end
        mexs = dir(fullfile(roots{r}, '**', ['*.' mexext]));
        for i = 1:numel(mexs)
            allDllDirs{end+1} = mexs(i).folder; %#ok<AGROW>
        end
    catch
    end
end
allDllDirs = unique(allDllDirs);
if isempty(allDllDirs)
    fprintf('  ONNX support DLL folders: <none found>\n');
    return;
end
oldPath = getenv('PATH');
% Prepend only folders not already present.
parts = regexp(oldPath, pathsep, 'split');
toAdd = {};
for i = 1:numel(allDllDirs)
    if ~any(strcmpi(parts, allDllDirs{i}))
        toAdd{end+1} = allDllDirs{i}; %#ok<AGROW>
    end
end
if ~isempty(toAdd)
    setenv('PATH', [strjoin(toAdd, pathsep) pathsep oldPath]);
end
fprintf('  ONNX support DLL/MEX folders added to PATH: %d\n', numel(toAdd));
end

function tf = function_exists_on_path(name)
code = exist(name, 'file');
tf = any(code == [2 3 6]);
end

function field_hz = forward_field_from_chi(chi_ppm, data, Mask)
N = size(Mask);
voxel_size = double(data.spatial_res(:).');
B0_dir = double(data.B0_dir(:).');
B0_dir = B0_dir ./ max(norm(B0_dir), eps);
if exist('create_dipole_kernel', 'file') == 2
    D = create_dipole_kernel(N, voxel_size, B0_dir);
else
    D = local_dipole_kernel(N, voxel_size, B0_dir);
end
chi_ppm = double(chi_ppm);
chi_ppm(~Mask) = 0;
field_ppm = real(ifftn(D .* fftn(chi_ppm)));
field_ppm(~Mask) = 0;
field_hz = field_ppm * (double(data.B0) * 42.576);
end

function D = local_dipole_kernel(N, voxel_size, B0_dir)
N = double(N(:).'); voxel_size = double(voxel_size(:).'); B0_dir = double(B0_dir(:).');
B0_dir = B0_dir ./ max(norm(B0_dir), eps);
kx_vec = ifftshift((-floor(N(1)/2):ceil(N(1)/2)-1) / (N(1)*voxel_size(1)));
ky_vec = ifftshift((-floor(N(2)/2):ceil(N(2)/2)-1) / (N(2)*voxel_size(2)));
kz_vec = ifftshift((-floor(N(3)/2):ceil(N(3)/2)-1) / (N(3)*voxel_size(3)));
[kx,ky,kz] = ndgrid(kx_vec, ky_vec, kz_vec);
k2 = kx.^2 + ky.^2 + kz.^2;
kdot = kx*B0_dir(1) + ky*B0_dir(2) + kz*B0_dir(3);
D = zeros(N);
idx = k2 > 0;
D(idx) = 1/3 - (kdot(idx).^2 ./ k2(idx));
end

function s = ensure_trailing_filesep(s)
s = char(s);
if isempty(s), return; end
if s(end) ~= filesep && s(end) ~= '/' && s(end) ~= '\'
    s = [s filesep];
end
end

function v = get_cfg(cfg, pathCells, default)
v = default;
try
    s = cfg;
    for i = 1:numel(pathCells)
        if isfield(s, pathCells{i})
            s = s.(pathCells{i});
        else
            return;
        end
    end
    if ~isempty(s), v = s; end
catch
    v = default;
end
end
