function DIAGNOSE_CHISEP_ONNX()
% DIAGNOSE_CHISEP_ONNX.m
% ============================================================================
% Diagnose why SNU Chisep ONNX import fails. This does not run WH-QSM or
% chi-separation reconstruction; it only tests MATLAB ONNX import on the SNU
% model files and writes a report.
% ============================================================================

adapterDir = fileparts(mfilename('fullpath'));
repoRoot = fileparts(adapterDir);
addpath(adapterDir, '-begin');
addpath(fullfile(repoRoot, 'modules'), '-begin');

P = whqsm_local_paths();
if isfield(P, 'chiSepRoot') && exist(P.chiSepRoot, 'dir') == 7
    addpath(P.chiSepRoot, '-begin');
    addpath(genpath(P.chiSepRoot), '-begin');
end
addpath(P.modulesDir, '-begin');

reportDir = fullfile(P.projectRoot, '_chisep_api_report');
if ~exist(reportDir, 'dir'), mkdir(reportDir); end
reportFile = fullfile(reportDir, ['diagnose_chisep_onnx_' datestr(now,'yyyymmdd_HHMMSS') '.txt']);

diary(reportFile);
cleanupObj = onCleanup(@() diary('off')); %#ok<NASGU>

fprintf('\n============================================================\n');
fprintf(' Diagnose SNU Chisep ONNX support\n');
fprintf('============================================================\n');
fprintf('MATLAB      : %s\n', version);
fprintf('projectRoot : %s\n', P.projectRoot);
fprintf('chiSepRoot  : %s\n', P.chiSepRoot);
fprintf('modulesDir  : %s\n', P.modulesDir);
fprintf('Report      : %s\n', reportFile);

fprintf('\n[1] Function resolution before forcing shim\n');
print_which('importONNXNetwork');
print_which('importONNXLayers');
print_which('importNetworkFromONNX');
print_which('assembleNetwork');
print_which('chi_sepnet_general_sinc');
print_which('chi_sepnet_general_new_wResolGen');

fprintf('\n[2] Force project importONNXNetwork shim to front\n');
addpath(P.modulesDir, '-begin');
clear importONNXNetwork;
clear mex;
rehash;
print_which('importONNXNetwork');

fprintf('\n[3] Add ONNX support DLL/MEX folders to PATH\n');
configure_onnx_support_path_local();

fprintf('\n[4] Locate ONNX models under Chisep toolbox\n');
models = dir(fullfile(P.chiSepRoot, '**', '*.onnx'));
if isempty(models)
    fprintf('No .onnx models found under: %s\n', P.chiSepRoot);
else
    for i = 1:numel(models)
        fprintf('[%d] %10d bytes  %s\n', i, models(i).bytes, fullfile(models(i).folder, models(i).name));
    end
end

fprintf('\n[5] Test import of first ONNX model by new MATLAB API\n');
if ~isempty(models)
    modelFile = fullfile(models(1).folder, models(1).name);
    fprintf('Testing model: %s\n', modelFile);
    test_import_new_api(modelFile);
    fprintf('\n[6] Test import through project importONNXNetwork shim\n');
    test_import_shim(modelFile);
else
    fprintf('Skipped: no ONNX model found.\n');
end

fprintf('\n[7] Installed add-ons containing ONNX / Deep Learning\n');
try
    T = matlab.addons.installedAddons;
    names = string(T.Name);
    idx = contains(lower(names), 'onnx') | contains(lower(names), 'deep learning');
    disp(T(idx,:));
catch ME
    fprintf('matlab.addons.installedAddons failed: %s\n', ME.message);
end

fprintf('\n[8] Interpretation\n');
fprintf(['If importNetworkFromONNX/importONNXLayers both fail with onnxmex.mexw64 invalid,\n' ...
         'then the MATLAB ONNX support package or one of its DLL dependencies is broken.\n' ...
         'Fix by repairing/reinstalling ONNX Converter support package and Microsoft VC++ 2015-2022 x64 runtime.\n' ...
         'If direct importNetworkFromONNX succeeds but shim fails, then the project shim needs adjustment.\n']);
fprintf('\nReport saved to: %s\n', reportFile);
fprintf('============================================================\n');
end

%% =========================================================================
function print_which(name)
fprintf('\n--- %s ---\n', name);
fprintf('exist(file)=%d builtin=%d\n', exist(name,'file'), exist(name,'builtin'));
try
    disp(which(name, '-all'));
catch ME
    fprintf('which failed: %s\n', ME.message);
end
try
    fprintf('nargin=%g nargout=%g\n', nargin(name), nargout(name));
catch
end
end

function test_import_new_api(modelFile)
if exist('importNetworkFromONNX', 'file') == 2
    try
        fprintf('Trying importNetworkFromONNX...\n');
        net = importNetworkFromONNX(modelFile); %#ok<NASGU>
        fprintf('SUCCESS: importNetworkFromONNX\n');
        return;
    catch ME
        fprintf('FAILED importNetworkFromONNX:\n%s\n', getReport(ME,'extended','hyperlinks','off'));
    end
else
    fprintf('importNetworkFromONNX not found.\n');
end

if exist('importONNXLayers', 'file') == 2
    try
        fprintf('Trying importONNXLayers...\n');
        lgraph = importONNXLayers(modelFile); %#ok<NASGU>
        fprintf('SUCCESS: importONNXLayers\n');
    catch ME
        fprintf('FAILED importONNXLayers:\n%s\n', getReport(ME,'extended','hyperlinks','off'));
    end
else
    fprintf('importONNXLayers not found.\n');
end
end

function test_import_shim(modelFile)
try
    fprintf('Trying importONNXNetwork shim...\n');
    net = importONNXNetwork(modelFile); %#ok<NASGU>
    fprintf('SUCCESS: importONNXNetwork shim\n');
catch ME
    fprintf('FAILED importONNXNetwork shim:\n%s\n', getReport(ME,'extended','hyperlinks','off'));
end
end

function configure_onnx_support_path_local()
roots = {};
for nm = {'importNetworkFromONNX','importONNXLayers','importONNXNetwork'}
    try
        p = which(nm{1});
        if ~isempty(p), roots{end+1} = fileparts(p); end %#ok<AGROW>
    catch
    end
end
roots = unique(roots);
fprintf('ONNX roots:\n'); disp(roots(:));
allDirs = {};
for r = 1:numel(roots)
    dlls = dir(fullfile(roots{r}, '**', '*.dll'));
    mexs = dir(fullfile(roots{r}, '**', ['*.' mexext]));
    for i = 1:numel(dlls), allDirs{end+1} = dlls(i).folder; end %#ok<AGROW>
    for i = 1:numel(mexs), allDirs{end+1} = mexs(i).folder; end %#ok<AGROW>
end
allDirs = unique(allDirs);
oldPath = getenv('PATH');
parts = regexp(oldPath, pathsep, 'split');
toAdd = {};
for i = 1:numel(allDirs)
    if ~any(strcmpi(parts, allDirs{i}))
        toAdd{end+1} = allDirs{i}; %#ok<AGROW>
    end
end
if ~isempty(toAdd)
    setenv('PATH', [strjoin(toAdd, pathsep) pathsep oldPath]);
end
fprintf('Added %d ONNX DLL/MEX folders to PATH. Total candidate dirs=%d\n', numel(toAdd), numel(allDirs));
end
