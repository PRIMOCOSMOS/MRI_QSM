function net = importONNXNetwork(modelFile, varargin)
% importONNXNetwork.m
% ============================================================================
% Compatibility shim for legacy toolboxes (e.g. SNU Chisep_Toolbox_v1.2.1)
% that call MATLAB's deprecated importONNXNetwork function.
%
% R2025a warning recommends importNetworkFromONNX. Therefore this shim uses the
% newest importer first, then falls back to importONNXLayers+assembleNetwork.
% It also prepends ONNX support-package DLL/MEX folders to PATH before import,
% which can fix Windows loader errors such as:
%   onnxmex.mexw64 invalid: The specified procedure/module could not be found.
% ============================================================================

modelFile = char(modelFile);
if exist(modelFile, 'file') ~= 2
    error('importONNXNetwork compatibility shim: ONNX model file not found: %s', modelFile);
end

fprintf('  [compat] importONNXNetwork shim called for: %s\n', modelFile);
configure_onnx_support_path();

% -------------------------------------------------------------------------
% 1) New-style path first: importNetworkFromONNX
% -------------------------------------------------------------------------
if exist('importNetworkFromONNX', 'file') == 2
    try
        args = strip_importONNXNetwork_only_args(varargin);
        try
            net = importNetworkFromONNX(modelFile, args{:});
        catch ME1
            fprintf('  [compat] importNetworkFromONNX with converted args failed: %s\n', ME1.message);
            net = importNetworkFromONNX(modelFile);
        end
        fprintf('  [compat] Imported ONNX via importNetworkFromONNX.\n');
        return;
    catch ME_new
        fprintf('  [compat] importNetworkFromONNX path failed: %s\n', ME_new.message);
        lastErr = ME_new;
    end
else
    lastErr = [];
end

% -------------------------------------------------------------------------
% 2) Fallback: importONNXLayers + assembleNetwork
% -------------------------------------------------------------------------
if exist('importONNXLayers', 'file') == 2 && exist('assembleNetwork', 'file') == 2
    try
        args = strip_importONNXNetwork_only_args(varargin);
        try
            lgraph = importONNXLayers(modelFile, args{:});
        catch ME2
            fprintf('  [compat] importONNXLayers with converted args failed: %s\n', ME2.message);
            lgraph = importONNXLayers(modelFile);
        end
        try
            net = assembleNetwork(lgraph);
        catch ME3
            fprintf('  [compat] assembleNetwork failed, returning imported object: %s\n', ME3.message);
            net = lgraph;
        end
        fprintf('  [compat] Imported ONNX via importONNXLayers/assembleNetwork.\n');
        return;
    catch ME_layers
        fprintf('  [compat] importONNXLayers path failed: %s\n', ME_layers.message);
        lastErr = ME_layers;
    end
end

% -------------------------------------------------------------------------
% No working backend
% -------------------------------------------------------------------------
if ~isempty(lastErr)
    error(['importONNXNetwork compatibility shim: no working ONNX importer backend.\n' ...
           'Last importer error:\n%s\n' ...
           'Model: %s\n' ...
           'If the message mentions onnxmex.mexw64 invalid/not found, repair MATLAB ONNX support package or VC++ runtime.'], ...
           lastErr.message, modelFile);
else
    error(['importONNXNetwork compatibility shim: no ONNX importer backend found.\n' ...
           'Please install/check Deep Learning Toolbox ONNX support. Model: %s'], modelFile);
end
end

%% =========================================================================
function configure_onnx_support_path()
roots = {};
for nm = {'importNetworkFromONNX','importONNXLayers'}
    try
        p = which(nm{1});
        if ~isempty(p)
            roots{end+1} = fileparts(p); %#ok<AGROW>
        end
    catch
    end
end
roots = unique(roots);
allDirs = {};
for r = 1:numel(roots)
    try
        dlls = dir(fullfile(roots{r}, '**', '*.dll'));
        mexs = dir(fullfile(roots{r}, '**', ['*.' mexext]));
        for i = 1:numel(dlls), allDirs{end+1} = dlls(i).folder; end %#ok<AGROW>
        for i = 1:numel(mexs), allDirs{end+1} = mexs(i).folder; end %#ok<AGROW>
    catch
    end
end
allDirs = unique(allDirs);
if isempty(allDirs)
    fprintf('  [compat] ONNX DLL/MEX folders found: 0\n');
    return;
end
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
fprintf('  [compat] ONNX DLL/MEX folders prepended to PATH: %d\n', numel(toAdd));
end

%% =========================================================================
function argsOut = strip_importONNXNetwork_only_args(argsIn)
% Map or remove old importONNXNetwork name-value pairs.
argsOut = {};
i = 1;
while i <= numel(argsIn)
    if ~(ischar(argsIn{i}) || isstring(argsIn{i}))
        argsOut{end+1} = argsIn{i}; %#ok<AGROW>
        i = i + 1;
        continue;
    end
    name = char(argsIn{i});
    lname = lower(name);
    if i < numel(argsIn)
        val = argsIn{i+1};
    else
        val = [];
    end
    switch lname
        case {'outputlayertype','importweights','generatecustomlayers','packagename','classes','classnames'}
            % Legacy-only or frequently incompatible with new importer. Drop.
            i = i + 2;
        otherwise
            argsOut{end+1} = argsIn{i}; %#ok<AGROW>
            if i < numel(argsIn)
                argsOut{end+1} = argsIn{i+1}; %#ok<AGROW>
                i = i + 2;
            else
                i = i + 1;
            end
    end
end
end
