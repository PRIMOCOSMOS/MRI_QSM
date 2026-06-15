function diagnose_onnx_support()
% diagnose_onnx_support.m
% Diagnose MATLAB ONNX import support for SNU Chisep Toolbox.

fprintf('\n=== ONNX support diagnostics ===\n');
fprintf('MATLAB: %s\n', version);
funcs = {'importONNXNetwork','importONNXLayers','importNetworkFromONNX','assembleNetwork','dlnetwork','predict'};
for i = 1:numel(funcs)
    name = funcs{i};
    fprintf('\n--- %s ---\n', name);
    fprintf('exist(file) = %d, exist(builtin) = %d\n', exist(name,'file'), exist(name,'builtin'));
    try disp(which(name,'-all')); catch ME, fprintf('which failed: %s\n', ME.message); end
end

fprintf('\nInstalled add-ons containing ONNX / Deep Learning:\n');
try
    T = matlab.addons.installedAddons;
    if isempty(T)
        fprintf('<none or unavailable>\n');
    else
        names = string(T.Name);
        idx = contains(lower(names), 'onnx') | contains(lower(names), 'deep learning');
        disp(T(idx,:));
    end
catch ME
    fprintf('matlab.addons.installedAddons failed: %s\n', ME.message);
end
fprintf('=== End ONNX diagnostics ===\n\n');
end
