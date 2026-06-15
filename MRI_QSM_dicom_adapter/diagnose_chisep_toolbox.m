function diagnose_chisep_toolbox(rootDir)
% diagnose_chisep_toolbox.m
% Diagnose SNU Chisep toolbox function signatures and script call examples.

if nargin < 1 || isempty(rootDir)
    P = whqsm_local_paths();
    rootDir = P.chiSepRoot;
end
rootDir = char(rootDir);
addpath(rootDir, '-begin');
addpath(genpath(rootDir), '-begin');
rehash;

fprintf('\n=== Diagnose Chisep Toolbox ===\n');
fprintf('rootDir: %s\n', rootDir);
fprintf('which chi_sepnet_general -all:\n');
try
    disp(which('chi_sepnet_general', '-all'));
catch ME
    fprintf('which failed: %s\n', ME.message);
end
try
    fprintf('nargin(chi_sepnet_general) = %g\n', nargin('chi_sepnet_general'));
catch ME
    fprintf('nargin failed: %s\n', ME.message);
end
try
    fprintf('help chi_sepnet_general:\n');
    help chi_sepnet_general;
catch ME
    fprintf('help failed: %s\n', ME.message);
end

fprintf('\nScanning .m scripts for chi_sepnet_general calls...\n');
files = dir(fullfile(rootDir, '**', '*.m'));
found = 0;
for i = 1:numel(files)
    f = fullfile(files(i).folder, files(i).name);
    try
        txt = fileread(f);
    catch
        continue;
    end
    if contains(txt, 'chi_sepnet_general')
        found = found + 1;
        fprintf('\n--- %s ---\n', f);
        lines = regexp(txt, '\r?\n', 'split');
        idx = find(contains(lines, 'chi_sepnet_general'));
        for j = 1:numel(idx)
            a = max(1, idx(j)-5); b = min(numel(lines), idx(j)+8);
            for k = a:b
                fprintf('%5d: %s\n', k, lines{k});
            end
        end
    end
end
if found == 0
    fprintf('No .m script references to chi_sepnet_general were found.\n');
end
fprintf('=== End diagnose ===\n\n');
end
