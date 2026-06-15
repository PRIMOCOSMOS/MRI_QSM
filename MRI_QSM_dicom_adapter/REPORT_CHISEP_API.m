function REPORT_CHISEP_API(rootDir)
% REPORT_CHISEP_API.m
% ============================================================================
% Generate an API/report for SNU Chisep toolbox without opening encrypted P-code.
%
% Usage:
%   cd D:\MRI_PRO\MRILAB_X\20170327_qsm2016_recon_challenge\MRI_QSM_dicom_adapter
%   REPORT_CHISEP_API
% or
%   REPORT_CHISEP_API('D:\...\Chisep_Toolbox_v1.2.1')
% ============================================================================

if nargin < 1 || isempty(rootDir)
    try
        P = whqsm_local_paths();
        rootDir = P.chiSepRoot;
    catch
        rootDir = 'D:\MRI_PRO\MRILAB_X\20170327_qsm2016_recon_challenge\Chisep_Toolbox_v1.2.1';
    end
end
rootDir = char(rootDir);

if exist(rootDir, 'dir') ~= 7
    error('Chisep toolbox root not found: %s', rootDir);
end

reportDir = fullfile(fileparts(rootDir), '_chisep_api_report');
if ~exist(reportDir, 'dir'), mkdir(reportDir); end
reportFile = fullfile(reportDir, ['chisep_api_report_' datestr(now,'yyyymmdd_HHMMSS') '.txt']);

diary(reportFile);
cleanupObj = onCleanup(@() diary('off')); %#ok<NASGU>

fprintf('============================================================\n');
fprintf(' SNU Chisep Toolbox API Report\n');
fprintf('============================================================\n');
fprintf('rootDir : %s\n', rootDir);
fprintf('MATLAB  : %s\n', version);
fprintf('Report  : %s\n', reportFile);

addpath(rootDir, '-begin');
addpath(genpath(rootDir), '-begin');
rehash;

fprintf('\n[1] Function discovery\n');
funcs = {'chi_sepnet_general','chi_sepnet','Chisep_script','Chisep_script_v1', ...
         'chi_separation','chi_separation_general','QSMnet_general','R2PRIMEnet'};
for i = 1:numel(funcs)
    name = funcs{i};
    fprintf('\n--- %s ---\n', name);
    try
        w = which(name, '-all');
        if isempty(w)
            fprintf('which: <not found>\n');
        else
            disp(w);
        end
    catch ME
        fprintf('which failed: %s\n', ME.message);
    end
    try
        fprintf('exist(file) = %d\n', exist(name, 'file'));
    catch
    end
    try
        fprintf('nargin = %g\n', nargin(name));
    catch ME
        fprintf('nargin failed: %s\n', ME.message);
    end
    try
        fprintf('nargout = %g\n', nargout(name));
    catch ME
        fprintf('nargout failed: %s\n', ME.message);
    end
    try
        fprintf('help:\n');
        help(name);
    catch ME
        fprintf('help failed: %s\n', ME.message);
    end
end

fprintf('\n[2] Top-level directory listing\n');
try
    d = dir(rootDir);
    for i = 1:numel(d)
        if ~strcmp(d(i).name,'.') && ~strcmp(d(i).name,'..')
            fprintf('%-8s %10d  %s\n', ternary(d(i).isdir,'<DIR>',''), d(i).bytes, d(i).name);
        end
    end
catch ME
    fprintf('dir failed: %s\n', ME.message);
end

fprintf('\n[3] Model files\n');
patterns = {'*.onnx','*.mat','*.pth','*.pt','*.h5'};
for p = 1:numel(patterns)
    files = dir(fullfile(rootDir, '**', patterns{p}));
    for i = 1:numel(files)
        fprintf('%10d  %s\n', files(i).bytes, fullfile(files(i).folder, files(i).name));
    end
end

fprintf('\n[4] Search text files for chi_sepnet_general calls\n');
textPatterns = {'*.m','*.txt','*.md','*.rst'};
needleList = {'chi_sepnet_general','chi_sepnet(','x_pos','x_neg','have_r2map','R2star','R2*'};
foundAny = false;
for pp = 1:numel(textPatterns)
    files = dir(fullfile(rootDir, '**', textPatterns{pp}));
    for i = 1:numel(files)
        f = fullfile(files(i).folder, files(i).name);
        try
            txt = fileread(f);
        catch
            continue;
        end
        hit = false;
        for nn = 1:numel(needleList)
            if contains(txt, needleList{nn})
                hit = true; break;
            end
        end
        if ~hit, continue; end
        foundAny = true;
        fprintf('\n--- %s ---\n', f);
        lines = regexp(txt, '\r?\n', 'split');
        idx = [];
        for nn = 1:numel(needleList)
            idx = [idx find(contains(lines, needleList{nn}))]; %#ok<AGROW>
        end
        idx = unique(idx);
        idx = idx(1:min(numel(idx), 40));
        for jj = 1:numel(idx)
            a = max(1, idx(jj)-4); b = min(numel(lines), idx(jj)+8);
            fprintf('\nContext around line %d:\n', idx(jj));
            for k = a:b
                fprintf('%5d: %s\n', k, lines{k});
            end
        end
    end
end
if ~foundAny
    fprintf('No readable text files contained target needles. Core API may be P-coded without scripts.\n');
end

fprintf('\n[5] Recommendation\n');
fprintf(['If chi_sepnet_general is P-coded and no readable script shows its 11-arg signature,\n' ...
         'the only reliable options are:\n' ...
         '  (a) use the official top-level Chisep_script supplied by SNU; or\n' ...
         '  (b) request the v1.2.1 function signature / batch API from SNU; or\n' ...
         '  (c) share this report so the adapter can be aligned to the actual script/API.\n']);

fprintf('\nReport saved to: %s\n', reportFile);
fprintf('============================================================\n');
end

function out = ternary(cond, a, b)
if cond, out = a; else, out = b; end
end
