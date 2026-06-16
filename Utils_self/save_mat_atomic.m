function save_mat_atomic(targetFile, varargin)
% save_mat_atomic.m
% ============================================================================
% Robust, ATOMIC, VERIFIED .mat save to prevent corrupt/truncated files.
%
% Why: a large `save(...,'-v7.3')` that is interrupted (MATLAB closed, error
% thrown mid-write, disk briefly full, antivirus lock) leaves a half-written
% HDF5 file that later fails to load with "文件可能已损坏 / cannot read MAT file".
% That is exactly what happened to whqsm_elderly_complete.mat.
%
% Strategy:
%   1) save to a temporary file  <target>.tmp_<rand>.mat
%   2) verify it by reading the variable names back (whos -file)
%   3) atomically replace the target via movefile (overwrite)
%   4) clean up temp on any failure
%
% Usage (drop-in for save):
%   save_mat_atomic('out.mat', 'a', 'b', 'c')            % like save('out.mat','a','b','c')
%   save_mat_atomic('out.mat', S)                        % save all fields of struct S
%   save_mat_atomic('out.mat', {'a','b'}, '-v7.3')       % explicit names + flags
%
% Notes:
%   - Variables are pulled from the CALLER workspace by name (like save).
%   - Default format is -v7.3 (needed for >2GB / large arrays).
% ============================================================================

if nargin < 2
    error('save_mat_atomic: need a target file and at least one variable.');
end

% ---- parse args: separate var names, flags, and a possible struct ----
names = {};
flags = {};
structToSave = [];
for i = 1:numel(varargin)
    a = varargin{i};
    if ischar(a) || isstring(a)
        a = char(a);
        if ~isempty(a) && a(1) == '-'
            flags{end+1} = a; %#ok<AGROW>
        else
            names{end+1} = a; %#ok<AGROW>
        end
    elseif iscell(a)
        for j = 1:numel(a), names{end+1} = char(a{j}); end %#ok<AGROW>
    elseif isstruct(a)
        structToSave = a;
    else
        error('save_mat_atomic: unsupported argument of type %s', class(a));
    end
end
if isempty(flags)
    flags = {'-v7.3'};
end

[tdir, tname, ~] = fileparts(targetFile);
if isempty(tdir), tdir = pwd; end
if ~exist(tdir, 'dir'), mkdir(tdir); end
tmpFile = fullfile(tdir, sprintf('%s.tmp_%s.mat', tname, dec2hex(randi(1e9))));

cleaner = onCleanup(@() cleanup_tmp(tmpFile));

% ---- write to temp ----
if ~isempty(structToSave)
    save(tmpFile, '-struct', 'structToSave', flags{:});
else
    if isempty(names)
        error('save_mat_atomic: no variables specified to save.');
    end
    % Build a struct from caller workspace, then save it (avoids evalin per var).
    S = struct();
    for i = 1:numel(names)
        S.(names{i}) = evalin('caller', names{i});
    end
    save(tmpFile, '-struct', 'S', flags{:});
end

% ---- verify temp is readable ----
ok = false;
try
    info = whos('-file', tmpFile); %#ok<NASGU>
    ok = ~isempty(info);
catch ME
    ok = false;
    verifyErr = ME.message;
end
if ~ok
    if exist('verifyErr','var')
        error('save_mat_atomic: verification read failed (%s). Target NOT replaced: %s', ...
            verifyErr, targetFile);
    else
        error('save_mat_atomic: temp file produced no variables. Target NOT replaced: %s', targetFile);
    end
end

% ---- atomically replace target ----
if exist(targetFile, 'file') == 2
    try, delete(targetFile); catch, end
end
[mok, msg] = movefile(tmpFile, targetFile, 'f');
if ~mok
    error('save_mat_atomic: could not move temp to target (%s). Temp kept: %s', msg, tmpFile);
end

% movefile succeeded; disable cleanup of (now-renamed) temp
clear cleaner;
end

function cleanup_tmp(tmpFile)
if exist(tmpFile, 'file') == 2
    try, delete(tmpFile); catch, end
end
end
