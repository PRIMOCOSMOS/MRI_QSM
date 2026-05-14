function chi = dl_python_bridge(model_name, input_norm, Mask, N, ~, norm_factor, cfg)
% dl_python_bridge.m
% MATLAB -> Python 桥接（xQSM .pth）
%
% 本版本要点:
% 1) 不再依赖 PATH 中必须有 python
% 2) 自动优先使用 cfg.deeplearning.python_executable
% 3) 再尝试 pyenv.Executable
% 4) 再尝试 where python
% 5) 若仍失败，抛出可操作错误信息

chi = [];

if ~strcmpi(model_name, 'xqsm')
    error('dl_python_bridge 当前仅支持 model_name = ''xqsm''。');
end

if ~isfield(cfg, 'deeplearning')
    error('cfg.deeplearning 不存在。');
end

% -------------------------------------------------------------------------
% 1) 配置校验
% -------------------------------------------------------------------------
required_fields = {'xqsm_pth', 'xqsm_repo_root', 'xqsm_bridge_script'};
for i = 1:numel(required_fields)
    f = required_fields{i};
    if ~isfield(cfg.deeplearning, f) || isempty(cfg.deeplearning.(f))
        error('cfg.deeplearning.%s 未配置。', f);
    end
end

ckpt_path = cfg.deeplearning.xqsm_pth;
xqsm_repo_root = cfg.deeplearning.xqsm_repo_root;
script_path = cfg.deeplearning.xqsm_bridge_script;

if exist(ckpt_path, 'file') ~= 2
    error('未找到 xQSM checkpoint: %s', ckpt_path);
end
if exist(xqsm_repo_root, 'dir') ~= 7
    error('未找到 xQSM repo 根目录: %s', xqsm_repo_root);
end
if exist(script_path, 'file') ~= 2
    error('未找到 xQSM bridge 脚本: %s', script_path);
end

xqsm_py_file = fullfile(xqsm_repo_root, 'python', 'xQSM.py');
if exist(xqsm_py_file, 'file') ~= 2
    error('xQSM.py 不存在: %s', xqsm_py_file);
end

device_mode = 'auto';
if isfield(cfg.deeplearning, 'xqsm_device') && ~isempty(cfg.deeplearning.xqsm_device)
    device_mode = lower(char(cfg.deeplearning.xqsm_device));
end

% -------------------------------------------------------------------------
% 2) Python 可执行文件解析（修复 9009）
% -------------------------------------------------------------------------
python_exe = resolve_python_executable(cfg);

fprintf('  Python executable:\n    %s\n', python_exe);

% -------------------------------------------------------------------------
% 3) 写入输入
% -------------------------------------------------------------------------
input_mat = fullfile(cfg.resultDir, 'temp_xqsm_input.mat');
output_mat = fullfile(cfg.resultDir, 'temp_xqsm_output.mat');

if exist(output_mat, 'file') == 2
    delete(output_mat);
end

save(input_mat, 'input_norm', 'Mask', 'N', 'norm_factor', '-v7');

% -------------------------------------------------------------------------
% 4) 调用独立脚本
% -------------------------------------------------------------------------
cmd = sprintf('"%s" "%s" --input_mat "%s" --output_mat "%s" --checkpoint "%s" --xqsm_root "%s" --device "%s"', ...
    python_exe, script_path, input_mat, output_mat, ckpt_path, xqsm_repo_root, device_mode);

fprintf('  Python bridge 命令:\n    %s\n', cmd);

[status, result] = system(cmd);
fprintf('%s\n', result);

if status ~= 0
    cleanup_temp_files(input_mat, output_mat);
    if status == 9009
        error(['Python 推理失败，退出码=9009（命令未找到）。' newline ...
               '请在 pipeline_config.m 设置:' newline ...
               'cfg.deeplearning.python_executable = ''C:\path\to\python.exe'';']);
    else
        error('Python 推理失败，退出码=%d。', status);
    end
end

if exist(output_mat, 'file') ~= 2
    cleanup_temp_files(input_mat, output_mat);
    error('Python 推理完成但未生成输出文件: %s', output_mat);
end

S = load(output_mat);
if ~isfield(S, 'chi')
    cleanup_temp_files(input_mat, output_mat);
    error('输出 mat 中未找到变量 chi。');
end

chi = double(S.chi);

if ~isequal(size(chi), N)
    cleanup_temp_files(input_mat, output_mat);
    error('xQSM 输出尺寸不匹配，期望 [%d %d %d]，实际 [%d %d %d]。', ...
        N(1), N(2), N(3), size(chi,1), size(chi,2), size(chi,3));
end

chi(~Mask) = 0;

cleanup_temp_files(input_mat, output_mat);

end

%% =========================================================================
function python_exe = resolve_python_executable(cfg)
% 解析可用 python.exe 路径

python_exe = '';

% A) 配置显式指定（最高优先级）
if isfield(cfg.deeplearning, 'python_executable') && ~isempty(cfg.deeplearning.python_executable)
    candidate = char(cfg.deeplearning.python_executable);
    if exist(candidate, 'file') == 2
        python_exe = candidate;
        return;
    else
        warning('配置的 python_executable 不存在: %s', candidate);
    end
end

% B) MATLAB pyenv
try
    pe = pyenv;
    if ~isempty(pe.Executable) && exist(char(pe.Executable), 'file') == 2
        python_exe = char(pe.Executable);
        return;
    end
catch
end

% C) Windows where python
if ispc
    [st, out] = system('where python');
    if st == 0
        lines = regexp(strtrim(out), '\r?\n', 'split');
        for i = 1:numel(lines)
            p = strtrim(lines{i});
            if ~isempty(p) && exist(p, 'file') == 2
                python_exe = p;
                return;
            end
        end
    end
else
    % Linux/macOS
    [st, out] = system('which python3');
    if st == 0
        p = strtrim(out);
        if exist(p, 'file') == 2
            python_exe = p;
            return;
        end
    end
    [st, out] = system('which python');
    if st == 0
        p = strtrim(out);
        if exist(p, 'file') == 2
            python_exe = p;
            return;
        end
    end
end

error(['未找到可用 Python 可执行文件。' newline ...
       '请在 config/pipeline_config.m 设置:' newline ...
       'cfg.deeplearning.python_executable = ''C:\path\to\python.exe'';']);
end

%% =========================================================================
function cleanup_temp_files(varargin)
for i = 1:numel(varargin)
    f = varargin{i};
    if exist(f, 'file') == 2
        try
            delete(f);
        catch
        end
    end
end
end