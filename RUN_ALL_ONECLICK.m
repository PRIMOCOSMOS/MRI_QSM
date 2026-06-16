function RUN_ALL_ONECLICK(varargin)
% RUN_ALL_ONECLICK.m
% ============================================================================
% 总入口：一步完成完整研究流程。
%
%   阶段 1  WH-QSM         : DICOM → 多回波相位拟合 → SEPIA/FANSI WH-QSM
%   阶段 2  磁化率分离对比 : 深度学习 χ-sepnet (ONNX) vs 传统凸优化
%   阶段 3  两被试配准比较 : 59 vs 72 (NORMAL vs ELDERLY) 配准后差异/统计
%
% 默认从阶段 1 一路跑到阶段 3。也可只跑其中一段（见 'from'/'to' 参数），
% 因此当你手头已有预处理数据 (whqsm_*_complete.mat) 时，可直接从阶段 2 开始，
% 无需重跑 DICOM/WH-QSM。
%
% 用法（项目根目录）：
%   RUN_ALL_ONECLICK                          % 全流程 1→3
%   RUN_ALL_ONECLICK('from','whqsm')          % 同上
%   RUN_ALL_ONECLICK('from','chisep')         % 跳过 WH-QSM，从分离对比开始
%   RUN_ALL_ONECLICK('from','chisep','to','chisep')  % 只做分离对比
%   RUN_ALL_ONECLICK('from','compare')        % 只做两被试配准比较
%   RUN_ALL_ONECLICK('subject','normal')      % 限定某被试(对阶段1/2有效)
%
% 提示：只想"单独跑磁化率分离"且不做对比，请用 RUN_CHISEP_ONLY，
%       或设 whqsm_local_paths.m 里 P.chisepRunMethodCompare=false 后用本入口
%       的 from='chisep',to='chisep'。
% ============================================================================

% ---- parse args ----
p = inputParser;
addParameter(p, 'from', 'whqsm', @(x) ischar(x) || isstring(x));
addParameter(p, 'to',   'compare', @(x) ischar(x) || isstring(x));
addParameter(p, 'subject', 'all', @(x) ischar(x) || isstring(x));
parse(p, varargin{:});
fromStage = lower(char(p.Results.from));
toStage   = lower(char(p.Results.to));
subject   = char(p.Results.subject);

order = {'whqsm','chisep','compare'};
iFrom = stage_index(order, fromStage);
iTo   = stage_index(order, toStage);
if iFrom > iTo
    error('from (%s) 不能晚于 to (%s)', fromStage, toStage);
end

repoRoot = fileparts(mfilename('fullpath'));
adapterDir = fullfile(repoRoot, 'MRI_QSM_dicom_adapter');
addpath(adapterDir, '-begin');
addpath(fullfile(repoRoot, 'modules'), '-begin');
addpath(fullfile(repoRoot, 'Utils_self'), '-begin');

banner();
fprintf(' 阶段范围: %s → %s   被试: %s\n', upper(order{iFrom}), upper(order{iTo}), subject);
fprintf('============================================================\n');

t0 = tic;

% ---- 阶段 1: WH-QSM ----
if in_range(order, 'whqsm', iFrom, iTo)
    fprintf('\n########## 阶段 1/3: WH-QSM (DICOM → WH-QSM) ##########\n');
    feval('RUN_WHQSM_ONECLICK');
else
    fprintf('\n[跳过 阶段 1: WH-QSM]\n');
end

% ---- 阶段 2: 磁化率分离 + 方法对比 ----
if in_range(order, 'chisep', iFrom, iTo)
    fprintf('\n########## 阶段 2/3: 磁化率分离 (深度学习 vs 传统优化) ##########\n');
    safe_run(@() RUN_CHISEP_COMPARE(subject), '磁化率分离对比');
else
    fprintf('\n[跳过 阶段 2: 磁化率分离]\n');
end

% ---- 阶段 3: 两被试配准比较 ----
if in_range(order, 'compare', iFrom, iTo)
    fprintf('\n########## 阶段 3/3: 两被试配准比较 (59 vs 72) ##########\n');
    safe_run(@() RUN_TWO_SUBJECT_COMPARE(), '两被试配准比较');
else
    fprintf('\n[跳过 阶段 3: 两被试配准比较]\n');
end

fprintf('\n============================================================\n');
fprintf('✅ RUN_ALL_ONECLICK 完成，用时 %.1f min\n', toc(t0)/60);
try
    P = whqsm_local_paths();
    fprintf('结果根目录: %s\n', fullfile(P.dataRoot, '_qsm_comparison_results'));
catch
end
fprintf('============================================================\n');
end

%% ========================================================================
function idx = stage_index(order, name)
idx = find(strcmp(order, name), 1);
if isempty(idx)
    error('未知阶段名 "%s"。可选: whqsm | chisep | compare', name);
end
end

function tf = in_range(order, name, iFrom, iTo)
i = find(strcmp(order, name), 1);
tf = (i >= iFrom) && (i <= iTo);
end

function safe_run(fn, label)
try
    fn();
catch ME
    warning('阶段[%s]失败但继续: %s', label, ME.message);
    fprintf('  错误详情:\n');
    disp(getReport(ME, 'extended', 'hyperlinks', 'off'));
end
end

function banner()
fprintf('\n');
fprintf('╔══════════════════════════════════════════════════════════════╗\n');
fprintf('║  MRI_QSM 总入口  RUN_ALL_ONECLICK                          ║\n');
fprintf('║  WH-QSM → 磁化率分离(DL vs 优化) → 两被试配准比较          ║\n');
fprintf('╚══════════════════════════════════════════════════════════════╝\n');
end
