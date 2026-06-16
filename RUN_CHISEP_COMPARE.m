function RUN_CHISEP_COMPARE(whichSubject)
% RUN_CHISEP_COMPARE.m
% ============================================================================
% Run χ-separation with BOTH methods (deep-learning χ-sepnet via ONNX Runtime
% AND traditional convex optimization) on existing WH-QSM subjects, and produce
% a side-by-side comparison (maps + stats + voxelwise correlation).
%
% Usage (project root):
%   RUN_CHISEP_COMPARE                 % all subjects
%   RUN_CHISEP_COMPARE('normal')
%   RUN_CHISEP_COMPARE('elderly')
% ============================================================================

if nargin < 1 || isempty(whichSubject), whichSubject = 'all'; end

repoRoot = fileparts(mfilename('fullpath'));
addpath(fullfile(repoRoot,'modules'),'-begin');
addpath(fullfile(repoRoot,'MRI_QSM_dicom_adapter'),'-begin');
addpath(fullfile(repoRoot,'Utils_self'),'-begin');

P = whqsm_local_paths();
if isfield(P,'chiSepRoot') && exist(P.chiSepRoot,'dir')==7
    addpath(P.chiSepRoot,'-begin'); addpath(genpath(P.chiSepRoot),'-begin');
end
addpath(fullfile(repoRoot,'modules'),'-begin');  % keep our shims first

outRoot = fullfile(P.dataRoot, '_qsm_comparison_results');
subDirs = resolve_subject_dirs(outRoot, whichSubject);
if isempty(subDirs), error('未找到被试结果目录（先跑 WH-QSM）：%s', outRoot); end

for s = 1:numel(subDirs)
    subDir = subDirs{s};
    fprintf('\n==================== %s ====================\n', subDir);
    [data, chi, cfg] = load_subject(subDir, P);
    cfg = inject_cfg(cfg, P, subDir);
    cmp = mod_chisep_method_comparison(data, chi, cfg); %#ok<NASGU>
    save(fullfile(subDir, 'chisep_method_comparison_result.mat'), 'cmp', '-v7.3');
end
fprintf('\n✅ RUN_CHISEP_COMPARE done.\n');
end

%% helpers ----------------------------------------------------------------
function [data, chi, cfg] = load_subject(subDir, P)
[~,name]=fileparts(subDir);
if startsWith(lower(name),'elderly_'),label='elderly';
elseif startsWith(lower(name),'normal_'),label='normal';else,label='subject';end
f=fullfile(subDir,['whqsm_' label '_complete.mat']);
if exist(f,'file')~=2
    d=dir(fullfile(subDir,'whqsm_*_complete.mat'));
    if isempty(d), error('未找到 complete.mat: %s',subDir); end
    f=fullfile(d(1).folder,d(1).name);
end
S=load(f); data=S.data; chi=S.chi;
if isfield(S,'cfg'),cfg=S.cfg;else,cfg=struct();end
% ensure R2star present
if ~isfield(data,'R2star_Hz')||isempty(data.R2star_Hz)
    df=fullfile(subDir,'qsm2016_format','data_full.mat');
    if exist(df,'file')==2
        T=load(df);
        if isfield(T,'data')&&isfield(T.data,'R2star_Hz'), data.R2star_Hz=T.data.R2star_Hz; end
    end
end
if ~isfield(data,'R2star_Hz')||isempty(data.R2star_Hz)
    error('缺少 R2star_Hz，无法做磁化率分离: %s', subDir);
end
end

function cfg = inject_cfg(cfg, P, subDir)
cfg.resultDir = fullfile(subDir,'results');
if ~exist(cfg.resultDir,'dir'), mkdir(cfg.resultDir); end
cfg.sep.enable = true;
cfg.sep.chiSepRoot = P.chiSepRoot;
cfg.sep.compare_methods = getf(P,'chisepCompareMethods',{'onnx','opt'});
% onnx bridge
cfg.sep.onnx_python_executable = getf(P,'onnxPythonExecutable','');
cfg.sep.onnx_bridge_script = getf(P,'onnxBridgeScript','');
cfg.sep.onnx_qsm_model = getf(P,'onnxQsmModel','');
cfg.sep.onnx_xsep_model = getf(P,'onnxXsepModel','');
cfg.sep.onnx_r2prime_model = getf(P,'onnxR2primeModel','');
cfg.sep.onnx_norm_factor = getf(P,'onnxNormFactor','');
cfg.sep.onnx_pipeline = getf(P,'onnxPipeline','auto');
cfg.sep.onnx_field_unit = getf(P,'onnxFieldUnit','Hz');
cfg.sep.onnx_device = getf(P,'onnxDevice','auto');
cfg.sep.onnx_resgen = getf(P,'onnxResgen','auto');
cfg.sep.onnx_qsm_source = getf(P,'onnxQsmSource','qsmnet');
cfg.sep.snu_local_field_mode = getf(P,'snuLocalFieldMode','forward_from_whqsm');
cfg.sep.snu_Dr = getf(P,'snuDr',114);
% optimization
cfg.sep.opt_method = getf(P,'optMethod','iLSQR');
cfg.sep.opt_lambda = getf(P,'optLambda',1e-2);
cfg.sep.opt_w_r2 = getf(P,'optWr2',1.0);
cfg.sep.opt_maxiter = getf(P,'optMaxIter',100);
cfg.sep.roi_label_file = getf(P,'roiLabelFile','');
end

function subDirs = resolve_subject_dirs(outRoot, whichSubject)
whichSubject=char(whichSubject);
if exist(whichSubject,'dir')==7, subDirs={whichSubject}; return; end
allD=[dir(fullfile(outRoot,'normal_*'));dir(fullfile(outRoot,'elderly_*'))];
allD=allD([allD.isdir]); subDirs={};
for i=1:numel(allD)
    nm=allD(i).name;
    if strcmpi(whichSubject,'all')||contains(lower(nm),lower(whichSubject))
        subDirs{end+1}=fullfile(allD(i).folder,nm); %#ok<AGROW>
    end
end
end

function v=getf(s,n,d),if isstruct(s)&&isfield(s,n)&&~isempty(s.(n)),v=s.(n);else,v=d;end,end
