function RUN_TWO_SUBJECT_COMPARE()
% RUN_TWO_SUBJECT_COMPARE.m
% ============================================================================
% Registration-based comparison of the two project subjects (e.g. 59 vs 72 yr,
% NORMAL vs ELDERLY). Loads existing WH-QSM results, registers them with the
% Image Processing Toolbox (already a project dependency), and computes a VALID
% voxelwise difference + ROI/stats for WH-QSM total χ and (if available) the
% χ_para / χ_dia maps from the chi-separation step.
%
% Usage (project root):
%   RUN_TWO_SUBJECT_COMPARE
%
% Config (whqsm_local_paths.m):
%   P.twoSubjFixed     'normal'|'elderly'   reference space
%   P.twoSubjTransform 'rigid'|'affine'|'rigid+affine'
%   P.roiLabelFile     optional label volume (.mat) in the FIXED subject space
% ============================================================================

repoRoot = fileparts(mfilename('fullpath'));
addpath(fullfile(repoRoot,'modules'),'-begin');
addpath(fullfile(repoRoot,'MRI_QSM_dicom_adapter'),'-begin');
addpath(fullfile(repoRoot,'Utils_self'),'-begin');

P = whqsm_local_paths();
outRoot = fullfile(P.dataRoot, '_qsm_comparison_results');

normalDir  = first_dir(outRoot, 'normal_*');
elderlyDir = first_dir(outRoot, 'elderly_*');
if isempty(normalDir) || isempty(elderlyDir)
    error('需要 normal_* 和 elderly_* 两个被试结果目录（先跑 WH-QSM）。outRoot=%s', outRoot);
end

subjN = load_subject_for_compare(normalDir,  'normal');
subjE = load_subject_for_compare(elderlyDir, 'elderly');

opts = struct();
opts.fixed     = getf(P,'twoSubjFixed','normal');
opts.transform = getf(P,'twoSubjTransform','rigid');
opts.roi_label_file = getf(P,'roiLabelFile','');

outDir = fullfile(outRoot, '_two_subject_registered_compare');
out = mod_two_subject_registered_compare(subjN, subjE, outDir, opts); %#ok<NASGU>

fprintf('\n✅ RUN_TWO_SUBJECT_COMPARE done. Output:\n  %s\n', outDir);
end

%% helpers ----------------------------------------------------------------
function S = load_subject_for_compare(subDir, label)
f = fullfile(subDir, ['whqsm_' label '_complete.mat']);
if exist(f,'file')~=2
    d = dir(fullfile(subDir,'whqsm_*_complete.mat'));
    if isempty(d), error('未找到 complete.mat: %s', subDir); end
    f = fullfile(d(1).folder,d(1).name);
end
L = load(f); data = L.data;
S = struct();
S.name = getf(data,'subject_name',label);
S.chi  = double(L.chi);
S.mask = logical(data.Mask);
S.spatial_res = getf(data,'spatial_res',[1 1 1]);
% registration reference: prefer T1, else magnitude
if isfield(data,'mp_rage') && ~isempty(data.mp_rage), S.t1 = double(data.mp_rage); end
if isfield(data,'magn'), m=data.magn; if ndims(m)==4, m=m(:,:,:,1); end, S.magn=double(m); end
% optional source-separation maps (from chi-sep step), if present
sepFile = fullfile(subDir,'results','susceptibility_separation','susceptibility_separation_results.mat');
if exist(sepFile,'file')==2
    try
        T = load(sepFile);
        if isfield(T,'chi_para'), S.chi_para=double(T.chi_para); end
        if isfield(T,'chi_dia'),  S.chi_dia =double(T.chi_dia);  end
    catch
    end
end
end

function d = first_dir(root, pat)
x = dir(fullfile(root, pat)); x = x([x.isdir]);
if isempty(x), d=''; else, d=fullfile(x(1).folder,x(1).name); end
end

function v=getf(s,n,dft),if isstruct(s)&&isfield(s,n)&&~isempty(s.(n)),v=s.(n);else,v=dft;end,end
