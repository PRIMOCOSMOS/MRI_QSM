function out = mod_two_subject_registered_compare(subjNormal, subjElderly, outDir, opts)
% mod_two_subject_registered_compare.m
% ============================================================================
% Registration-based comparison of two subjects (e.g. 59 vs 72 yr).
%
% The original compare_subjects.m deliberately avoided subtraction because it
% requires registration. This module ADDS valid registration (using MATLAB
% Image Processing Toolbox: imregtform/imregister/imwarp, already a project
% dependency) and then computes a VALID voxelwise comparison + ROI/histogram
% statistics for:
%   - WH-QSM total susceptibility (chi)
%   - optionally χ_para / χ_dia maps if provided
%
% Registration strategy (robust, dependency-light):
%   moving = elderly, fixed = normal (or swap via opts.fixed='elderly')
%   1) rigid (translation+rotation) on magnitude/T1 (intensity-based)
%   2) optional affine refinement
%   3) apply the SAME transform to all elderly maps (chi, para, dia)
%   4) intersect brain masks; compute difference & stats inside intersection
%
% INPUTS (structs, fields tolerant):
%   subj*.chi           [X Y Z]  WH-QSM total (ppm)      (required)
%   subj*.mask          [X Y Z]  brain mask              (required)
%   subj*.magn / .t1    [X Y Z]  registration reference  (one required)
%   subj*.spatial_res   [3]      voxel size (mm)         (recommended)
%   subj*.chi_para/.chi_dia      optional source maps
%
% opts (optional):
%   opts.fixed        'normal'(default) | 'elderly'
%   opts.transform    'rigid'(default) | 'affine' | 'rigid+affine'
%   opts.roi_label_file  .mat with an integer label volume (in FIXED space)
%
% OUTPUT: struct with registered elderly maps, difference maps, stats table.
% ============================================================================

if nargin < 3 || isempty(outDir), outDir = pwd; end
if ~exist(outDir,'dir'), mkdir(outDir); end
if nargin < 4, opts = struct(); end
fixedWho   = lower(char(getf(opts,'fixed','normal')));
xfWanted   = lower(char(getf(opts,'transform','rigid')));
roiFile    = char(getf(opts,'roi_label_file',''));

assert(license('test','image_toolbox') || exist('imregtform','file')==2, ...
    'Image Processing Toolbox (imregtform/imwarp) is required.');

% Decide fixed/moving
if strcmp(fixedWho,'elderly')
    F = subjElderly; M = subjNormal; fixedName='ELDERLY'; movingName='NORMAL';
else
    F = subjNormal;  M = subjElderly; fixedName='NORMAL';  movingName='ELDERLY';
end

resF = get_res(F); resM = get_res(M);
RF = imref3d(size(F.chi), resF(2), resF(1), resF(3));
RM = imref3d(size(M.chi), resM(2), resM(1), resM(3));

refF = reg_reference(F);
refM = reg_reference(M);

fprintf('\n============================================================\n');
fprintf(' Two-subject REGISTERED comparison\n');
fprintf(' fixed = %s, moving = %s, transform = %s\n', fixedName, movingName, xfWanted);
fprintf('============================================================\n');

% ---- intensity-based registration: moving -> fixed ----
[optimizer, metric] = imregconfig('multimodal');
optimizer.MaximumIterations = 300;

tform = [];
if contains(xfWanted,'rigid')
    fprintf('  estimating rigid...\n');
    tform = imregtform(mat2gray(refM), RM, mat2gray(refF), RF, 'rigid', optimizer, metric);
end
if contains(xfWanted,'affine')
    fprintf('  estimating affine...\n');
    if isempty(tform)
        tform = imregtform(mat2gray(refM), RM, mat2gray(refF), RF, 'affine', optimizer, metric);
    else
        tform = imregtform(mat2gray(refM), RM, mat2gray(refF), RF, 'affine', optimizer, metric, ...
            'InitialTransformation', tform);
    end
end
if isempty(tform)
    error('No transform estimated; check opts.transform.');
end

% ---- apply transform to all moving maps, resample into fixed grid ----
warp = @(vol, interp) imwarp(double(vol), RM, tform, interp, 'OutputView', RF);
M_chi_r  = warp(M.chi, 'linear');
M_mask_r = warp(double(M.mask), 'nearest') > 0.5;
maps_moving = struct('chi', M_chi_r);
if isfield(M,'chi_para') && ~isempty(M.chi_para), maps_moving.chi_para = warp(M.chi_para,'linear'); end
if isfield(M,'chi_dia')  && ~isempty(M.chi_dia),  maps_moving.chi_dia  = warp(M.chi_dia,'linear'); end

% intersection mask
maskF = logical(F.mask);
common = maskF & M_mask_r;
fprintf('  common mask voxels: %d (fixed=%d, movedmoving=%d)\n', nnz(common), nnz(maskF), nnz(M_mask_r));

% ---- comparisons ----
out = struct();
out.fixedName = fixedName; out.movingName = movingName;
out.transform = tform; out.common_mask = common;
out.fixed = struct('chi', double(F.chi));
out.moving_registered = maps_moving;

statRows = {};
[statRows, diffChi] = add_diff(statRows, 'chi_total', double(F.chi), M_chi_r, common, fixedName, movingName);
out.diff_chi = diffChi;

if isfield(F,'chi_para') && isfield(maps_moving,'chi_para')
    [statRows, dP] = add_diff(statRows, 'chi_para', double(F.chi_para), maps_moving.chi_para, common, fixedName, movingName);
    out.diff_chi_para = dP;
end
if isfield(F,'chi_dia') && isfield(maps_moving,'chi_dia')
    [statRows, dD] = add_diff(statRows, 'chi_dia_abs', abs(double(F.chi_dia)), abs(maps_moving.chi_dia), common, fixedName, movingName);
    out.diff_chi_dia_abs = dD;
end
T = cell2table(statRows, 'VariableNames', ...
    {'map','fixed_mean','moving_mean','diff_mean','diff_std','abs_diff_median','t_stat','n'});
out.stats = T;
writetable(T, fullfile(outDir,'registered_compare_stats.csv'));
disp(T);

% ---- ROI table (labels in FIXED space) ----
if ~isempty(roiFile) && exist(roiFile,'file')==2
    out.roi = roi_compare(roiFile, F, maps_moving, common, fixedName, movingName);
    if ~isempty(out.roi)
        writetable(out.roi, fullfile(outDir,'registered_compare_roi.csv'));
    end
end

% ---- figures ----
make_fig(double(F.chi), M_chi_r, diffChi, common, fixedName, movingName, outDir);

save(fullfile(outDir,'registered_compare.mat'), 'out', '-v7.3');
fprintf('\nRegistered comparison saved to: %s\n', outDir);
end

%% ========================================================================
function [rows, d] = add_diff(rows, name, fixedMap, movedMap, common, fN, mN)
fv = fixedMap(common); mv = movedMap(common);
d = zeros(size(fixedMap)); d(common) = movedMap(common) - fixedMap(common);
df = mv - fv; df=df(isfinite(df));
% paired t-stat (descriptive)
tstat = mean(df) / (std(df)/sqrt(max(numel(df),1)) + eps);
rows(end+1,:) = {name, mean(fv), mean(mv), mean(df), std(df), median(abs(df)), tstat, numel(df)};
fprintf('  %-12s : %s mean=%.4f, %s mean=%.4f, diff(%s-%s)=%.4f±%.4f\n', ...
    name, fN, mean(fv), mN, mean(mv), mN, fN, mean(df), std(df));
end

function ref = reg_reference(S)
if isfield(S,'t1') && ~isempty(S.t1) && nnz(S.t1)>0
    ref = double(S.t1);
elseif isfield(S,'magn') && ~isempty(S.magn)
    m = S.magn; if ndims(m)==4, m = m(:,:,:,1); end
    ref = double(m);
else
    ref = double(S.chi);  % last resort
end
ref(~isfinite(ref))=0;
end

function r = get_res(S)
if isfield(S,'spatial_res') && numel(S.spatial_res)>=3
    r = double(S.spatial_res(:).');
else
    r = [1 1 1];
end
end

function T = roi_compare(roiFile, F, movingMaps, common, fN, mN)
T = [];
L = load(roiFile); fn=fieldnames(L); labels=[];
for j=1:numel(fn)
    v=L.(fn{j});
    if isnumeric(v) && isequal(size(v),size(F.chi)) && max(v(:))>1, labels=round(v); break; end
end
if isempty(labels), return; end
ids=unique(labels(labels>0)); rows={};
for k=1:numel(ids)
    m=(labels==ids(k))&common; if nnz(m)<10, continue; end
    fixedMean=mean(F.chi(m)); movMean=mean(movingMaps.chi(m));
    rows(end+1,:)={ids(k),fixedMean,movMean,movMean-fixedMean,nnz(m)}; %#ok<AGROW>
end
if isempty(rows), return; end
T=cell2table(rows,'VariableNames',{'roi_id',[fN '_chi_mean'],[mN '_chi_mean'],'diff_chi','nvox'});
end

function make_fig(fixedChi, movedChi, diffChi, common, fN, mN, outDir)
[~,sz]=max(squeeze(sum(sum(common,1),2)));
cl=[-0.10 0.10]; dcl=[-0.06 0.06];
fig=figure('Color','w','Position',[40 40 1500 480]);
tiledlayout(1,3,'Padding','compact','TileSpacing','compact');
nexttile; showg(fixedChi(:,:,sz),common(:,:,sz),cl,gray(256)); title([fN ' \chi']); colorbar;
nexttile; showg(movedChi(:,:,sz),common(:,:,sz),cl,gray(256)); title([mN ' \chi (registered)']); colorbar;
nexttile; showg(diffChi(:,:,sz),common(:,:,sz),dcl,redblue(256)); title([mN '-' fN ' \Delta\chi']); colorbar;
sgtitle('Registered two-subject QSM comparison');
try, exportgraphics(fig,fullfile(outDir,'registered_compare.png'),'Resolution',200);
catch, saveas(fig,fullfile(outDir,'registered_compare.png')); end
end

function showg(img,mask,cl,cmap)
img=rot90(squeeze(img)); mask=rot90(squeeze(mask));
h=imagesc(img,cl); set(h,'AlphaData',double(mask)); axis image off;
set(gca,'Color','k'); colormap(gca,cmap);
end

function c = redblue(n)
if nargin<1, n=256; end
m=floor(n/2); r=[linspace(0,1,m) ones(1,n-m)]'; b=[ones(1,m) linspace(1,0,n-m)]';
g=[linspace(0,1,m) linspace(1,0,n-m)]'; c=[r g b];
end

function v = getf(s,n,d)
if isstruct(s)&&isfield(s,n)&&~isempty(s.(n)), v=s.(n); else, v=d; end
end
