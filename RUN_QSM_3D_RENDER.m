function RUN_QSM_3D_RENDER(whichSubject)
% RUN_QSM_3D_RENDER.m
% ============================================================================
% Generate interactive 3D MATLAB figures (.fig) for WH-QSM total chi and
% chi-separation maps. The saved .fig files can be opened in MATLAB and
% rotated/zoomed/dragged interactively (rotate3d on).
%
% Usage (project root):
%   RUN_QSM_3D_RENDER
%   RUN_QSM_3D_RENDER('normal')
%   RUN_QSM_3D_RENDER('elderly')
%   RUN_QSM_3D_RENDER('all')
% ============================================================================

if nargin < 1 || isempty(whichSubject), whichSubject = 'all'; end
repoRoot = fileparts(mfilename('fullpath'));
addpath(fullfile(repoRoot,'MRI_QSM_dicom_adapter'),'-begin');
addpath(fullfile(repoRoot,'Utils_self'),'-begin');

P = whqsm_local_paths();
outRoot = fullfile(P.dataRoot, '_qsm_comparison_results');
subDirs = resolve_subject_dirs(outRoot, whichSubject);
if isempty(subDirs), error('未找到被试结果目录: %s', outRoot); end

for i = 1:numel(subDirs)
    subDir = subDirs{i};
    fprintf('\n================ 3D RENDER: %s ================\n', subDir);
    S = load_subject(subDir);
    renderDir = fullfile(subDir, 'results', 'render_3d');
    if ~exist(renderDir,'dir'), mkdir(renderDir); end
    try
        make_total_qsm_fig(S, renderDir, P);
        make_chisep_fig(S, renderDir, P);
    catch ME
        warning('3D render failed for %s: %s', subDir, ME.message);
    end
end
fprintf('\n✅ RUN_QSM_3D_RENDER completed.\n');
end

%% -----------------------------------------------------------------------
function subDirs = resolve_subject_dirs(outRoot, whichSubject)
whichSubject = char(whichSubject);
if exist(whichSubject,'dir') == 7, subDirs = {whichSubject}; return; end
D = [dir(fullfile(outRoot,'normal_*')); dir(fullfile(outRoot,'elderly_*'))];
D = D([D.isdir]);
subDirs = {};
for i = 1:numel(D)
    if strcmpi(whichSubject,'all') || contains(lower(D(i).name), lower(whichSubject))
        subDirs{end+1} = fullfile(D(i).folder, D(i).name); %#ok<AGROW>
    end
end
end

function S = load_subject(subDir)
[~,nm] = fileparts(subDir);
if startsWith(lower(nm),'normal_'), label='normal';
elseif startsWith(lower(nm),'elderly_'), label='elderly';
else, label='subject'; end
f = fullfile(subDir, ['whqsm_' label '_complete.mat']);
if exist(f,'file') ~= 2
    d = dir(fullfile(subDir,'whqsm_*_complete.mat'));
    if isempty(d), error('No complete.mat found: %s', subDir); end
    f = fullfile(d(1).folder, d(1).name);
end
L = load(f);
S.name = getf(L.data,'subject_name',nm);
S.mask = logical(L.data.Mask);
S.chi = double(L.chi);
S.spatial_res = double(getf(L.data,'spatial_res',[1 1 1]));
S.hasSep = false;
sepFile = fullfile(subDir,'results','susceptibility_separation','susceptibility_separation_results.mat');
if exist(sepFile,'file') == 2
    T = load(sepFile);
    if isfield(T,'chi_para') && isfield(T,'chi_dia')
        S.chi_para = double(T.chi_para);
        S.chi_dia_abs = abs(double(T.chi_dia));
        S.hasSep = true;
    end
end
end

function make_total_qsm_fig(S, renderDir, P)
fig = figure('Color','w','Name',['3D QSM total - ' S.name], 'Position', [80 80 1200 900]);
ax = axes(fig); hold(ax,'on'); axis(ax,'equal'); axis(ax,'off'); view(ax,3);
rotate3d(fig,'on');

add_brain_surface(ax, S.mask, S.spatial_res, [0.82 0.82 0.82], 0.10, getf(P,'render3DReduceRatio',0.20));
add_iso_surface(ax, S.chi, S.mask, S.spatial_res, getf(P,'render3DQsmPosIso',0.03), [0.85 0.20 0.15], 0.80, getf(P,'render3DReduceRatio',0.20));
add_iso_surface(ax, -S.chi, S.mask, S.spatial_res, abs(getf(P,'render3DQsmNegIso',-0.03)), [0.10 0.30 0.85], 0.60, getf(P,'render3DReduceRatio',0.20));
camlight(ax,'headlight'); camlight(ax,'right'); lighting(ax,'gouraud'); material(ax,'dull');
title(ax, sprintf('3D WH-QSM total χ: %s', S.name), 'Interpreter','none');
annotation(fig,'textbox',[0.02 0.02 0.35 0.08],'String', ...
    sprintf('Gray=brain mask  Red=χ>%.3f ppm  Blue=χ<%.3f ppm', getf(P,'render3DQsmPosIso',0.03), abs(getf(P,'render3DQsmNegIso',-0.03))), ...
    'FitBoxToText','on','BackgroundColor','w');
try, savefig(fig, fullfile(renderDir, 'qsm_total_3d.fig')); catch, end
try, exportgraphics(fig, fullfile(renderDir, 'qsm_total_3d.png'),'Resolution',180); catch, end
end

function make_chisep_fig(S, renderDir, P)
if ~S.hasSep, return; end
fig = figure('Color','w','Name',['3D chi-separation - ' S.name], 'Position', [100 100 1200 900]);
ax = axes(fig); hold(ax,'on'); axis(ax,'equal'); axis(ax,'off'); view(ax,3);
rotate3d(fig,'on');

add_brain_surface(ax, S.mask, S.spatial_res, [0.85 0.85 0.85], 0.08, getf(P,'render3DReduceRatio',0.20));
add_iso_surface(ax, S.chi_para, S.mask, S.spatial_res, getf(P,'render3DChiSepIso',0.03), [0.90 0.45 0.10], 0.85, getf(P,'render3DReduceRatio',0.20));
add_iso_surface(ax, S.chi_dia_abs, S.mask, S.spatial_res, getf(P,'render3DChiSepIso',0.03), [0.10 0.70 0.95], 0.55, getf(P,'render3DReduceRatio',0.20));
camlight(ax,'headlight'); camlight(ax,'left'); lighting(ax,'gouraud'); material(ax,'dull');
title(ax, sprintf('3D χ-separation: %s', S.name), 'Interpreter','none');
annotation(fig,'textbox',[0.02 0.02 0.42 0.08],'String', ...
    sprintf('Gray=brain mask  Orange=χ_{para}>%.3f ppm  Cyan=|χ_{dia}|>%.3f ppm', getf(P,'render3DChiSepIso',0.03), getf(P,'render3DChiSepIso',0.03)), ...
    'FitBoxToText','on','BackgroundColor','w');
try, savefig(fig, fullfile(renderDir, 'chisep_3d.fig')); catch, end
try, exportgraphics(fig, fullfile(renderDir, 'chisep_3d.png'),'Resolution',180); catch, end
end

function add_brain_surface(ax, mask, res, colorRGB, alphaVal, reduceRatio)
vol = smooth3(double(mask), 'box', 3);
[f,v] = safe_isosurface(vol, 0.5);
if isempty(f), return; end
v = vox2mm(v, res);
[f,v] = maybe_reduce(f,v,reduceRatio);
p = patch(ax, 'Faces', f, 'Vertices', v, 'FaceColor', colorRGB, 'FaceAlpha', alphaVal, 'EdgeColor', 'none'); %#ok<NASGU>
end

function add_iso_surface(ax, vol, mask, res, isoVal, colorRGB, alphaVal, reduceRatio)
X = double(vol); X(~mask) = 0;
X = smooth3(X, 'box', 3);
[f,v] = safe_isosurface(X, isoVal);
if isempty(f), return; end
v = vox2mm(v, res);
[f,v] = maybe_reduce(f,v,reduceRatio);
p = patch(ax, 'Faces', f, 'Vertices', v, 'FaceColor', colorRGB, 'FaceAlpha', alphaVal, 'EdgeColor', 'none'); %#ok<NASGU>
end

function [f,v] = safe_isosurface(vol, isoVal)
f = []; v = [];
try
    S = isosurface(vol, isoVal);
    if isstruct(S) && isfield(S,'faces') && ~isempty(S.faces)
        f = S.faces; v = S.vertices;
    end
catch
end
end

function [f2,v2] = maybe_reduce(f,v,ratio)
f2 = f; v2 = v;
if nargin < 3 || isempty(ratio) || ratio <= 0 || ratio >= 1, return; end
try
    [f2,v2] = reducepatch(f,v,ratio);
catch
end
end

function vmm = vox2mm(v,res)
% MATLAB isosurface vertices are in voxel coordinates with x=columns, y=rows, z=slices.
vmm = [v(:,1)*res(2), v(:,2)*res(1), v(:,3)*res(3)];
end

function v = getf(s,n,d)
if isstruct(s) && isfield(s,n) && ~isempty(s.(n)), v = s.(n); else, v = d; end
end
