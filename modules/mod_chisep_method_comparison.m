function cmp = mod_chisep_method_comparison(data, chi_total_ppm, cfg)
% mod_chisep_method_comparison.m
% ============================================================================
% Run χ-separation with MULTIPLE methods on the same subject and compare:
%   - 'onnx'  : deep-learning χ-sepnet via ONNX Runtime (snu_chisep_onnxruntime_adapter)
%   - 'opt'   : traditional convex optimization (snu_chisep_optimization_adapter)
%
% Produces, per method, χ_para / χ_dia maps; then a comparison figure and a
% quantitative table (global stats + voxelwise correlation between methods,
% optional ROI means).
%
% Usage: called by RUN_CHISEP_COMPARE or directly:
%   cmp = mod_chisep_method_comparison(data, chi, cfg)
%
% Required cfg.sep fields are the same as the individual adapters; add:
%   cfg.sep.compare_methods = {'onnx','opt'};   % which methods to run
%   cfg.sep.opt_method      = 'iLSQR' or 'MEDI'
% ============================================================================

cmp = struct();
if ~isfield(cfg,'resultDir') || isempty(cfg.resultDir)
    error('cfg.resultDir required');
end
outDir = fullfile(cfg.resultDir, 'chisep_method_comparison');
if ~exist(outDir,'dir'), mkdir(outDir); end

Mask = logical(data.Mask);
chi_total_ppm = double(chi_total_ppm); chi_total_ppm(~Mask)=0;
R2star_Hz = double(data.R2star_Hz); R2star_Hz(~Mask)=0;
localField_Hz = double(data.fieldmap_Hz); localField_Hz(~Mask)=0;

methods = get_cfg(cfg, {'sep','compare_methods'}, {'onnx','opt'});
if ischar(methods), methods = {methods}; end

res = struct();
fprintf('\n============================================================\n');
fprintf(' chi-separation METHOD COMPARISON\n');
fprintf('============================================================\n');

% Run onnx BEFORE opt so we can share onnx's R2PRIMEnet-estimated R2' with opt
% (fair comparison: both methods then use the SAME R2', not raw R2* for opt).
methods = reorder_onnx_first(methods);
shared_r2prime_ppm = [];   % captured from onnx, fed to opt

for i = 1:numel(methods)
    mth = lower(methods{i});
    fprintf('\n>>> method: %s\n', mth);
    try
        switch mth
            case {'onnx','dl','chisepnet','deep'}
                r = snu_chisep_onnxruntime_adapter(data, chi_total_ppm, R2star_Hz, localField_Hz, Mask, cfg, outDir);
                key = 'onnx';
                % capture R2' (ppm) that the network actually used, for fair opt
                if isfield(r,'r2p_map') && ~isempty(r.r2p_map)
                    shared_r2prime_ppm = r.r2p_map;
                end
            case {'opt','optimization','traditional','medi','ilsqr'}
                cfg_opt = cfg;
                if ~isempty(shared_r2prime_ppm)
                    cfg_opt.sep.opt_r2prime_ppm = shared_r2prime_ppm;
                    fprintf('  [fair] opt 使用 onnx 的 R2PRIMEnet 估计 R2''(共享口径)\n');
                end
                r = snu_chisep_optimization_adapter(data, chi_total_ppm, R2star_Hz, localField_Hz, Mask, cfg_opt, outDir);
                key = 'opt';
            otherwise
                warning('unknown method %s, skipped', mth); continue;
        end
        res.(key) = r;
        % save NIfTI per method
        try
            niftiwrite(single(r.chi_para),       fullfile(outDir, sprintf('%s_chi_para_ppm.nii', key)));
            niftiwrite(single(abs(r.chi_dia)),   fullfile(outDir, sprintf('%s_chi_dia_abs_ppm.nii', key)));
        catch ME
            warning('NIfTI save failed for %s: %s', key, ME.message);
        end
    catch ME
        warning('method %s failed: %s', mth, ME.message);
    end
end

cmp.outDir = outDir;
cmp.results = res;
keys = fieldnames(res);
if numel(keys) < 1
    cmp.status = 'no_method_succeeded'; return;
end

% ---- quantitative comparison ----
T = build_stats_table(res, Mask);
writetable(T, fullfile(outDir, 'method_comparison_stats.csv'));
disp(T);
cmp.stats = T;

% pairwise correlation if >=2 methods
if numel(keys) >= 2
    a = res.(keys{1}); b = res.(keys{2});
    m = Mask;
    cmp.corr_para = corr_safe(a.chi_para(m), b.chi_para(m));
    cmp.corr_dia  = corr_safe(abs(a.chi_dia(m)), abs(b.chi_dia(m)));
    fprintf('\nVoxelwise correlation (%s vs %s): chi_para r=%.3f, |chi_dia| r=%.3f\n', ...
        keys{1}, keys{2}, cmp.corr_para, cmp.corr_dia);
end

% ---- comparison figure ----
make_compare_figure(res, Mask, outDir);

% optional ROI table
roiT = try_roi_table(cfg, res, Mask);
if ~isempty(roiT)
    writetable(roiT, fullfile(outDir,'method_comparison_roi.csv'));
    cmp.roi = roiT;
end

cmp.status = 'ok';
save(fullfile(outDir,'chisep_method_comparison.mat'), 'cmp', '-v7.3');
fprintf('\nComparison saved to: %s\n', outDir);
end

%% ========================================================================
function T = build_stats_table(res, Mask)
keys = fieldnames(res);
rows = {};
for i=1:numel(keys)
    r = res.(keys{i});
    p = r.chi_para(Mask); d = abs(r.chi_dia(Mask));
    rows(end+1,:) = {keys{i}, r.method, mean(p), std(p), prctile(p,99), ...
                     mean(d), std(d), prctile(d,99)}; %#ok<AGROW>
end
T = cell2table(rows, 'VariableNames', {'key','method', ...
    'para_mean','para_std','para_p99','dia_mean','dia_std','dia_p99'});
end

function r = corr_safe(a,b)
a=a(:); b=b(:); ok=isfinite(a)&isfinite(b);
if nnz(ok)<10, r=NaN; return; end
c = corrcoef(a(ok),b(ok)); r=c(1,2);
end

function make_compare_figure(res, Mask, outDir)
keys = fieldnames(res);
[~,sz] = max(squeeze(sum(sum(Mask,1),2)));
nM = numel(keys);
fig = figure('Color','w','Position',[40 40 520*nM 700]);
tiledlayout(2, nM, 'Padding','compact','TileSpacing','compact');
cpara=[0 0.15]; cdia=[0 0.15];   % χpara/|χdia| 多在 0-0.15ppm; 原 0-0.20 偏宽显淡
for r = 1:2
    for i = 1:nM
        nexttile;
        R = res.(keys{i});
        if r==1, img = R.chi_para; ttl=sprintf('%s  \\chi_{para}',keys{i}); cl=cpara;
        else,    img = abs(R.chi_dia); ttl=sprintf('%s  |\\chi_{dia}|',keys{i}); cl=cdia; end
        showimg(img(:,:,sz), Mask(:,:,sz), cl);
        title(ttl,'Interpreter','tex');
    end
end
sgtitle('χ-separation method comparison (para top, |dia| bottom)');
try, exportgraphics(fig, fullfile(outDir,'method_comparison.png'),'Resolution',200);
catch, saveas(fig, fullfile(outDir,'method_comparison.png')); end
end

function showimg(img, mask, cl)
img=rot90(squeeze(img)); mask=rot90(squeeze(mask));
h=imagesc(img,cl); set(h,'AlphaData',double(mask)); axis image off;
set(gca,'Color','k'); colormap(gca,turbo(256)); colorbar;
end

function T = try_roi_table(cfg, res, Mask)
T = [];
roiFile = get_cfg(cfg, {'sep','roi_label_file'}, '');
if isempty(roiFile) || exist(roiFile,'file')~=2, return; end
L = load(roiFile); fn=fieldnames(L); labels=[];
for j=1:numel(fn)
    v=L.(fn{j});
    if isnumeric(v) && isequal(size(v),size(Mask)) && max(v(:))>1, labels=round(v); break; end
end
if isempty(labels), return; end
keys=fieldnames(res); ids=unique(labels(labels>0)); rows={};
for i=1:numel(keys)
    r=res.(keys{i});
    for k=1:numel(ids)
        m=(labels==ids(k))&Mask; if nnz(m)<10, continue; end
        rows(end+1,:)={keys{i},ids(k),mean(r.chi_para(m)),mean(abs(r.chi_dia(m))),nnz(m)}; %#ok<AGROW>
    end
end
if isempty(rows), return; end
T=cell2table(rows,'VariableNames',{'method','roi_id','para_mean','dia_abs_mean','nvox'});
end

function v = get_cfg(cfg, pathCells, default)
v = default;
try
    s = cfg;
    for i = 1:numel(pathCells)
        if isfield(s, pathCells{i}), s = s.(pathCells{i}); else, return; end
    end
    if ~isempty(s), v = s; end
catch
    v = default;
end
end

function out = reorder_onnx_first(methods)
% Ensure onnx runs before opt so opt can reuse onnx's R2PRIMEnet R2' (fair).
isOnnx = @(m) any(strcmpi(m, {'onnx','dl','chisepnet','deep'}));
onnxM = methods(cellfun(isOnnx, methods));
other = methods(~cellfun(isOnnx, methods));
out = [onnxM(:); other(:)]';
end
