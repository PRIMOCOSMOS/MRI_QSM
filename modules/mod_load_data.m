function data = mod_load_data(cfg)
% mod_load_data.m
% 加载 QSM2016 官方数据集
%
% QSM2016 Challenge 数据说明:
%   - phs_tissue: 局部组织场 (ppm), 已完成背景场去除
%   - phs_wrap:   缠绕相位 (rad)
%   - phs_unwrap: 解缠相位 (rad)
%   - msk:        脑 mask
%   - magn:       幅值图像
%   - magn_raw:   原始幅值
%   - mp_rage:    T1w 结构像 (用于 MEDI 结构先验)
%   - chi_33:     STI 参考 (χ_33 分量)
%   - chi_cosmos: COSMOS 参考
%   - spatial_res: 体素尺寸 [dx dy dz] mm
%   - evaluation_mask: 评估区域 mask (WM>6, GM 1-6)

dataDir = cfg.dataDir;
fprintf('数据目录: %s\n', dataDir);

% 加载各变量
data.phs_tissue  = double(load_var(dataDir, 'phs_tissue'));
data.phs_wrap    = double(load_var(dataDir, 'phs_wrap'));
data.phs_unwrap  = double(load_var(dataDir, 'phs_unwrap'));
data.spatial_res = double(load_var(dataDir, 'spatial_res'));
data.msk         = logical(load_var(dataDir, 'msk'));
data.magn        = double(load_var(dataDir, 'magn'));
data.magn_raw    = double(load_var(dataDir, 'magn_raw'));
data.mp_rage     = double(load_var(dataDir, 'mp_rage'));
data.chi_33      = double(load_var(dataDir, 'chi_33'));
data.chi_cosmos  = double(load_var(dataDir, 'chi_cosmos'));

% 评估 mask
try
    data.evaluation_mask = double(load_var(dataDir, 'evaluation_mask'));
catch
    warning('evaluation_mask 未找到，使用 msk 替代');
    data.evaluation_mask = double(data.msk);
end

% 规范化
data.spatial_res = data.spatial_res(:).';
data.N = size(data.msk);
data.Mask = data.msk;

% 验证
assert(numel(data.spatial_res) == 3, 'spatial_res 应有3个元素');
assert(isequal(size(data.phs_tissue), data.N), 'phs_tissue 尺寸与 mask 不匹配');

% mask 外置零
data.phs_tissue(~data.Mask) = 0;
data.magn(~data.Mask) = 0;

% 打印信息
fprintf('矩阵尺寸  : [%d %d %d]\n', data.N(1), data.N(2), data.N(3));
fprintf('体素尺寸  : [%.4f %.4f %.4f] mm\n', ...
    data.spatial_res(1), data.spatial_res(2), data.spatial_res(3));
fprintf('Mask 体素数: %d\n', nnz(data.Mask));
fprintf('\n');

print_volume_summary('phs_tissue (ppm局部场)', data.phs_tissue, data.Mask);
print_volume_summary('chi_33 (STI参考)', data.chi_33, data.Mask);
print_volume_summary('mp_rage (T1w结构像)', data.mp_rage, data.Mask);

end

%% =========================================================================
function v = load_var(folder, varname)
% 鲁棒加载 QSM2016 .mat 文件

f1 = fullfile(folder, [varname '.mat']);
f2 = fullfile(folder, varname);

if exist(f1, 'file')
    S = load(f1);
elseif exist(f2, 'file')
    S = load(f2);
else
    error('找不到变量文件: %s 或 %s', f1, f2);
end

if isfield(S, varname)
    v = S.(varname);
else
    names = fieldnames(S);
    if numel(names) == 1
        v = S.(names{1});
    else
        error('文件 "%s" 包含多个变量且无匹配名称', varname);
    end
end

end