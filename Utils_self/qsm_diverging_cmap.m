function cmap = qsm_diverging_cmap(clim, n)
% qsm_diverging_cmap.m
% 零中心发散色图 (蓝-灰-红)
%
% 设计原则:
%   负磁化率 (抗磁性, 如髓鞘/钙化): 蓝色
%   零: 浅灰
%   正磁化率 (顺磁性, 如铁沉积): 红色
%
% 输入:
%   clim - [cmin cmax], 应跨零 (如 [-0.12 0.12])
%   n    - 色图级数 (默认 256)
%
% 输出:
%   cmap - [n x 3] RGB 色图矩阵

if nargin < 2
    n = 256;
end

cmin = clim(1);
cmax = clim(2);

% 非跨零时回退到 parula
if ~(cmin < 0 && cmax > 0)
    cmap = parula(n);
    return;
end

% 零点在色图中的位置
zeroPos = round((0 - cmin) / (cmax - cmin) * (n - 1)) + 1;
zeroPos = max(2, min(n-1, zeroPos));

nNeg = zeroPos;
nPos = n - zeroPos + 1;

% 颜色锚点
blue  = [0.05 0.18 0.68];
gray0 = [0.85 0.85 0.85];
red   = [0.72 0.05 0.05];

% 线性插值
neg = interp1([1 nNeg], [blue; gray0], 1:nNeg);
pos = interp1([1 nPos], [gray0; red], 1:nPos);

cmap = [neg; pos(2:end, :)];
cmap = max(min(cmap, 1), 0);

end