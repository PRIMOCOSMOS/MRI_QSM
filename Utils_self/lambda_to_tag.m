function tag = lambda_to_tag(lambda)
% lambda_to_tag.m
% 将 lambda 数值转换为文件名安全的字符串标签
%
% 示例:
%   0.03  -> '0p03'
%   1e-4  -> '0p0001'

tag = sprintf('%.6g', lambda);
tag = strrep(tag, '.', 'p');
tag = strrep(tag, '-', 'm');
tag = strrep(tag, '+', '');

end