function print_volume_summary(name, vol, Mask)
% print_volume_summary.m
% 打印体积数据在 mask 内的统计摘要
%
% 输入:
%   name - 描述字符串
%   vol  - 3D 体积
%   Mask - logical mask

vol = double(vol);
Mask = logical(Mask);
v = vol(Mask);
v = v(isfinite(v));

if isempty(v)
    fprintf('  %s: mask 内无有效体素', name);
    return;
end

fprintf('  %s (mask内统计):', name);
fprintf('    mean  = %+.6g', mean(v));
fprintf('    std   = %.6g', std(v));
fprintf('    min   = %+.6g', min(v));
fprintf('    p01   = %+.6g', prctile(v, 1));
fprintf('    p50   = %+.6g', prctile(v, 50));
fprintf('    p99   = %+.6g', prctile(v, 99));
fprintf('    max   = %+.6g', max(v));
fprintf('');

end