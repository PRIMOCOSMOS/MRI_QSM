% setup.m
% ============================================================================
%   dicom_adapter 一键路径配置脚本
%
%   用法（在 MATLAB 中执行）:
%     >> setup
%
%   自动检测 / 配置:
%     - 原库 MRI_QSM 路径
%     - SEPIA 工具箱路径
%     - MEDI 工具箱路径（可选）
%     - dicom_adapter 自身路径
% ============================================================================

fprintf('\n');
fprintf('╔══════════════════════════════════════════════════════════════╗\n');
fprintf('║  dicom_adapter - 路径配置                                     ║\n');
fprintf('╚══════════════════════════════════════════════════════════════╝\n\n');

% ====== 1. dicom_adapter 自身 ======
this_dir = fileparts(mfilename('fullpath'));
addpath(genpath(this_dir));
fprintf('✅ dicom_adapter: %s\n', this_dir);

% ====== 2. 自动检测原库 ======
fprintf('\n[1/3] 检测原库 MRI_QSM...\n');
candidates = {
    fileparts(this_dir);                                  % 父目录
    'D:\MRI_PRO\MRILAB_X\20170327_qsm2016_recon_challenge\MRI_QSM';
    'D:\MRI_PRO\MRILAB_X\MRI_QSM';
    'C:\MRI_QSM';
    pwd;
};

mri_qsm_root = '';
for k = 1:length(candidates)
    c = candidates{k};
    if isempty(c), continue; end
    if exist(fullfile(c, 'modules', 'mod_dipole_inversion.m'), 'file') == 2
        mri_qsm_root = c;
        break;
    end
end

if isempty(mri_qsm_root)
    fprintf('  ⚠️ 未自动检测到原库，请手动设置:\n');
    fprintf('     mri_qsm_root = uigetdir\n');
    fprintf('     addpath(genpath(fullfile(mri_qsm_root, ''modules'')))\n');
else
    fprintf('  ✅ 原库路径: %s\n', mri_qsm_root);
    addpath(genpath(fullfile(mri_qsm_root, 'modules')));
    addpath(genpath(fullfile(mri_qsm_root, 'Utils_self')));
    addpath(fullfile(mri_qsm_root, 'config'));
    addpath(mri_qsm_root);
end

% ====== 3. 自动检测 SEPIA ======
fprintf('\n[2/3] 检测 SEPIA 工具箱...\n');
sepia_candidates = {
    'D:\MRI_PRO\MRILAB_X\sepia';
    fullfile(mri_qsm_root, 'sepia');
    'C:\sepia';
    '/opt/sepia';
    '/usr/local/sepia';
};

sepia_found = false;
for k = 1:length(sepia_candidates)
    c = sepia_candidates{k};
    if isempty(c), continue; end
    if exist(c, 'dir') == 7
        addpath(genpath(c));
        if exist('QSMMacroIOWrapper', 'file') == 2
            fprintf('  ✅ SEPIA: %s\n', c);
            sepia_found = true;
            break;
        else
            fprintf('  ⚠️ 路径存在但缺 QSMMacroIOWrapper: %s\n', c);
        end
    end
end

if ~sepia_found
    fprintf('  ⚠️ SEPIA 工具箱未找到\n');
    fprintf('     WH-QSM 将不可用，其它方法 (TKD/CFL2/iLSQR/MEDI) 仍可用\n');
end

% ====== 4. 检测 MEDI 工具箱 ======
fprintf('\n[3/3] 检测 MEDI 工具箱...\n');
medi_candidates = {
    'D:\MRI_PRO\MRILAB_X\MEDI_toolbox-2024.11.26';
    fullfile(mri_qsm_root, 'MEDI_toolbox-2024.11.26');
    'C:\MEDI_toolbox';
};

medi_found = false;
for k = 1:length(medi_candidates)
    c = medi_candidates{k};
    if isempty(c), continue; end
    if exist(fullfile(c, 'MEDI_L1.m'), 'file') == 2
        addpath(genpath(c));
        fprintf('  ✅ MEDI: %s\n', c);
        medi_found = true;
        break;
    end
end

if ~medi_found
    fprintf('  ⚠️ MEDI toolbox 未找到（MEDI 方法将 fallback 到内置 ADMM-TV）\n');
end

% ====== 5. 检查 Image Processing Toolbox ======
fprintf('\n[Final] 检查依赖工具箱...\n');
if exist('niftiwrite', 'file') == 2
    fprintf('  ✅ Image Processing Toolbox (niftiwrite 可用)\n');
else
    fprintf('  ❌ Image Processing Toolbox 缺失（SEPIA 调用会失败）\n');
end

if exist('containers.Map', 'file') == 2
    fprintf('  ✅ MATLAB R2008b+ (containers.Map 可用)\n');
end

% ====== 完成 ======
fprintf('\n');
fprintf('╔══════════════════════════════════════════════════════════════╗\n');
fprintf('║  ✅ 配置完成！\n');
fprintf('║\n');
fprintf('║  下一步:\n');
fprintf('║    1) 快速测试:    test_pipeline()\n');
fprintf('║    2) 完整运行:    run_whqsm_comparison()\n');
fprintf('║    3) DICOM 扫描:  SWI202606_dicom_scanner()\n');
fprintf('╚══════════════════════════════════════════════════════════════╝\n\n');
