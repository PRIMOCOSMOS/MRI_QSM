### Project Summary
---

```
D:\MRI_PRO\MRILAB_X\20170327_qsm2016_recon_challenge\
├── main_qsm_pipeline.m
├── run_xqsm_bridge_smoketest.m % xQSM Python 桥接测试脚本
├── config/
│   └── pipeline_config.m
├── modules/
│   ├── mod_load_data.m
│   ├── mod_phase_unwrap.m
│   ├── mod_background_removal.m
│   ├── mod_dipole_inversion.m
│   ├── mod_deep_learning.m              % DL 调度入口
│   ├── mod_evaluation.m
│   └── mod_visualization.m
├── modules/dl/
│   ├── dl_prepare_input.m               % DL 公共预处理
│   ├── dl_qsmnet_plus.m                 % QSMnet+
│   ├── dl_xqsm.m                        % xQSM
│   ├── dl_matlab_unet.m                 % MATLAB 原生 3D U-Net
│   └── dl_python_bridge.m              % Python 桥接调用
├── utils/
│   ├── create_dipole_kernel.m
│   ├── qsm_diverging_cmap.m
│   ├── lambda_to_tag.m
│   └── print_volume_summary.m
├── models/                              % 预训练模型权重
│   └── xQSM_invivo.pth
├── output/
│   ├── results/
│   └── figures/
└── data/
```

---

