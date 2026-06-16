# 离线自检（不需要 MATLAB）

这个目录用合成的小模型/数据验证 `infer_chisep_from_mat.py` 桥接逻辑本身是否正确，
方便在没有真实 .onnx 时先确认 Python 环境 OK。

```bash
pip install numpy scipy onnx onnxruntime
python make_dummy.py        # 生成 dummy 的 QSMnet/chi_sepnet/R2PRIMEnet onnx + norm_factor.mat
python make_input.py        # 生成合成输入 (48x52x50 -> 自动裁到 48x48x48)

# r2' 流程
python ../../modules/DL/python/infer_chisep_from_mat.py \
  --input_mat in_r2p.mat --output_mat out_r2p.mat \
  --qsm_onnx models/240904_QSMnet.onnx --xsep_onnx models/chi_sepnet.onnx \
  --norm_factor models/norm_factor.mat --pipeline auto --field_unit Hz --device cpu

# r2* 流程
python ../../modules/DL/python/infer_chisep_from_mat.py \
  --input_mat in_r2s.mat --output_mat out_r2s.mat \
  --qsm_onnx models/240904_QSMnet.onnx --xsep_onnx models/chi_sepnet.onnx \
  --r2prime_onnx models/R2PRIMEnet.onnx \
  --norm_factor models/norm_factor.mat --pipeline auto --field_unit Hz --device cpu
```

注意：这里的 dummy 模型只是结构占位（QSMnet 1->1, R2PRIMEnet 1->1, chi-sepnet 3->2），
用于验证链路；真实结果请用 Chisep_Toolbox 的真实 .onnx。
