"""
infer_chisep_from_mat.py
========================================================================
MATLAB -> Python(onnxruntime) bridge for SNU-LIST chi-separation (chi-sepnet).

WHY THIS FILE EXISTS
--------------------
MATLAB's importNetworkFromONNX / importONNXNetwork depends on a compiled MEX
(onnxmex.mexw64). On some Windows/MATLAB installs that MEX fails to load with:
    "onnxmex.mexw64 invalid: The specified procedure could not be found."
This is a broken VC++/support-package dependency, NOT a model problem. SNU-LIST
issue #5 reports the same failure.

This script bypasses MATLAB's ONNX importer completely. It loads the .onnx
models directly with onnxruntime and reproduces the official chi_sepnet
inference logic (Code/test.py, Code/custom_dataset.py, Code/test_params.py from
https://github.com/SNU-LIST/chi_sepnet), so results match the official toolbox.

PIPELINE (mirrors official)
---------------------------
  QSMnet:     in field(1ch)            -> QSM(1ch)
  R2PRIMEnet: in R2*(1ch)              -> R2'(1ch)     [only r2s pipeline]
  chi-sepnet: in concat[QSM,field,R2'] -> [chi_para, chi_dia]  (2ch)

QSM SOURCE (configurable)
-------------------------
  --qsm_source qsmnet   : use QSMnet (default, official behaviour)
  --qsm_source external : use an externally supplied QSM map (e.g. your WH-QSM
                          chi_total in ppm). The external QSM is normalized with
                          cosmos_sus_mean/std (same as the official
                          resolution-generalization branch that feeds a
                          pre-computed QSM into chi-sepnet) so it lands in the
                          exact input space chi-sepnet expects.
                          The external QSM is read from the input .mat key
                          'external_qsm' (preferred) or 'chi_total_ppm'.

  normalization : x_n = (x - mean)/std * mask
  concat order  : [QSM, local_field, R2']   (channel dim, fixed!)
  de-norm       : chi_para = pred0*x_pos_std + x_pos_mean ; clip>=0
                  chi_dia  = pred1*x_neg_std + x_neg_mean ; clip>=0
  spatial dims must be multiples of 16 (center-crop, crop_img_16x)

CONTRACT WITH MATLAB ADAPTER
----------------------------
Input  .mat (v7, written by MATLAB) must contain:
    local_field_hz   [X,Y,Z]   local/tissue field in Hz   (or ppm/radian, see --field_unit)
    r2star_hz        [X,Y,Z]   R2* map (1/s, i.e. "Hz")
    mask             [X,Y,Z]   brain mask (0/1)
  optional:
    r2_hz            [X,Y,Z]   R2 map (1/s) -> enables r2' pipeline (R2'=R2*-R2)
Physics params (CF, Dr, delta_TE) and unit are passed as CLI args.

Normalization factors are read from --norm_factor (the toolbox norm_factor.mat).
Model paths are passed as CLI args.

Output .mat (written here, read by MATLAB) contains:
    x_para, x_dia, x_tot, qsm_map, r2prime_map, mask_out   (all cropped to 16x)

USAGE
-----
  python infer_chisep_from_mat.py \
     --input_mat in.mat --output_mat out.mat \
     --qsm_onnx QSMnet.onnx --xsep_onnx chi_sepnet.onnx \
     [--r2prime_onnx R2PRIMEnet.onnx] \
     --norm_factor norm_factor.mat \
     --pipeline auto --field_unit Hz --CF 123177385 --Dr 114 --delta_TE 0.0056 \
     --device auto

Dependencies: numpy, scipy, onnxruntime (or onnxruntime-gpu)
"""
import os
import sys
import math
import argparse
import numpy as np
import scipy.io as sio


# ----------------------------------------------------------------------------
def log(msg):
    print(msg, flush=True)


def crop_img_16x(img):
    """Center-crop the first 3 spatial dims to multiples of 16.
    Mirrors utils.crop_img_16x of official chi_sepnet (only used for
    fully-dynamic networks)."""
    img = np.asarray(img)
    for ax in range(3):
        r = img.shape[ax] % 16
        if r != 0:
            lo = r // 2
            hi = img.shape[ax] - (r - lo)
            sl = [slice(None)] * img.ndim
            sl[ax] = slice(lo, hi)
            img = img[tuple(sl)]
    return img


def fit_to_size(img, target):
    """Center crop-or-pad a 3D volume to exactly `target` (tuple of 3).

    This matches the official χ-sepnet practice: the SNU onnx models are
    exported at a fixed matrix size, and inputs are fitted to it by
    cropping (if larger) / zero-padding (if smaller). Cf. Kim et al.,
    χ-sepnet, Human Brain Mapping 2025 (resolution-generalization section):
    "...the k-space of the input was cropped (for lower resolution) or
    zero-padded (for higher resolution)." For same-resolution data that only
    differs in matrix size (your case: 1 mm iso, 336x384x96 vs net 192x192x128),
    image-domain center crop/pad is the correct, artifact-free equivalent
    (no k-space ringing, so the Tukey window used for resolution changes is
    not needed here).

    Returns (out, info) where info lets you invert it with restore_from_size.
    For each axis: if smaller -> zero-pad centered; if larger -> center crop."""
    img = np.asarray(img, dtype=np.float32)
    out = np.zeros(target, dtype=np.float32)
    info = []  # per axis: dict with mode + offsets
    src_sl = []
    dst_sl = []
    for ax in range(3):
        s = img.shape[ax]
        t = target[ax]
        if s == t:
            src_sl.append(slice(0, s)); dst_sl.append(slice(0, t))
            info.append(('same', 0, 0, s))
        elif s < t:                      # pad
            off = (t - s) // 2
            src_sl.append(slice(0, s)); dst_sl.append(slice(off, off + s))
            info.append(('pad', off, 0, s))
        else:                            # crop
            off = (s - t) // 2
            src_sl.append(slice(off, off + t)); dst_sl.append(slice(0, t))
            info.append(('crop', 0, off, s))
    out[tuple(dst_sl)] = img[tuple(src_sl)]
    return out, info


def restore_from_size(img, info):
    """Invert fit_to_size: map a network-sized volume back to the original grid."""
    img = np.asarray(img, dtype=np.float32)
    orig = tuple(d[3] for d in info)     # original sizes stored per axis
    out = np.zeros(orig, dtype=np.float32)
    src_sl = []
    dst_sl = []
    for ax in range(3):
        mode, padoff, cropoff, s = info[ax]
        t = img.shape[ax]
        if mode == 'same':
            src_sl.append(slice(0, s)); dst_sl.append(slice(0, s))
        elif mode == 'pad':
            # network volume has the original data at [padoff:padoff+s]
            src_sl.append(slice(padoff, padoff + s)); dst_sl.append(slice(0, s))
        else:  # crop -> original was larger; only center region was inferred
            src_sl.append(slice(0, t)); dst_sl.append(slice(cropoff, cropoff + t))
    out[tuple(dst_sl)] = img[tuple(src_sl)]
    return out


def mask_bbox(mask, margin=8):
    """Return a tuple of slices for the bounding box of mask>0 (+margin),
    or None if mask is empty / already full."""
    m = mask > 0
    if not m.any():
        return None
    sl = []
    for ax in range(3):
        axes = tuple(i for i in range(3) if i != ax)
        prof = m.any(axis=axes)
        idx = np.where(prof)[0]
        lo = max(0, idx[0] - margin)
        hi = min(m.shape[ax], idx[-1] + 1 + margin)
        sl.append(slice(lo, hi))
    # if bbox == full volume, skip
    if all(sl[a].start == 0 and sl[a].stop == mask.shape[a] for a in range(3)):
        return None
    return tuple(sl)


def bbox_shape_after(bbox, full_shape):
    if bbox is None:
        return full_shape
    return tuple(bbox[a].stop - bbox[a].start for a in range(3))


def place_in_full(vol, bbox, full_shape):
    """Place a cropped volume back into a zero full-size volume at bbox."""
    if bbox is None:
        return vol
    out = np.zeros(full_shape, dtype=np.float32)
    out[bbox] = vol
    return out


def tukey3d(shape, alpha=0.2):
    """Separable 3D Tukey window (used to suppress k-space crop/pad ringing)."""
    def w1(n):
        if n <= 1:
            return np.ones(n)
        x = np.linspace(0, 1, n)
        w = np.ones(n)
        a = alpha
        if a <= 0:
            return w
        lo = x < a / 2
        hi = x >= 1 - a / 2
        w[lo] = 0.5 * (1 + np.cos(2 * np.pi / a * (x[lo] - a / 2)))
        w[hi] = 0.5 * (1 + np.cos(2 * np.pi / a * (x[hi] - 1 + a / 2)))
        return w
    wx, wy, wz = w1(shape[0]), w1(shape[1]), w1(shape[2])
    return (wx[:, None, None] * wy[None, :, None] * wz[None, None, :]).astype(np.float32)


def kspace_resample(vol, target_shape, tukey_alpha=0.2):
    """Resample a volume to target_shape via k-space crop (smaller) / zero-pad
    (larger), with a Tukey window to minimize ringing. This is the official
    resolution-generalization operation (Ji et al., ISMRM 2023) used by
    chi_sepnet_general_new_wResolGen: changing matrix size in k-space changes
    the effective resolution while keeping the same FOV/anatomy extent.

    Returns the resampled real-valued volume at target_shape."""
    vol = np.asarray(vol, dtype=np.float32)
    src = vol.shape
    if tuple(target_shape) == tuple(src):
        return vol.copy()
    # forward FFT (centered)
    K = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(vol)))
    out = np.zeros(target_shape, dtype=complex)
    sl_src, sl_dst = [], []
    for ax in range(3):
        s, t = src[ax], target_shape[ax]
        n = min(s, t)
        cs = s // 2 - n // 2
        ct = t // 2 - n // 2
        sl_src.append(slice(cs, cs + n))
        sl_dst.append(slice(ct, ct + n))
    # apply Tukey on the copied central k-space block to reduce ringing
    block = K[tuple(sl_src)]
    block = block * tukey3d(block.shape, tukey_alpha)
    out[tuple(sl_dst)] = block
    # scale so intensity is preserved under matrix-size change
    scale = np.prod(target_shape) / np.prod(src)
    rec = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(out))) * scale
    return np.real(rec).astype(np.float32)


def field_to_ppm(field, unit, CF, delta_TE):
    unit = unit.lower()
    if unit == "ppm":
        return field
    if unit == "hz":
        return field / CF * 1e6
    if unit == "radian":
        return field / (2.0 * math.pi * delta_TE) / CF * 1e6
    raise ValueError("field_unit must be Hz / radian / ppm, got %r" % unit)


# ----------------------------------------------------------------------------
NORM_KEYS = [
    "field_mean", "field_std",
    "r2prime_mean", "r2prime_std",
    "r2star_mean", "r2star_std",
    "x_pos_mean", "x_pos_std",
    "x_neg_mean", "x_neg_std",
    "cosmos_sus_mean", "cosmos_sus_std",
]


def load_norm_factors(path):
    d = sio.loadmat(path)
    out = {}
    # try direct keys
    for k in NORM_KEYS:
        if k in d:
            out[k] = float(np.array(d[k]).ravel()[0])
    # some toolboxes nest the factors inside a struct; try common containers
    if not out:
        for cand in ("norm_factor", "value_file", "factors"):
            if cand in d:
                s = d[cand]
                try:
                    names = s.dtype.names
                    for k in NORM_KEYS:
                        if k in names:
                            out[k] = float(np.array(s[k][0, 0]).ravel()[0])
                except Exception:
                    pass
    missing = [k for k in NORM_KEYS if k not in out]
    if missing:
        log("  [warn] norm_factor.mat missing keys: %s" % missing)
    return out


# ----------------------------------------------------------------------------
class OnnxNet:
    def __init__(self, path, device="auto"):
        import onnxruntime as ort
        avail = ort.get_available_providers()
        if device == "cpu":
            providers = ["CPUExecutionProvider"]
        elif device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:  # auto
            providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                         if "CUDAExecutionProvider" in avail
                         else ["CPUExecutionProvider"])
        self.sess = ort.InferenceSession(path, providers=providers)
        self.in_name = self.sess.get_inputs()[0].name
        self.in_shape = self.sess.get_inputs()[0].shape
        log("  loaded %s | in=%s shape=%s | providers=%s" % (
            os.path.basename(path), self.in_name, self.in_shape,
            self.sess.get_providers()))

    def run(self, x):
        x = np.ascontiguousarray(x, dtype=np.float32)
        return self.sess.run(None, {self.in_name: x})[0]

    def spatial_size(self):
        """Return (D,H,W) if the model has a FIXED spatial input size, else None.
        ONNX input shape is [N, C, D, H, W]; entries can be ints or strings
        (dynamic). Only returns a tuple when the 3 spatial dims are concrete."""
        sh = self.in_shape
        if len(sh) != 5:
            return None
        spatial = sh[2:5]
        if all(isinstance(d, int) and d > 0 for d in spatial):
            return tuple(int(d) for d in spatial)
        return None


def to5d(vol):
    return vol[np.newaxis, np.newaxis, ...].astype(np.float32)


# ----------------------------------------------------------------------------
def get_field(mat, *names):
    for n in names:
        if n in mat:
            return np.asarray(mat[n], dtype=np.float32)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_mat", required=True)
    ap.add_argument("--output_mat", required=True)
    ap.add_argument("--qsm_onnx", required=True)
    ap.add_argument("--xsep_onnx", required=True)
    ap.add_argument("--r2prime_onnx", default="")
    ap.add_argument("--norm_factor", required=True)
    ap.add_argument("--pipeline", default="auto", choices=["auto", "r2p", "r2s"])
    ap.add_argument("--qsm_source", default="qsmnet",
                    choices=["qsmnet", "external"],
                    help="qsmnet: run QSMnet; external: use WH-QSM/external QSM map")
    ap.add_argument("--field_unit", default="Hz")
    ap.add_argument("--CF", type=float, default=123177385.0)
    ap.add_argument("--Dr", type=float, default=114.0)
    ap.add_argument("--delta_TE", type=float, default=0.0056)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--voxel_size", default="1,1,1",
                    help="voxel size mm 'vx,vy,vz' (for resolution generalization)")
    ap.add_argument("--resgen", default="auto",
                    choices=["auto", "on", "off"],
                    help="resolution generalization: resample to 1mm in k-space "
                         "before feeding the fixed-size network")
    ap.add_argument("--tukey_alpha", type=float, default=0.2)
    args = ap.parse_args()

    log("=" * 60)
    log(" chi-separation ONNX-Runtime bridge (no MATLAB onnxmex)")
    log("=" * 60)

    required = [(args.input_mat, "input_mat"),
                (args.xsep_onnx, "xsep_onnx"),
                (args.norm_factor, "norm_factor")]
    if args.qsm_source == "qsmnet":
        required.append((args.qsm_onnx, "qsm_onnx"))
    for p, what in required:
        if not os.path.isfile(p):
            raise FileNotFoundError("%s not found: %s" % (what, p))

    mat = sio.loadmat(args.input_mat)

    mask = get_field(mat, "mask", "Mask", "mask_4d")
    if mask is None:
        raise KeyError("input mat must contain 'mask'")
    mask = (mask > 0).astype(np.float32)

    field_raw = get_field(mat, "local_field_hz", "local_f_hz_4d",
                          "local_field", "field", "localField_Hz")
    if field_raw is None:
        raise KeyError("input mat must contain 'local_field_hz'")

    r2star = get_field(mat, "r2star_hz", "r2star_4d", "R2star_Hz", "r2star")
    if r2star is None:
        raise KeyError("input mat must contain 'r2star_hz'")

    r2 = get_field(mat, "r2_hz", "r2_4d", "R2_Hz", "r2")

    # optional external QSM (e.g. WH-QSM chi_total in ppm)
    ext_qsm = None
    if args.qsm_source == "external":
        ext_qsm = get_field(mat, "external_qsm", "chi_total_ppm",
                            "whqsm_chi", "qsm_external")
        if ext_qsm is None:
            raise KeyError("qsm_source=external requires 'external_qsm' "
                           "(or 'chi_total_ppm') in the input mat")

    # decide pipeline
    pipeline = args.pipeline
    if pipeline == "auto":
        pipeline = "r2p" if r2 is not None else "r2s"
    if pipeline == "r2p" and r2 is None:
        raise ValueError("pipeline=r2p requires r2_hz in input mat")
    if pipeline == "r2s" and not args.r2prime_onnx:
        raise ValueError("pipeline=r2s requires --r2prime_onnx")
    log("  pipeline      : %s" % pipeline)
    log("  field unit    : %s" % args.field_unit)
    log("  CF/Dr/dTE     : %g / %g / %g" % (args.CF, args.Dr, args.delta_TE))

    # unit conversion -> ppm
    field_ppm = field_to_ppm(field_raw, args.field_unit, args.CF, args.delta_TE)
    r2star_ppm = r2star / args.Dr

    if pipeline == "r2p":
        r2prime = r2star - r2
        r2prime[r2prime < 0] = 0
        r2prime_ppm = r2prime / args.Dr
    else:
        r2prime_ppm = None

    log("  original size : %s" % (field_ppm.shape,))
    log("  QSM source    : %s" % args.qsm_source)

    # normalization factors
    nf = load_norm_factors(args.norm_factor)
    need = ["field_mean", "field_std", "x_pos_mean", "x_pos_std",
            "x_neg_mean", "x_neg_std"]
    if pipeline == "r2p":
        need += ["r2prime_mean", "r2prime_std"]
    else:
        need += ["r2star_mean", "r2star_std"]
    if args.qsm_source == "external":
        # external QSM must be normalized into chi-sepnet input space
        need += ["cosmos_sus_mean", "cosmos_sus_std"]
    miss = [k for k in need if k not in nf]
    if miss:
        raise KeyError("norm_factor.mat is missing required keys for "
                       "pipeline=%s: %s" % (pipeline, miss))

    # load nets
    log("== loading ONNX models ==")
    xsep_net = OnnxNet(args.xsep_onnx, args.device)
    # QSMnet only needed when generating QSM internally
    qsm_net = OnnxNet(args.qsm_onnx, args.device) if args.qsm_source == "qsmnet" else None
    r2p_net = OnnxNet(args.r2prime_onnx, args.device) if pipeline == "r2s" else None

    # ---- Size & resolution handling (official resolution-generalization) ----
    # SNU onnx models are exported with a FIXED size (e.g. 192x192x128), trained
    # at 1mm isotropic. For data that differs in resolution or matrix size we
    # follow Ji et al. (ISMRM 2023) / chi_sepnet_general_new_wResolGen:
    #   1) k-space resample the data to a 1mm-equivalent matrix (Tukey window),
    #   2) image-domain center crop/pad that to the network's fixed size,
    #   3) infer, then invert (2) then (1).
    target = xsep_net.spatial_size()
    fit_info = None
    resgen_src_shape = None       # shape after step-1 (1mm grid), for inversion
    orig_shape = field_ppm.shape
    bbox_for_restore = None
    bbox_orig_shape = orig_shape

    # parse voxel size
    try:
        vox = [float(v) for v in str(args.voxel_size).split(",")]
        if len(vox) != 3:
            vox = [1.0, 1.0, 1.0]
    except Exception:
        vox = [1.0, 1.0, 1.0]

    if target is not None:
        # --- pre-step: crop to brain bounding box (+margin) to drop background.
        # This prevents losing brain when the FOV/matrix is much larger than the
        # network and avoids wasting network FOV on empty background.
        bbox = mask_bbox(mask, margin=8)
        if bbox is not None:
            field_ppm = field_ppm[bbox]
            r2star_ppm = r2star_ppm[bbox]
            mask = mask[bbox]
            if r2prime_ppm is not None: r2prime_ppm = r2prime_ppm[bbox]
            if ext_qsm is not None: ext_qsm = ext_qsm[bbox]
            log("  brain bbox crop: %s -> %s" % (orig_shape, field_ppm.shape))
        bbox_for_restore = bbox
        bbox_orig_shape = orig_shape

        # decide whether resolution generalization is needed
        do_resgen = (args.resgen == "on")
        if args.resgen == "auto":
            do_resgen = any(abs(v - 1.0) > 0.05 for v in vox)
        # also force resgen-style downscale if (post-bbox) in-plane still exceeds net
        post = field_ppm.shape
        if not do_resgen and any(post[a] > target[a] for a in range(3)):
            do_resgen = True
            log("  matrix still > network after bbox -> enabling k-space downscale")

        cur_shape = field_ppm.shape   # post-bbox shape
        if do_resgen:
            # 1mm-equivalent matrix: N_1mm = round(N * vox / 1.0)
            res_shape = [max(1, int(round(cur_shape[a] * vox[a]))) for a in range(3)]
            # never exceed network size on any axis (cap to target, even mid-vox)
            res_shape = tuple(min(res_shape[a], target[a]) for a in range(3))
            log("  resgen ON  | voxel=%s mm | %s -> 1mm-equiv grid %s (k-space)" %
                (vox, cur_shape, res_shape))
            field_ppm  = kspace_resample(field_ppm,  res_shape, args.tukey_alpha)
            r2star_ppm = kspace_resample(r2star_ppm, res_shape, args.tukey_alpha)
            # mask: resample then re-binarize
            mask = (kspace_resample(mask.astype(np.float32), res_shape, args.tukey_alpha) > 0.5).astype(np.float32)
            if r2prime_ppm is not None:
                r2prime_ppm = kspace_resample(r2prime_ppm, res_shape, args.tukey_alpha)
            if ext_qsm is not None:
                ext_qsm = kspace_resample(ext_qsm, res_shape, args.tukey_alpha)
            resgen_src_shape = res_shape
        else:
            log("  resgen OFF | voxel=%s mm (≈1mm)" % (vox,))

        # step 2: fit (crop/pad) to the network's fixed size
        log("  fitting to network size: %s -> %s" % (field_ppm.shape, target))
        field_ppm, fit_info = fit_to_size(field_ppm, target)
        r2star_ppm, _ = fit_to_size(r2star_ppm, target)
        mask, _ = fit_to_size(mask, target)
        if r2prime_ppm is not None:
            r2prime_ppm, _ = fit_to_size(r2prime_ppm, target)
        if ext_qsm is not None:
            ext_qsm, _ = fit_to_size(ext_qsm, target)
    else:
        log("  network is dynamic -> crop to multiples of 16")
        field_ppm = crop_img_16x(field_ppm)
        r2star_ppm = crop_img_16x(r2star_ppm)
        mask = crop_img_16x(mask)
        if r2prime_ppm is not None:
            r2prime_ppm = crop_img_16x(r2prime_ppm)
        if ext_qsm is not None:
            ext_qsm = crop_img_16x(ext_qsm)
    log("  network input size : %s" % (field_ppm.shape,))

    m = mask
    f_n = (field_ppm - nf["field_mean"]) / nf["field_std"] * m

    log("== inference ==")
    if args.qsm_source == "external":
        # Use external QSM (e.g. WH-QSM chi_total, ppm). Normalize the SAME way
        # the official resolution-generalization branch normalizes a pre-computed
        # QSM before feeding chi-sepnet: (qsm - cosmos_sus_mean)/cosmos_sus_std.
        qsm = (ext_qsm - nf["cosmos_sus_mean"]) / nf["cosmos_sus_std"] * m
        log("  using external QSM (WH-QSM), normalized by cosmos_sus_mean/std")
    else:
        # QSMnet (official default)
        qsm = qsm_net.run(to5d(f_n))[0, 0] * m

    # R2'  (rp = normalized R2' fed to chi-sepnet; rp_ppm = physical R2' in ppm)
    if pipeline == "r2s":
        rs_n = (r2star_ppm - nf["r2star_mean"]) / nf["r2star_std"] * m
        rp = r2p_net.run(to5d(rs_n))[0, 0] * m          # normalized (network output space)
        # de-normalize to ppm using r2prime stats (R2PRIMEnet trained to output
        # R2' in the same normalized space as the r2prime_mean/std)
        rp_ppm = (rp * nf["r2prime_std"] + nf["r2prime_mean"]) * m
    else:
        rp = (r2prime_ppm - nf["r2prime_mean"]) / nf["r2prime_std"] * m  # normalized
        rp_ppm = r2prime_ppm * m                         # already ppm

    # chi-sepnet : concat[QSM, field, R2']  (uses NORMALIZED rp)
    xin = np.concatenate([to5d(qsm), to5d(f_n), to5d(rp)], axis=1)
    pred = xsep_net.run(xin)  # [1,2,X,Y,Z]
    if pred.shape[1] < 2:
        raise RuntimeError("chi-sepnet output has %d channels, expected 2 "
                           "[chi_para, chi_dia]" % pred.shape[1])

    x_pos = pred[0, 0] * nf["x_pos_std"] + nf["x_pos_mean"]
    x_neg = pred[0, 1] * nf["x_neg_std"] + nf["x_neg_mean"]
    x_pos[x_pos < 0] = 0
    x_neg[x_neg < 0] = 0
    x_pos *= m
    x_neg *= m
    x_tot = x_pos - x_neg   # toolbox convention: total = para - dia(>=0)

    # de-normalize QSM back to ppm for output/QC (input space was normalized)
    if "cosmos_sus_mean" in nf and "cosmos_sus_std" in nf:
        qsm_ppm = (qsm * nf["cosmos_sus_std"] + nf["cosmos_sus_mean"]) * m
    else:
        qsm_ppm = qsm  # best effort if cosmos_sus factors absent (qsmnet pipeline)

    # ---- restore to the original grid if we fixed the size for the network ----
    if fit_info is not None:
        # invert step 2 (image crop/pad back to the 1mm grid or original)
        x_pos   = restore_from_size(x_pos,   fit_info)
        x_neg   = restore_from_size(x_neg,   fit_info)
        x_tot   = restore_from_size(x_tot,   fit_info)
        qsm_ppm = restore_from_size(qsm_ppm, fit_info)
        rp_ppm  = restore_from_size(rp_ppm,  fit_info)
        m       = restore_from_size(m,       fit_info)
        # invert step 1 (k-space resample 1mm grid -> post-bbox grid)
        if resgen_src_shape is not None:
            cur = bbox_shape_after(bbox_for_restore, bbox_orig_shape)
            x_pos   = kspace_resample(x_pos,   cur, args.tukey_alpha)
            x_neg   = kspace_resample(x_neg,   cur, args.tukey_alpha)
            x_tot   = kspace_resample(x_tot,   cur, args.tukey_alpha)
            qsm_ppm = kspace_resample(qsm_ppm, cur, args.tukey_alpha)
            rp_ppm  = kspace_resample(rp_ppm,  cur, args.tukey_alpha)
            m       = (kspace_resample(m, cur, args.tukey_alpha) > 0.5).astype(np.float32)
            x_pos[x_pos < 0] = 0; x_neg[x_neg < 0] = 0
            x_pos *= m; x_neg *= m; x_tot = (x_pos - x_neg) * m
        # invert bbox crop -> place back into full original volume
        if bbox_for_restore is not None:
            x_pos   = place_in_full(x_pos,   bbox_for_restore, bbox_orig_shape)
            x_neg   = place_in_full(x_neg,   bbox_for_restore, bbox_orig_shape)
            x_tot   = place_in_full(x_tot,   bbox_for_restore, bbox_orig_shape)
            qsm_ppm = place_in_full(qsm_ppm, bbox_for_restore, bbox_orig_shape)
            rp_ppm  = place_in_full(rp_ppm,  bbox_for_restore, bbox_orig_shape)
            m       = place_in_full(m,       bbox_for_restore, bbox_orig_shape)
        log("  restored to original size: %s" % (x_pos.shape,))

    log("== saving ==")
    out = {
        "x_para": x_pos.astype(np.float32),
        "x_dia": x_neg.astype(np.float32),
        "x_tot": x_tot.astype(np.float32),
        "qsm_map": (qsm_ppm).astype(np.float32),
        "qsm_source": args.qsm_source,
        "r2prime_map": (rp_ppm).astype(np.float32),  # PPM (physical), usable by opt
        "mask_out": m.astype(np.float32),
        "pipeline": pipeline,
    }
    sio.savemat(args.output_mat, out)
    log("  wrote %s" % args.output_mat)
    log("chi-separation ONNX inference done.")


if __name__ == "__main__":
    main()
