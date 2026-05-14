import os
import sys
import argparse
from collections import OrderedDict

import numpy as np
import scipy.io as sio
import torch


def zero_pad(arr, multiple=8):
    pad_spec = []
    pos = []
    for s in arr.shape:
        total = (multiple - s % multiple) % multiple
        before = total // 2
        after = total - before
        pad_spec.append((before, after))
        pos.append((before, before + s))
    arr_pad = np.pad(arr, pad_spec, mode="constant")
    return arr_pad, pos


def zero_remove(arr, pos):
    (x1, x2), (y1, y2), (z1, z2) = pos
    return arr[x1:x2, y1:y2, z1:z2]


def load_model(checkpoint, xqsm_root, device):
    py_dir = os.path.join(xqsm_root, "python")
    if py_dir not in sys.path:
        sys.path.insert(0, py_dir)

    from xQSM import xQSM  # pylint: disable=import-error

    net = xQSM(2)

    weights = torch.load(checkpoint, map_location="cpu")
    new_state = OrderedDict()
    for k, v in weights.items():
        name = k[7:] if k.startswith("module.") else k
        new_state[name] = v

    net.load_state_dict(new_state, strict=True)
    net = net.to(device).eval()
    return net


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_mat", required=True)
    parser.add_argument("--output_mat", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--xqsm_root", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    mat = sio.loadmat(args.input_mat)

    input_norm = np.array(mat["input_norm"], dtype=np.float32)
    mask = (np.array(mat["Mask"]) > 0).astype(np.float32)
    norm_factor = float(np.array(mat["norm_factor"]).squeeze())

    field = input_norm * norm_factor
    if field.ndim != 3:
        raise RuntimeError("field input must be 3D")

    field_pad, pos = zero_pad(field, 8)

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda:0")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = load_model(args.checkpoint, args.xqsm_root, device)

    x = torch.from_numpy(field_pad).float().unsqueeze(0).unsqueeze(0).to(device)

    with torch.inference_mode():
        y = net(x)

    chi = y.squeeze().detach().cpu().numpy().astype(np.float32)
    chi = zero_remove(chi, pos)
    chi = chi * mask

    sio.savemat(args.output_mat, {"chi": chi})
    print("xQSM inference done.")


if __name__ == "__main__":
    main()