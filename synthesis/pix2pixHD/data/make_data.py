# make_pairs_5ch.py
import os, re
from pathlib import Path
import numpy as np
from PIL import Image

SRC = "/Users/cristina/thesis/DWI_DCE_CDFR-DNN_/dce"
OUT = "/Users/cristina/Desktop/SimulatingDCE/datasets/dce_p12345"
SIZE = 512  # = --loadSize = --fineSize

Path(f"{OUT}/train_A").mkdir(parents=True, exist_ok=True)
Path(f"{OUT}/train_B").mkdir(parents=True, exist_ok=True)

groups = {}
for f in os.listdir(SRC):
    m = re.match(r"dce_img(\d+)_channel_(\d)\.png$", f)
    if m:
        i, ch = int(m.group(1)), int(m.group(2))
        groups.setdefault(i, {})[ch] = os.path.join(SRC, f)

for i, g in sorted(groups.items()):
    # 至少要有 0..3，4/5 缺失可用 0 填
    if not all(c in g for c in [0,1,2,3]):
        continue
    def readL(p): return Image.open(p).convert("L").resize((SIZE,SIZE), Image.BICUBIC)
    pre = readL(g[0])
    A = Image.merge("RGB", (pre, pre, pre))
    A.save(f"{OUT}/train_A/img{i:05d}.png")

    chans = []
    for c in [1,2,3,4,5]:
        if c in g:
            chans.append(np.array(readL(g[c])))
        else:
            chans.append(np.zeros((SIZE,SIZE), np.uint8))
    B = np.stack(chans, axis=-1)   # H×W×5, uint8
    np.save(f"{OUT}/train_B/img{i:05d}.npy", B)

print("Done.")
