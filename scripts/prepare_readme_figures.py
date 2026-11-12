"""Curate reproducible, compact figures for the public RainNet GitHub README.

The 2023 figures are copied from the user's historic dissertation archive
without changing the plotted DSC values. Newly generated charts are directly
derived from original .npy and retrospective validation JSON.
JMA Himawari images remain attributed to JMA and are research visualisations.
"""
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import json
import shutil
import hashlib

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"legacy"/"NEW_PROJECT"
OUT=ROOT/"docs"/"figures"
OUT.mkdir(parents=True,exist_ok=True)

pairs={
    "2023_1400_rai_unet_reference.png": SOURCE/"Hasil PRediksi"/"1400.png",
    "2023_1400_groundtruth_masks.png": SOURCE/"Hasil GroundTruth"/"1400.png",
    "2023_1500_rai_unet_reference.png": SOURCE/"Hasil PRediksi"/"1500.png",
    "2023_1500_groundtruth_masks.png": SOURCE/"Hasil GroundTruth"/"1500.png",
}
for name,path in pairs.items():
    if not path.is_file():raise FileNotFoundError(path)
    with Image.open(path) as im:
        if im.width<800 or im.height<300:
            raise ValueError(f"Unexpected figure size: {path}: {im.size}")
        im.save(OUT/name,optimize=True)

history_root=SOURCE/"saved-model-and-history"/"RainNet"/"DATASET 4 dropoutRate"
history_file=next(history_root.glob("*dropoutRate_0.5_history.npy"))
hist=np.load(str(history_file),allow_pickle=True).item()
for label in ("dice_coeff","val_dice_coeff","loss","val_loss"):
    if not isinstance(hist.get(label), (list, np.ndarray)) or len(hist[label])!=1000:
        raise ValueError(f"Missing full 1,000-epoch history: {label}")

plt.rcParams.update({"font.size":11,"figure.dpi":130})
fig,axs=plt.subplots(1,2,figsize=(12.8,4.4),constrained_layout=True)
epochs=np.arange(1,len(hist["dice_coeff"])+1)
axs[0].plot(epochs,hist["dice_coeff"],label="Train DSC",linewidth=.85,color="#2455a4",alpha=.8)
axs[0].plot(epochs,hist["val_dice_coeff"],label="Validation DSC",linewidth=.85,color="#d07a27",alpha=.8)
axs[0].set(ylabel="DSC (training metric)",xlabel="Epoch",title="Historical RainNet training – Dice coefficient")
axs[0].set_ylim(0,1)
axs[0].legend(loc="lower right",frameon=False)
axs[0].grid(alpha=.18)
axs[1].plot(epochs,hist["loss"],label="Train loss",linewidth=.85,color="#2455a4",alpha=.8)
axs[1].plot(epochs,hist["val_loss"],label="Validation loss",linewidth=.85,color="#d07a27",alpha=.8)
axs[1].set(ylabel="Original training loss",xlabel="Epoch",title="Historical RainNet training – loss")
axs[1].legend(loc="upper right",frameon=False)
axs[1].grid(alpha=.18)
fig.suptitle("Archived experiment: Dataset 4, dropout 0.5, 1,000 epochs",fontweight="bold")
fig.savefig(OUT/"training_history_dropout05.png",bbox_inches="tight",dpi=120)
plt.close(fig)

report=json.loads((ROOT/"reports"/"validation_model_selection.json").read_text(encoding="utf-8"))
records=sorted(report["candidate_models"],key=lambda d:d["dropout"])
rates=[float(x["dropout"]) for x in records]
dice=[x["mean_dsc"] for x in records]
iou=[x["mean_iou"] for x in records]
fig,ax=plt.subplots(figsize=(10.2,4.5),constrained_layout=True)
xx=np.arange(len(rates))
bars=ax.bar(xx-.18,dice,width=.36,label="Mean DSC",color="#2054a7")
ax.bar(xx+.18,iou,width=.36,label="Mean IoU",color="#da8439")
ax.set_xticks(xx,[f"{r:.1f}" for r in rates])
ax.set(ylabel="Mean validation mask overlap",xlabel="RainNet dropout rate",
       title="Retrospective comparison of five trained RainNet checkpoints")
ax.set_ylim(0,1)
ax.legend(loc="upper right",frameon=False)
ax.grid(axis="y",alpha=.19)
ax.set_axisbelow(True)
for i,(d,j) in enumerate(zip(dice,iou)):
    ax.text(i-.18,d+.012,f"{d:.3f}",ha="center",fontsize=8)
    ax.text(i+.18,j+.012,f"{j:.3f}",ha="center",fontsize=8)
ax.text(.01,-.22,"118 archived validation images; binary pixel threshold 0.5. Not independent test-set or rainfall validation.",
        transform=ax.transAxes,fontsize=8.7)
fig.savefig(OUT/"validation_dropout_comparison.png",bbox_inches="tight",dpi=135)
plt.close(fig)

# A lightweight live-system illustration of the BLUE overlay (not historical
# experiment performance). Label the timestamp precisely and don't claim live
# JMA source is a rain-gauge ground truth.
overlays=sorted((ROOT/"outputs"/"runs"/"2026-11-12").glob("*_RainNet_Provinces_Labeled.png"),
                key=lambda p:p.stat().st_mtime)
if not overlays:raise FileNotFoundError("No live province overlay")
latest=overlays[-1]
with Image.open(latest) as im:
    im.thumbnail((1300,900),Image.Resampling.LANCZOS if hasattr(Image,"Resampling") else Image.LANCZOS)
    im.convert("RGB").save(OUT/"prototype_2026_blue_province_overlay.jpg",
                          quality=82,optimize=True,progressive=True)

manifest=[]
for f in sorted(OUT.iterdir()):
    if f.suffix.lower() not in {".png",".jpg"}:continue
    with Image.open(f) as im:size=im.size
    meta={"name":f.name,"source":str(pairs.get(f.name, latest if "prototype" in f.name else "archived training history / validation audit")),
          "size":size,"bytes":f.stat().st_size,
          "sha256":hashlib.sha256(f.read_bytes()).hexdigest()}
    manifest.append(meta)
    print("FIGURE",f.name,size,"bytes",meta["bytes"],flush=True)
(ROOT/"reports"/"readme_figures_manifest.json").write_text(
    json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
print("FIGURES_PREPARED",len(manifest),flush=True)
