"""Compare dissertation dropout checkpoints using the 118-sample validation set.

This is a post hoc audit, not an independent evaluation. Dataset provenance,
label quality, possible near-temporal duplicates and spatial leakage require
manual verification. The public repo ships no data/weights.
"""
from __future__ import annotations
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/"legacy"/"NEW_PROJECT"
sys.path.insert(0,str(SRC/"model-rain-net"))
import rainNetModel2labels as original

WEIGHTS=SRC/"saved-model-and-history"/"RainNet"/"DATASET 4 dropoutRate"
INPUT=SRC/"hdf5-out"/"DATASET 4"/"input_image_validation.h5"
LABEL=SRC/"hdf5-out"/"DATASET 4"/"output_mask_validation.h5"
SELECTED=ROOT/"models"/"selected"
REPORT=ROOT/"reports"/"validation_model_selection.json"

def per_image_metrics(actual: np.ndarray, predicted: np.ndarray):
    y=np.asarray(actual)>0.5
    p=np.asarray(predicted)>0.5
    if p.ndim==4 and p.shape[-1]==1:p=p[...,0]
    if y.ndim==4 and y.shape[-1]==1:y=y[...,0]
    if y.shape!=p.shape:
        raise ValueError("GT and model prediction shape mismatch: "+str((y.shape,p.shape)))
    intersection=np.count_nonzero(p&y,axis=(1,2))
    n_p=np.count_nonzero(p,axis=(1,2))
    n_y=np.count_nonzero(y,axis=(1,2))
    union=n_p+n_y-intersection
    # Class empty for both: perfect agreement, documented.
    dsc=np.where(n_p+n_y==0,1.0,2*intersection/np.maximum(n_p+n_y,1))
    iou=np.where(union==0,1.0,intersection/np.maximum(union,1))
    return dsc,iou

def main():
    SELECTED.mkdir(parents=True,exist_ok=True)
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    files=sorted(WEIGHTS.glob("*dropoutRate_*.h5"))
    if len(files)!=5:
        raise RuntimeError(f"Expected 5 historical dropout candidates, found {len(files)}")
    with h5py.File(INPUT,"r") as fi,h5py.File(LABEL,"r") as fl:
        x=fi["input_image_validation"];y=fl["output_mask_validation"]
        if x.shape[0]!=y.shape[0] or tuple(x.shape[1:])!=(400,400,3):
            raise ValueError("Validation input/mask shapes changed")
        report={"dataset":"DATASET 4 validation","samples":int(x.shape[0]),
                "protocol":"full validation set, threshold=0.5, per-image DSC/IoU, no model retraining",
                "selection":"highest mean validation DSC, ties decided on mean validation IoU",
                "candidate_models":[]}
        for file in files:
            rate=re.search(r"dropoutRate_(0\.\d+)",file.name).group(1)
            model=tf.keras.models.load_model(str(file),custom_objects={
                "dice_coeff":original.dice_coeff,"Encoder":original.Encoder,
                "Bridge":original.Bridge,"Decoder":original.Decoder},compile=False)
            all_dsc=[];all_iou=[]
            for start in range(0,len(x),4):
                pred=model(np.asarray(x[start:start+4],dtype=np.float32)/255.0,training=False).numpy()
                d,i=per_image_metrics(np.asarray(y[start:start+4]),pred)
                all_dsc.extend(d.tolist());all_iou.extend(i.tolist())
            result={"dropout":float(rate),"mean_dsc":round(float(np.mean(all_dsc)),6),
                    "std_dsc":round(float(np.std(all_dsc)),6),
                    "mean_iou":round(float(np.mean(all_iou)),6),
                    "zero_positive_masks":sum(not np.any(z) for z in y[:]),
                    "checkpoint":file.name}
            report["candidate_models"].append(result)
            print("VALIDATED",result,flush=True)
            del model
            tf.keras.backend.clear_session()
    ranked=sorted(report["candidate_models"],key=lambda d:(-d["mean_dsc"],-d["mean_iou"],d["dropout"]))
    winner=ranked[0]
    report["selected"]=winner
    report["caveats"]=[
        "The same validation set may have been used during historical training and hyperparameter search.",
        "This result is not an independent test-set benchmark or meteorological validation.",
        "Potential timestamp/spatial leakage between image tiles has not been audited.",
        "Reported DSC/IoU are mask agreement against historical labels, not rainfall gauge accuracy."
    ]
    source=next(f for f in files if f"dropoutRate_{winner['dropout']:.1f}.h5" in f.name)
    dest=SELECTED/"rainnet_binary_validation_best.h5"
    shutil.copy2(source,dest)
    h=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    if h(source)!=h(dest):raise RuntimeError("Selected weight checksum mismatch")
    report["selected_file"]={"name":dest.name,"sha256":h(dest),"source":str(source)}
    REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print("VALIDATION_SELECTION_OK",winner,"report",REPORT,flush=True)

if __name__=="__main__":main()
