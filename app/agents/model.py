"""RainNet legacy inference using ORIGINAL 400x400 RGB input and saved models.

No random predictions or dummy masks are returned when model loading fails.
The original Keras layers are imported from the preserved dissertation code.
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SELECTED = ROOT / "models" / "selected"
LEGACY_MODULE = ROOT / "legacy" / "NEW_PROJECT" / "model-rain-net"


def tiles_400(size: tuple[int,int]) -> list[tuple[int,int,int,int]]:
    width,height=size
    if width<400 or height<400:
        raise ValueError("Original RainNet requires both image dimensions >=400 pixels")
    rows=[i*400 for i in range(height//400)]+[height-400]
    cols=[i*400 for i in range(width//400)]+[width-400]
    rows=sorted(set(rows))
    cols=sorted(set(cols))
    return [(x,y,x+400,y+400) for y in rows for x in cols]


def predict_image(image: Image.Image, model, *, batch_size=1) -> np.ndarray:
    """Average probability in overlaps; preserve exact input resolution."""
    rgb=np.asarray(image.convert("RGB"),dtype=np.float32)/255.
    h,w=rgb.shape[:2]
    boxes=tiles_400((w,h))
    total=np.zeros((h,w,1),dtype=np.float32)
    n=np.zeros((h,w,1),dtype=np.float32)
    for k in range(0,len(boxes),batch_size):
        subset=boxes[k:k+batch_size]
        stack=np.stack([rgb[y0:y1,x0:x1] for x0,y0,x1,y1 in subset],axis=0)
        pred=model(stack,training=False)
        pred=np.asarray(pred)
        if pred.ndim!=4 or pred.shape[1:3]!=(400,400):
            raise RuntimeError(f"Model produced invalid tile shape {pred.shape}")
        if pred.shape[-1]!=1:
            raise RuntimeError("Expected one sigmoid channel for binary segmentation")
        for box,prob in zip(subset,pred):
            x0,y0,x1,y1=box
            total[y0:y1,x0:x1]+=prob
            n[y0:y1,x0:x1]+=1.0
    if np.any(n==0):
        raise RuntimeError("One or more image pixels were not covered by model tiles")
    return np.clip((total/n)[:,:,0],0,1)


def load_model(key: str):
    """Load selected legacy artifacts; fail closed on TensorFlow incompatibility."""
    paths={
        "RainNet (dropout 0.5)":"rainnet_binary_best_05.h5",
        "RainNet (original)":"rainnet_binary_baseline_00.h5",
        "U-Net baseline":"unet_binary.h5",
    }
    if key not in paths:
        raise ValueError(f"Unknown approved binary model: {key}")
    file=SELECTED/paths[key]
    if not file.is_file():
        raise FileNotFoundError(f"Selected trained checkpoint absent: {file}")
    modules = LEGACY_MODULE if (LEGACY_MODULE/"rainNetModel2labels.py").is_file() else ROOT/"app"/"architectures"
    if not (modules/"rainNetModel2labels.py").is_file():
        raise FileNotFoundError("Original RainNet architecture code is missing.")
    if str(modules) not in sys.path:
        sys.path.insert(0,str(modules))
    import tensorflow as tf
    import rainNetModel2labels as original
    custom = {"dice_coeff":original.dice_coeff}
    if not key.startswith("U-Net"):
        custom.update({"Encoder":original.Encoder,"Bridge":original.Bridge,
                       "Decoder":original.Decoder})
    model=tf.keras.models.load_model(str(file),custom_objects=custom,compile=False)
    if tuple(model.input_shape[1:])!=(400,400,3):
        raise RuntimeError(f"Checkpoint input does not match legacy preprocessing: {model.input_shape}")
    if model.output_shape[-1]!=1:
        raise RuntimeError(f"Checkpoint is not binary: {model.output_shape}")
    return model
