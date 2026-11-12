"""Reference comparison against JMA's Heavy Rainfall Potential IMAGE, not rain gauges."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
from PIL import Image


def jma_magenta_mask(image: Image.Image) -> np.ndarray:
    """Screen-space extraction of JMA's magenta HRP annotation.

    This is an approximation to visual product colors; JPEG compression may
    cause differences. It is NOT physical rainfall truth or mm/h data.
    """
    arr=np.asarray(image.convert("RGB"),dtype=np.int16)
    red,green,blue=arr[:,:,0],arr[:,:,1],arr[:,:,2]
    return ((red>=120)&(blue>=105)&
            (red>green+38)&(blue>green+32)&
            (np.abs(red-blue)<110))


def crop_valid_roi(size: tuple[int,int], margin_ratio=0.06) -> np.ndarray:
    w,h=size
    dx=int(w*margin_ratio);dy=int(h*margin_ratio)
    valid=np.zeros((h,w),dtype=bool)
    valid[dy:h-dy,dx:w-dx]=True
    return valid


def compare(pred_probability: np.ndarray, hrp: Image.Image,
            *, threshold=0.5, margin_ratio=0.06) -> dict:
    if tuple(pred_probability.shape)!=(hrp.height,hrp.width):
        raise ValueError("Unaligned image dimensions. Do not resize HRP to force a comparison")
    if not 0.1<=threshold<=0.9:
        raise ValueError("Model confidence threshold must lie within [0.1,0.9]")
    ref=jma_magenta_mask(hrp)
    predicted=pred_probability>=threshold
    valid=crop_valid_roi(hrp.size,margin_ratio)
    inter=int(np.count_nonzero(predicted&ref&valid))
    a=int(np.count_nonzero(predicted&valid))
    b=int(np.count_nonzero(ref&valid))
    union=a+b-inter
    dsc=(2*inter/(a+b)) if (a+b)>0 else None
    iou=(inter/union) if union>0 else None
    total=int(np.count_nonzero(valid))
    return {
      "dsc_vs_jma_visual":round(dsc,4) if dsc is not None else None,
      "iou_vs_jma_visual":round(iou,4) if iou is not None else None,
      "predicted_pixels":a, "jma_hrp_pixels":b,"intersection_pixels":inter,
      "evaluated_pixels":total,
      "predicted_fraction_pct":round(100*a/total,2),
      "jma_fraction_pct":round(100*b/total,2),
      "model_threshold":threshold, "edge_margin":margin_ratio,
      "comparison_type":"image-space agreement with JMA visual HRP product, not rainfall verification",
      "scientific_limits":[
        "JMA HRP does not measure actual rainfall or precipitation amount.",
        "JMA uses multiple IR bands; RainNet was trained on an RGB render of B13 imagery.",
        "JMA HRP reference is automatically extracted by approximate magenta threshold.",
        "The margin crop excludes visual borders but does not replace exact georeferencing.",
        "The same JMA product family was used to derive original research masks; this is not an independent ground-truth benchmark.",
      ],
    }


def visualization(b13: Image.Image, probabilities: np.ndarray,
                  reference: Image.Image, threshold=0.5) -> Image.Image:
    if b13.size != reference.size or probabilities.shape!=(b13.height,b13.width):
        raise ValueError("Images not registered at same pixels")
    base=np.asarray(b13.convert("RGB"),dtype=np.float32)
    mask=probabilities>=threshold
    out=base.copy()
    # Cyan for RainNet; pink/magenta is JMA HRP; white both.
    out[mask]=out[mask]*0.45+np.array([0,225,235])*0.55
    ref=jma_magenta_mask(reference)
    both=mask&ref
    out[both]=out[both]*0.20+np.array([255,255,255])*0.80
    return Image.fromarray(np.uint8(np.clip(out,0,255)),"RGB")
