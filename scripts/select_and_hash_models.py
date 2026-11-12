from pathlib import Path
import shutil,hashlib,json
root=Path(r"D:\DISERTASI_S3\PROGRAM_RAINNET")
source=Path(r"D:\DISERTASI_S3\ProjectDisertasi\NEW_PROJECT\saved-model-and-history")
entries={
 "rainnet_binary_best_05.h5": source/"RainNet"/"DATASET 4 dropoutRate"/"rain_net_epoch_1000_batchSize_1_lr_1e-4_filterSeq_4-8-16-32-64_lrSchedule_constant_dropoutRate_0.5.h5",
 "rainnet_binary_baseline_00.h5": source/"RainNet"/"DATASET 4"/"rain_net_epoch_1000_batchSize_1_lr_1e-4_filterSeq_4-8-16-32-64_lrSchedule_constant.h5",
 "unet_binary.h5": source/"UNet"/"DATASET 4"/"sergey_unet_epoch_1000_batchSize_1_lr_1e-4_filterSeq_4-8-16-32-64_lrSchedule_constant.h5",
 "rainnet_multiclass.h5": source/"RainNet"/"DATASET 5"/"rain_net_epoch_1000_batchSize_1_lr_1e-4_filterSeq_4-8-16-32-64_lrSchedule_constant.h5",
 "unet_multiclass.h5": source/"UNet"/"DATASET 5"/"sergey_unet_epoch_1000_batchSize_1_lr_1e-4_filterSeq_4-8-16-32-64_lrSchedule_constant.h5"
}
dest=root/"models"/"selected"
dest.mkdir(parents=True,exist_ok=True)
record={}
for name,src in entries.items():
 if not src.is_file():raise FileNotFoundError(src)
 dst=dest/name
 if not dst.exists() or dst.stat().st_size!=src.stat().st_size:shutil.copy2(src,dst)
 if dst.stat().st_size!=src.stat().st_size:raise IOError("Copy size differs")
 def h(p):
  x=hashlib.sha256()
  with p.open("rb") as f:
   for chunk in iter(lambda:f.read(2**20),b""):x.update(chunk)
  return x.hexdigest()
 digest=h(src)
 assert digest==h(dst)
 record[name]={"source":str(src),"sha256":digest,"bytes":src.stat().st_size}
 print("MODEL_SELECTED",name,"MB",round(src.stat().st_size/1e6,1),flush=True)
(root/"models"/"model_manifest.json").write_text(json.dumps(record,indent=2),encoding="utf-8")
print("SELECTED_MODELS_READY",len(record),flush=True)
