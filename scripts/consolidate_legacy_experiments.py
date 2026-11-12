from pathlib import Path
import shutil,hashlib,json,time
base=Path(r"D:\DISERTASI_S3")
root=base/"PROGRAM_RAINNET"
src=base/"ProjectDisertasi"/"NEW_PROJECT"
old=base/"ProjectDisertasi"/"Project-RainNet-20230309T073036Z-003"/"Project-RainNet"
entries=[]
for origin,target,historical in [(src,root/"legacy"/"NEW_PROJECT",False),(old,root/"legacy"/"PROJECT_RAINNET_2023",True)]:
    if not origin.exists():raise FileNotFoundError(origin)
    for p in origin.rglob("*"):
        if not p.is_file() or any(x in ("__pycache__", ".git", ".ipynb_checkpoints") for x in p.relative_to(origin).parts) or p.suffix==".pyc":continue
        if historical and (p.suffix.lower() in (".h5",".npy",".dat",".zip",".rar") or p.stat().st_size>40_000_000):continue
        entries.append((p,target/p.relative_to(origin)))
print("CONSOLIDATION_START",len(entries),"TOTAL_GB",round(sum(p.stat().st_size for p,_ in entries)/1e9,2),flush=True)
n=0;skipped=0;begin=time.monotonic()
for p,d in entries:
    d.parent.mkdir(parents=True,exist_ok=True)
    if d.exists() and d.stat().st_size==p.stat().st_size:skipped+=1
    else:shutil.copy2(p,d)
    if d.stat().st_size!=p.stat().st_size:raise IOError("Mismatch "+str(d))
    n+=1
    if n%100==0:print("COPY_CHECKED",n,"/",len(entries),"SECONDS",round(time.monotonic()-begin),flush=True)
def hashfile(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""):h.update(b)
    return h.hexdigest()
checks=[]
for glob in ["saved-model-and-history/RainNet/DATASET 4 dropoutRate/*0.5.h5","saved-model-and-history/RainNet/DATASET 5/*.h5","saved-model-and-history/UNet/DATASET 4/*4-8-16-32-64*.h5","saved-model-and-history/UNet/DATASET 5/*.h5","model-rain-net/gui-tkinter/rainnet-gui.py","model-rain-net/model_rainNet-gooleColabs.ipynb"]:
    for p in src.glob(glob):
        d=root/"legacy"/"NEW_PROJECT"/p.relative_to(src)
        h=hashfile(p)
        if hashfile(d)!=h:raise IOError("SHA256 mismatch "+str(d))
        checks.append({"file":str(d.relative_to(root)),"sha256":h})
manifest={"policy":"Copy-and-verify; originals preserved","source":str(src),"history":str(old),"destination":str(root),"files":n,"already_present":skipped,"selected_hashes":checks,"seconds":round(time.monotonic()-begin,1)}
(root/"logs"/"consolidation_20261112.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
print("CONSOLIDATION_COMPLETE",json.dumps({"files":n,"skipped":skipped,"checks":len(checks),"seconds":manifest["seconds"]}),flush=True)
