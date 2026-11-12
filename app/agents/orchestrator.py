"""Deterministic multi-agent workflow: acquire -> QC -> infer -> compare -> report."""
from __future__ import annotations
from datetime import datetime, timezone
from dataclasses import asdict
from pathlib import Path
import json
from PIL import Image

from .jma import AcquisitionAgent, JMAFrame
from .model import load_model, predict_image, ROOT
from .comparison import compare, visualization


class QualityAgent:
    def inspect(self, frame: JMAFrame, *, require_fresh=True):
        b13=Image.open(frame.b13_path).convert("RGB")
        hrp=Image.open(frame.hrp_path).convert("RGB")
        if b13.size != hrp.size:
            raise RuntimeError("JMA reference and source imagery are not pixel-aligned")
        if require_fresh and not frame.fresh:
            raise RuntimeError(
                f"JMA imagery is stale or undated ({frame.age_minutes} min). "
                "Live inference blocked; old frames are not called real-time.")
        if frame.area != "r2w":
            raise RuntimeError("Trained model expects Southeast Asia Large (r2w) JMA composition")
        return b13, hrp


class ReportingAgent:
    def __init__(self, root: Path = ROOT):
        self.root=Path(root)

    def save(self, record: dict, b13: Image.Image, hrp: Image.Image,
             overlay: Image.Image | None):
        directory=self.root/"outputs"/"runs"/record["frame"]["fetched_utc"][:10]
        directory.mkdir(parents=True, exist_ok=True)
        token=record["frame"]["time_token"]
        run=(record["frame"]["area"]+"_"+token+"_"+record["frame"]["fetched_utc"][11:19].replace(":",""))
        base=directory/run
        (directory/(run+"_report.json")).write_text(json.dumps(record,indent=2,ensure_ascii=False),encoding="utf-8")
        if overlay is not None:
            overlay.save(directory/(run+"_RainNet-vs-JMA.png"))
        return directory/(run+"_report.json")


class MultiAgentOrchestrator:
    def __init__(self, root: Path=ROOT):
        self.root=Path(root)
        self.acquisition=AcquisitionAgent(self.root/"data"/"live",area="r2w")
        self.quality=QualityAgent()
        self.reporting=ReportingAgent(self.root)

    def run(self, model_name="RainNet (dropout 0.5)",threshold=0.5,
            *,require_fresh=True,model=None):
        audit=[]
        frame=self.acquisition.fetch()
        audit.append({"agent":"JMA Acquisition","result":"OK","time_key":frame.time_token,
                      "fetched_utc":frame.fetched_utc,"fresh":frame.fresh})
        b13,hrp=self.quality.inspect(frame,require_fresh=require_fresh)
        audit.append({"agent":"Quality & Alignment","result":"OK",
                      "size":list(b13.size),"fresh":frame.fresh})
        trained_model=model if model is not None else load_model(model_name)
        prob=predict_image(b13,trained_model,batch_size=1)
        audit.append({"agent":"RainNet Inference","result":"OK",
                      "model":model_name,"input_size":list(b13.size)})
        statistics=compare(prob,hrp,threshold=threshold)
        audit.append({"agent":"JMA Reference Comparison","result":"OK",
                      "type":"screen-space agreement; not rainfall amount"})
        blend=visualization(b13,prob,hrp,threshold=threshold)
        record={
            "frame":{k:(str(v) if isinstance(v,Path) else v) for k,v in asdict(frame).items()},
            "model":model_name,"metrics":statistics,"agents":audit,
            "provenance":{"JMA_page":"https://www.data.jma.go.jp/mscweb/data/himawari/sat_hrp.php?area=r2s",
                          "model_source":"Agus Perdana Windarto, dissertation RainNet, 2024",
                          "data_source":"JMA Himawari; synchronized B13 and HRP render via identical area and asset key"},
            "use_limit":"Research visualization only, not a public warning or weather forecast."
        }
        report=self.reporting.save(record,b13,hrp,blend)
        audit.append({"agent":"Reporting","result":"OK","report":str(report)})
        return {"frame":frame,"b13":b13,"hrp":hrp,"overlay":blend,
                "record":record,"report_path":report}
