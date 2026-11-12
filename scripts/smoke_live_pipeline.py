from pathlib import Path
import sys,time,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.agents.orchestrator import MultiAgentOrchestrator
start=time.perf_counter()
try:
 r=MultiAgentOrchestrator().run()
 print("LIVE_PIPELINE_OK",json.dumps({
 "source":r["frame"].b13_url,
 "time":r["frame"].page_label,
 "age":r["frame"].age_minutes,
 "dsc":r["record"]["metrics"]["dsc_vs_jma_visual"],
 "iou":r["record"]["metrics"]["iou_vs_jma_visual"],
 "report":str(r["report_path"]),
 "elapsed_s":round(time.perf_counter()-start,1)},ensure_ascii=False),flush=True)
except Exception as exc:
 import traceback
 print("LIVE_PIPELINE_FAILED",type(exc).__name__,str(exc),flush=True)
 traceback.print_exc()
 raise
