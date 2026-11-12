from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.agents.jma import AcquisitionAgent
root=Path(__file__).resolve().parents[1]
f=AcquisitionAgent(root/"data"/"live",area="r2w").fetch()
print("JMA_PAIR",json.dumps({"area":f.area,"token":f.time_token,"site_time":f.page_label,
"dimensions":f.size,"age_minutes":f.age_minutes,"fresh":f.fresh,"b13":str(f.b13_path),
"hrp":str(f.hrp_path)},ensure_ascii=False),flush=True)
