"""CPU-only regression tests for live acquisition, quality, tiling and comparison."""
from __future__ import annotations
from datetime import datetime,timezone,timedelta
from email.utils import format_datetime
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase,main
from unittest.mock import patch
import sys
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.agents.jma import TimeOptions,AcquisitionAgent
from app.agents.model import tiles_400,predict_image
from app.agents.comparison import jma_magenta_mask,compare,visualization
from app.agents.orchestrator import QualityAgent,MultiAgentOrchestrator

class FakeResponse:
    def __init__(self,body,date=None):
        self.content=body
        self.headers={"Last-Modified":format_datetime(date)} if date else {}
    def raise_for_status(self):pass

class FakeHTTP:
    def __init__(self):
        self.today=datetime.now(timezone.utc)
        self.old=self.today-timedelta(days=1)
        im=Image.new("RGB",(500,400),(50,50,50))
        b=BytesIO();im.save(b,format="JPEG")
        self.jpeg=b.getvalue()
    def get(self,url,**kwargs):
        if "sat_hrp.php" in url:
            return FakeResponse(b'<select name="slt_time"><option value="1200">12:10 UTC</option><option value="1150">12:00 UTC</option></select>')
        return FakeResponse(self.jpeg,self.old if "_1200.jpg" in url else self.today)

class FakeModel:
    def __call__(self,tiles,training=False):
        h,w=tiles.shape[1:3]
        return np.ones((len(tiles),h,w,1),dtype=np.float32)*0.6

class AgentTests(TestCase):
    def test_html_time(self):
        h=TimeOptions();h.feed('<select name="slt_time"><option value="0400">04:10 UTC</option></select>')
        self.assertEqual(h.options,[("0400","04:10 UTC")])
    def test_tiles_cover_whole_image(self):
        for w,h in [(400,400),(1501,901),(800,800),(401,402)]:
            tiles=tiles_400((w,h));mask=np.zeros((h,w),np.uint8)
            for x0,y0,x1,y1 in tiles:mask[y0:y1,x0:x1]=1
            self.assertTrue(np.all(mask))
    def test_reference_color_not_rainfall(self):
        im=Image.new("RGB",(400,400),(0,0,0))
        im.putpixel((125,125),(255,0,255))
        mag=jma_magenta_mask(im)
        self.assertTrue(mag[125,125]);self.assertEqual(int(np.sum(mag)),1)
    def test_metrics_and_overlap(self):
        hrp=Image.new("RGB",(400,400),(0,0,0))
        for y in range(50,100):
            for x in range(50,100):hrp.putpixel((x,y),(255,0,255))
        probabilities=np.zeros((400,400),np.float32)
        probabilities[50:100,50:100]=.95
        m=compare(probabilities,hrp,margin_ratio=0)
        self.assertEqual(m["dsc_vs_jma_visual"],1.0)
        self.assertEqual(m["iou_vs_jma_visual"],1.0)
        overlay=visualization(hrp,probabilities,hrp)
        self.assertEqual(overlay.size,(400,400))
    def test_inference_requires_real_model(self):
        image=Image.new("RGB",(500,400),(0,0,0))
        pred=predict_image(image,FakeModel())
        self.assertEqual(pred.shape,(400,500))
        self.assertAlmostEqual(float(np.mean(pred)),.6,places=4)
    def test_acquisition_skips_stale_latest_token(self):
        with TemporaryDirectory() as temp:
            frame=AcquisitionAgent(Path(temp),session=FakeHTTP()).fetch()
            self.assertTrue(frame.fresh)
            self.assertEqual(frame.time_token,"1150")
            self.assertTrue(frame.b13_path.exists())
            self.assertEqual(frame.size,(500,400))
    def test_quality_agent_rejects_stale(self):
        with TemporaryDirectory() as temp:
            frame=AcquisitionAgent(Path(temp),session=FakeHTTP()).fetch()
            frame.fresh=False
            with self.assertRaises(RuntimeError):
                QualityAgent().inspect(frame)
    def test_orchestrator_mock_without_tensorflow(self):
        with TemporaryDirectory() as temp:
            root=Path(temp)
            job=MultiAgentOrchestrator(root=root)
            job.acquisition=AcquisitionAgent(root/"data"/"live",session=FakeHTTP())
            result=job.run(model=FakeModel(),threshold=.5)
            self.assertTrue(result["report_path"].exists())
            self.assertEqual(result["record"]["model"],"RainNet (dropout 0.5)")
            self.assertEqual(len(result["record"]["agents"]),5)
if __name__=="__main__":main()
