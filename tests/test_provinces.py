"""Regression tests of 38-province spatial attribution and deep blue RainNet overlay."""
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from app.agents.comparison import visualization
from app.agents.provinces import (
    GRID_SIZE, _features, _classify, longitude_latitude_to_pixel,
    province_mask, analyze_provinces, draw_province_labels,
)


class ProvinceUnitTests(unittest.TestCase):
    def test_indonesia_has_38_unique_provinces(self):
        feats=_features()
        self.assertEqual(len(feats),38)
        names={f["properties"]["PROVINSI"] for f in feats}
        self.assertEqual(len(names),38)
        self.assertIn("Papua Barat Daya",names)
        self.assertIn("Papua Selatan",names)
        self.assertIn("Sumatera Utara",names)

    def test_known_geographic_points_inside_correct_provinces(self):
        sample={
            "DKI Jakarta":(106.8272,-6.1754),
            "Sumatera Utara":(98.6722,3.5952),
            "Bali":(115.2167,-8.6500),
        }
        feats={f["properties"]["PROVINSI"]:f for f in _features()}
        for province,(lon,lat) in sample.items():
            x,y=longitude_latitude_to_pixel(lon,lat)
            with self.subTest(provinsi=province):
                self.assertTrue(province_mask(feats[province])[round(y),round(x)])

    def test_reject_non_native_georeferencing(self):
        feature=_features()[0]
        with self.assertRaises(ValueError):
            province_mask(feature,(500,400))
        with self.assertRaises(ValueError):
            analyze_provinces(np.zeros((400,500),np.float32),
                              Image.new("RGB",(500,400)))

    def test_province_allocation_and_labelling(self):
        feats={f["properties"]["PROVINSI"]:f for f in _features()}
        north=province_mask(feats["Sumatera Utara"])
        loc=np.argwhere(north)
        # 3 valid positive pixels *inside* North Sumatra
        sampled=loc[:6]
        prob=np.zeros((901,1501),dtype=np.float32)
        prob[sampled[:,0],sampled[:,1]]=0.90
        raw=Image.new("RGB",GRID_SIZE,(70,70,70))
        hrp=Image.new("RGB",GRID_SIZE,(70,70,70))
        output=analyze_provinces(prob,hrp)
        self.assertEqual(len(output),38)
        north_row=next(p for p in output if p["provinsi"]=="Sumatera Utara")
        self.assertGreaterEqual(north_row["piksel_rainnet"],3)
        self.assertGreater(north_row["luas_indikasi_rainnet_pct"],0)
        labelled=draw_province_labels(raw,output)
        self.assertEqual(labelled.size,GRID_SIZE)

    def test_deep_blue_model_blue_agreement(self):
        base=Image.new("RGB",(400,400),(40,40,40))
        jma=base.copy()
        jma.putpixel((80,80),(255,0,255))
        pred=np.zeros((400,400),np.float32)
        pred[80,80]=.99
        pred[100,100]=.99
        out=visualization(base,pred,jma)
        a=out.getpixel((100,100))
        b=out.getpixel((80,80))
        # Both model-only and model+JMA pixels must stay distinctly BLUE.
        self.assertGreater(a[2],a[0]+130)
        self.assertGreater(a[2],a[1]+130)
        self.assertGreater(b[2],b[0]+130)
        self.assertGreater(b[2],b[1]+85)
        self.assertGreater(b[1],a[1]+35)  # JMA agreement uses brighter blue

    def test_coverage_classes_descriptive_only(self):
        self.assertEqual(_classify(0.4,200),"Belum terindikasi")
        self.assertEqual(_classify(1.0,8),"Sebaran lokal")
        self.assertEqual(_classify(4.0,8),"Sebaran luas")
        self.assertEqual(_classify(11.0,100),"Sebaran sangat luas")
        self.assertEqual(_classify(25.0,2),"Belum terindikasi")


if __name__=="__main__":
    unittest.main()
