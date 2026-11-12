"""Fetch the attributed 38-province GeoJSON artifact. Run only for updates.

Source: denyherianto/indonesia-geojson-topojson-maps-with-38-provinces
GeoJSON data: CC BY 4.0, see upstream DATA_LICENSE.md.
"""
from pathlib import Path
import hashlib
import json
import requests

ROOT=Path(__file__).resolve().parents[1]
REPOSITORY="denyherianto/indonesia-geojson-topojson-maps-with-38-provinces"
API="https://api.github.com/repos/"+REPOSITORY+"/commits/main"
SOURCE="GeoJSON/indonesia-38-provinces.geojson"

def main():
    commit=requests.get(API,timeout=20)
    commit.raise_for_status()
    sha=commit.json()["sha"]
    url=f"https://raw.githubusercontent.com/{REPOSITORY}/{sha}/{SOURCE}"
    response=requests.get(url,timeout=30)
    response.raise_for_status()
    if len(response.content)>5_000_000:
        raise ValueError("Province boundary asset unexpectedly large")
    obj=response.json()
    provinces=obj.get("features")
    if not isinstance(provinces,list) or len(provinces)!=38:
        raise ValueError("Expected exactly 38 Indonesian provinces")
    names=[str(f.get("properties",{}).get("PROVINSI","")).strip() for f in provinces]
    if not all(names) or len(set(names))!=38:
        raise ValueError("Missing/duplicate province names")
    if not all(x.get("geometry",{}).get("type") in ("Polygon","MultiPolygon") for x in provinces):
        raise ValueError("Unsupported geographic geometry")
    dest=ROOT/"app"/"assets"/"indonesia_38_provinces.geojson"
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_bytes(response.content)
    meta={
        "source":f"https://github.com/{REPOSITORY}",
        "url":url,
        "revision":sha,
        "license":"CC BY 4.0",
        "source_license":f"https://github.com/{REPOSITORY}/blob/{sha}/DATA_LICENSE.md",
        "modified":"Included unchanged; names and simplified geometry are upstream data.",
        "sha256":hashlib.sha256(response.content).hexdigest(),
        "province_count":38,
        "names":sorted(names),
    }
    (ROOT/"app"/"assets"/"province_source.json").write_text(
        json.dumps(meta,indent=2,ensure_ascii=False),encoding="utf-8")
    print("PROVINCE_BOUNDARIES_READY",len(names),sha,meta["sha256"],len(response.content))
    print("NAMES",names)

if __name__=="__main__": main()
