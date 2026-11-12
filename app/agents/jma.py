"""JMA Himawari B13 / HRP synchronized near-real-time acquisition.

The website is authoritative for displayed time options; asset filename uses
the *value* of the selected option, not the displayed label. Both bands use
the same area and asset token. No claim of measured rainfall is made.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
import re
import time
import requests
from PIL import Image

BASE_URL = "https://www.data.jma.go.jp/mscweb/data/himawari/"
PAGE_URL = BASE_URL + "sat_hrp.php?area=r2s"
MAX_BYTES = 12 * 1024 * 1024
USER_AGENT = "RainNet-V1-Research/1.0 (educational scientific comparison)"


class TimeOptions(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_select = False
        self.in_option = False
        self.options = []
        self.token = ""
        self.text_parts = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "select" and a.get("name") == "slt_time":
            self.in_select = True
        elif tag == "option" and self.in_select:
            self.in_option = True
            self.token = a.get("value", "")
            self.text_parts = []

    def handle_data(self, data):
        if self.in_option:
            self.text_parts.append(data)

    def handle_endtag(self, tag):
        if tag == "option" and self.in_option:
            label = "".join(self.text_parts).strip()
            if re.fullmatch(r"\d{4}", self.token):
                self.options.append((self.token, label))
            self.in_option = False
        elif tag == "select" and self.in_select:
            self.in_select = False


@dataclass
class JMAFrame:
    area: str
    time_token: str
    page_label: str
    b13_path: Path
    hrp_path: Path
    b13_url: str
    hrp_url: str
    fetched_utc: str
    last_modified_b13: str
    last_modified_hrp: str
    age_minutes: float | None
    fresh: bool
    size: tuple[int, int]


class AcquisitionAgent:
    def __init__(self, outdir: Path, session=None, area: str = "r2w", max_age_minutes=60):
        if area not in ("r2s", "r2w"):
            raise ValueError("Only the JMA Southeast Asia r2s/r2w products are supported")
        self.area = area
        self.outdir = Path(outdir)
        self.http = session or requests.Session()
        self.max_age_minutes = float(max_age_minutes)

    def _get(self, url: str, image=False):
        res = self.http.get(url, timeout=(8, 24), headers={"User-Agent": USER_AGENT},
                            params={"_": str(int(time.time()))} if image else None,
                            allow_redirects=True)
        res.raise_for_status()
        if len(res.content) > MAX_BYTES:
            raise ValueError("JMA response exceeds image size safety limit")
        if image:
            if not res.content.startswith(b"\xff\xd8"):
                raise ValueError("JMA did not return valid JPEG bytes: " + url)
            with Image.open(BytesIO(res.content)) as im:
                im.verify()
        return res

    @staticmethod
    def _modified(value: str | None) -> datetime | None:
        try:
            d = parsedate_to_datetime(value)
            return d.astimezone(timezone.utc)
        except (TypeError, ValueError):
            return None

    def fetch(self) -> JMAFrame:
        page = self._get(PAGE_URL)
        parser = TimeOptions()
        parser.feed(page.content.decode("utf-8", "replace"))
        if not parser.options:
            raise RuntimeError("No JMA image time options found; website layout may have changed")
        # PHP options can expose a future token before that image is regenerated.
        # The matching filename may still contain yesterday's JPEG. Select the
        # newest *verified fresh* B13+HRP pair, not blindly the first option.
        found = False
        diagnostics = []
        now = datetime.now(timezone.utc)
        for token,label in parser.options[:12]:
            urls = {
                band: BASE_URL + f"img/{self.area}/{self.area}_{band}_{token}.jpg"
                for band in ("b13", "hrp")
            }
            try:
                b13 = self._get(urls["b13"], image=True)
                hrp = self._get(urls["hrp"], image=True)
                b13_img = Image.open(BytesIO(b13.content))
                hrp_img = Image.open(BytesIO(hrp.content))
                if b13_img.size != hrp_img.size or min(b13_img.size) < 400:
                    raise ValueError("dimensions invalid")
                t1 = self._modified(b13.headers.get("Last-Modified"))
                t2 = self._modified(hrp.headers.get("Last-Modified"))
                age = (now - min(t1, t2)).total_seconds()/60 if t1 and t2 else None
                paired = (t1 is not None and t2 is not None
                          and abs((t1-t2).total_seconds()) <= 20*60)
                fresh = (paired and age is not None
                         and -5 <= age <= self.max_age_minutes)
                if fresh:
                    found = True
                    break
                diagnostics.append(f"{token}: age={age}, B13={t1}, HRP={t2}")
            except (requests.RequestException, ValueError, OSError) as exc:
                diagnostics.append(f"{token}: {type(exc).__name__}")
        if not found:
            raise RuntimeError("No fresh matching JMA B13+HRP frame. "
                               + "; ".join(diagnostics[:4]))
        self.outdir.mkdir(parents=True, exist_ok=True)
        day = now.strftime("%Y%m%d")
        paths = {}
        for band, response in (("b13",b13),("hrp",hrp)):
            target = self.outdir / f"{day}_{self.area}_{band}_{token}.jpg"
            tmp = target.with_suffix(".part")
            tmp.write_bytes(response.content)
            tmp.replace(target)
            paths[band] = target
        return JMAFrame(
            area=self.area, time_token=token, page_label=label,
            b13_path=paths["b13"], hrp_path=paths["hrp"],
            b13_url=urls["b13"], hrp_url=urls["hrp"],
            fetched_utc=now.isoformat(),
            last_modified_b13=t1.isoformat() if t1 else "unknown",
            last_modified_hrp=t2.isoformat() if t2 else "unknown",
            age_minutes=round(age,1) if age is not None else None,
            fresh=fresh, size=b13_img.size
        )
