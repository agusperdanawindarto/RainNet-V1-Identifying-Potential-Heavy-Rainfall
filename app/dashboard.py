"""RainNet V1: local VS Code Streamlit GUI, four scientific agents + reporting.

Run: python -m streamlit run app/dashboard.py --server.port 8501
"""
from __future__ import annotations
import json
import csv
import io
from pathlib import Path
import sys
import streamlit as st

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from app.agents.orchestrator import MultiAgentOrchestrator
from app.agents.model import load_model

st.set_page_config(page_title="RainNet V1 | Himawari Live Comparison",
                   page_icon="☁️",layout="wide",initial_sidebar_state="expanded")

st.markdown("""
<style>
.block-container{max-width:1500px;padding-top:1.6rem}
div[data-testid="stMetric"]{border:1px solid #8ba9f7;border-radius:14px;padding:12px}
div.stButton > button[kind="primary"]{background:#1248d6!important;border:1px solid #073ba9!important;color:#ffffff!important;font-weight:700!important}
div.stButton > button[kind="primary"]:hover{background:#1b60f3!important;border-color:#0b45bf!important}
footer{visibility:hidden}
</style>
""",unsafe_allow_html=True)

st.title("RainNet V1 — Himawari Live Analysis")
st.caption("BUILD 2026-11-12 | RAINNET BIRU PEKAT | 38 PROVINSI — Versi terbaru")
st.markdown("**Legenda:** RainNet = :blue[BIRU PEKAT] | HRP JMA = MAGENTA | Area cocok = BIRU TERANG")
st.caption("Research-grade CNN segmentation of deep convective clouds | Dissertation 2024 | JMA reference")
st.warning("This is a scientific visualization of potential heavy-rainfall areas, NOT a rainfall amount measurement, flood warning, or validated weather forecast.")

with st.sidebar:
    st.subheader("Pengaturan Analisis")
    model_name=st.selectbox("Model terlatih",["RainNet (dropout 0.5)","RainNet (original)","U-Net baseline"])
    threshold=st.slider("Ambang probabilitas segmentasi",min_value=0.1,max_value=0.9,value=0.5,step=0.05)
    autorefresh=st.checkbox("Perbarui otomatis setiap 10 menit",value=True)
    st.write("Data input: **JMA B13 / r2w Southeast Asia Large**")
    st.write("Referensi: **JMA HRP / r2w, timestamp identik**")
    st.write("Mode: model asli, inference tanpa training")
    st.markdown("[JMA Heavy Rainfall Potential (r2s)](https://www.data.jma.go.jp/mscweb/data/himawari/sat_hrp.php?area=r2s)")
    st.markdown("[JMA Users' Guide](https://www.data.jma.go.jp/mscweb/data/himawari/Users_Guide.pdf)")
    st.markdown("[Data batas 38 provinsi · CC BY 4.0](https://github.com/denyherianto/indonesia-geojson-topojson-maps-with-38-provinces)")
    st.divider()
    st.info("Biru pekat = RainNet; biru lebih terang = area yang juga ditandai HRP JMA; magenta = warna produk JMA. "
            "Label provinsi menyatakan cakupan piksel terindikasi saat pengamatan, bukan prakiraan hujan atau peringatan BMKG.")

@st.cache_resource(show_spinner="Memuat model hasil eksperimen 2024...")
def loaded_checkpoint(name):
    return load_model(name)

def run_panel():
    st.subheader("Pemantauan Citra Satelit")
    start=st.button("Ambil Data & Analisis Sekarang",type="primary",use_container_width=True)
    if not autorefresh and not start and "last_analysis" not in st.session_state:
        st.info("Klik tombol di atas untuk memulai sinkronisasi citra dan inference.")
        return
    if start or autorefresh:
        try:
            m=loaded_checkpoint(model_name)
            with st.spinner("JMA Acquisition → Quality Gate → RainNet → Reference Comparison → Report..."):
                result=MultiAgentOrchestrator().run(model_name,threshold,require_fresh=True,model=m)
            st.session_state["last_analysis"]=result
        except Exception as exc:
            st.error(f"Proses dihentikan secara aman: {type(exc).__name__}: {exc}")
            st.info("Tanpa citra JMA yang fresh dan model yang valid, sistem tidak mengarang prediksi.")
            return
    result=st.session_state.get("last_analysis")
    if result is None:return
    frame=result["frame"]; stats=result["record"]["metrics"]
    st.success("Akuisisi, validasi, inference, komparasi, dan pelaporan berhasil.")
    st.caption(f"JMA site label: {frame.page_label} · asset key: {frame.time_token} · area: {frame.area} · updated HTTP: {frame.last_modified_hrp} · fetched UTC: {frame.fetched_utc}")
    if frame.age_minutes is None:
        st.warning("Timestamps HTTP unavailable — product freshness not verified")
    elif frame.age_minutes>30:
        st.warning(f"JMA product age: {frame.age_minutes} minutes")
    row=st.columns(4)
    row[0].metric("Dice vs JMA HRP",f'{stats["dsc_vs_jma_visual"]:.4f}' if stats["dsc_vs_jma_visual"] is not None else "N/A")
    row[1].metric("IoU vs JMA HRP",f'{stats["iou_vs_jma_visual"]:.4f}' if stats["iou_vs_jma_visual"] is not None else "N/A")
    row[2].metric("RainNet area in evaluated pixels",str(stats["predicted_fraction_pct"])+"%")
    row[3].metric("JMA HRP area in evaluated pixels",str(stats["jma_fraction_pct"])+"%")
    a,b,c=st.columns(3)
    a.image(result["b13"],caption="Sumber B13 — Himawari inframerah",use_container_width=True)
    b.image(result["hrp"],caption="Referensi HRP JMA — magenta",use_container_width=True)
    c.image(result["overlay"],caption="RainNet biru pekat · Kesesuaian dengan JMA biru terang",use_container_width=True)

    st.subheader("Indikasi Potensi Hujan Berdasarkan Provinsi")
    st.caption("Indikasi awan konvektif pada waktu observasi JMA — BUKAN prediksi hujan pada jam/hari berikutnya.")
    provinces=result.get("province_summary",[])
    province_image=result.get("province_overlay")
    if provinces and province_image is not None:
        indicated=[p for p in provinces if p["indikasi"]!="Belum terindikasi"]
        st.write("**"+str(len(indicated))+" dari 38 provinsi** memiliki area terindikasi; "
                 "label di peta menampilkan maksimum 12 provinsi dengan cakupan relatif tertinggi.")
        st.image(province_image,
                 caption="Peta Provinsi Indonesia · Biru pekat: model · Biru terang: kesesuaian dengan HRP JMA",
                 use_container_width=True)
        st.caption("Sebaran lokal: ≥0,5%, sebaran luas: ≥3%, sangat luas: ≥10% piksel provinsi. "
                   "Klasifikasi ini hanya menggambarkan sebaran piksel, bukan intensitas atau peluang hujan.")
        region_rows=[{
            "Provinsi":p["provinsi"],
            "Status sebaran":p["indikasi"],
            "RainNet (%)":p["luas_indikasi_rainnet_pct"],
            "JMA HRP (%)":p["area_hrp_jma_pct"],
            "Piksel model":p["piksel_rainnet"],
        } for p in provinces]
        st.dataframe(region_rows,hide_index=True,use_container_width=True)
        buffer=io.StringIO()
        writer=csv.DictWriter(buffer,fieldnames=list(region_rows[0]))
        writer.writeheader()
        writer.writerows(region_rows)
        st.download_button("Unduh Rekap 38 Provinsi (CSV)",buffer.getvalue().encode("utf-8-sig"),
                           file_name="RainNet_indikasi_38_provinsi.csv",mime="text/csv")
    else:
        st.warning("Label provinsi tidak tersedia: citra harus sesuai georeferensi JMA r2w 1501×901.")
    st.info("Validitas geografis: batas provinsi 38 wilayah diproyeksikan ke grid JMA 0,05°. "
            "Hasilnya adalah indikasi spasial dari citra, bukan estimasi mm/jam atau peringatan dini resmi.")
    with st.expander("Proses multi-agent, waktu observasi, dan audit ilmiah",expanded=True):
        for a in result["record"]["agents"]:
            st.write("**"+a["agent"]+":** "+a["result"])
        st.info("Comparison metric is an exploratory pixel overlap with visual JMA HRP. "
                "The original model was trained using masks derived from JMA imagery; these metrics do NOT provide independent validation.")
        st.json(stats["scientific_limits"])
        st.code(str(result["report_path"]),language="text")
    st.download_button("Unduh laporan JSON",json.dumps(result["record"],ensure_ascii=False,indent=2),
                       file_name="RainNet_JMA_report.json",mime="application/json")


if autorefresh and hasattr(st,"fragment"):
    st.caption("Pemutakhiran otomatis berjalan selama tab Streamlit aktif, sekitar setiap 10 menit.")
    st.fragment(run_every="10m")(run_panel)()
else:
    if autorefresh:
        st.info("Streamlit versi ini belum mendukung fragment refresh; gunakan tombol manual atau tingkatkan Streamlit.")
    run_panel()
