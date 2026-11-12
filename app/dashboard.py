"""RainNet V1: local VS Code Streamlit GUI, four scientific agents + reporting.

Run: python -m streamlit run app/dashboard.py --server.port 8501
"""
from __future__ import annotations
import json
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
div[data-testid="stMetric"]{border:1px solid #4660774d;border-radius:14px;padding:12px}
footer{visibility:hidden}
</style>
""",unsafe_allow_html=True)

st.title("RainNet V1 — Himawari Live Analysis")
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
    st.divider()
    st.info("Warna sian = hasil RainNet. Area putih pada citra gabungan = tumpang-tindih dengan tanda magenta JMA.")

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
    a.image(result["b13"],caption="Source B13: JMA Himawari IR",use_column_width=True)
    b.image(result["hrp"],caption="JMA HRP visual product (magenta)",use_column_width=True)
    c.image(result["overlay"],caption="RainNet (cyan) + overlap JMA (white)",use_column_width=True)
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
