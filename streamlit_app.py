import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# Mengatasi error DecompressionBomb karena ukuran gambar terlalu besar
Image.MAX_IMAGE_PIXELS = None

# 1. SET CONFIG UTAMA
st.set_page_config(
    page_title="SoyLeaf-Guard | UII",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed" 
)

# 2. SUNTIKAN CSS PREMIUM
st.markdown("""
    <style>
    /* Background Global Modern */
    .stApp {
        background-color: #F5F9F7;
        font-family: 'Inter', sans-serif;
    }
    
    /* Menyembunyikan Sidebar */
    [data-testid="collapsedControl"] { display: none !important; }
    [data-testid="stSidebar"] { display: none !important; }
    
    /* Navbar Atas Horizontal */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: #FFFFFF;
        padding: 10px 20px;
        border-radius: 14px;
        box-shadow: 0 4px 20px rgba(31, 111, 95, 0.06);
        border: 1px solid #E2EFEA;
        justify-content: center;
        margin-bottom: 40px;
    }
    
    /* Tombol Menu */
    .stTabs [data-baseweb="tab"] {
        height: 46px;
        background-color: #EEF5F2;
        border-radius: 10px;
        color: #1F6F5F;
        font-weight: 600;
        font-size: 0.95rem;
        padding: 0px 24px;
        transition: all 0.25s ease;
        border: none !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #6FCF97;
        color: #1F6F5F;
        transform: translateY(-1px);
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1F6F5F !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 12px rgba(31, 111, 95, 0.2) !important;
    }

    .stTabs [data-baseweb="tab-border"] { display: none !important; }

    /* Gaya Judul Utama */
    .hero-title {
        color: #1F6F5F;
        font-weight: 800;
        font-size: 3.2rem;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 4px;
    }
    .hero-sub {
        color: #2FA084;
        font-size: 1.25rem;
        text-align: center;
        margin-bottom: 45px;
        font-weight: 500;
    }
    
    /* Subjudul Bagian */
    .section-header {
        color: #1F6F5F;
        font-weight: 700;
        font-size: 1.6rem;
        margin-top: 35px;
        margin-bottom: 25px;
        padding-left: 5px;
        border-bottom: 2px solid #E2EFEA;
        padding-bottom: 8px;
    }

    /* KOTAK RINGKASAN LATAR BELAKANG & LANDASAN TEORI (SKRIPSI STYLE) */
    .bg-skripsi-box {
        background-color: #FFFFFF;
        border-radius: 14px;
        padding: 25px;
        border: 1px solid #E2EFEA;
        box-shadow: 0 4px 15px rgba(31, 111, 95, 0.04);
        margin-bottom: 30px;
    }

    .info-text {
        color: #2F3E3A;
        line-height: 1.7;
        font-size: 1rem;
        text-align: justify;
        margin-bottom: 15px;
    }

    /* KOTAK TIMBUL PENYAKIT (HORIZONTAL ROW DESIGN) */
    .disease-vertical-box {
        background-color: #FFFFFF;
        border-radius: 16px;
        box-shadow: 0 8px 20px rgba(31, 111, 95, 0.06);
        border: 1px solid #E2EFEA;
        padding: 20px;
        margin-bottom: 20px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .disease-vertical-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 25px rgba(31, 111, 95, 0.1);
    }

    /* WADAH FOTO LANDSCAPE DISAMPING - KECIL */
    .image-landscape-wrapper {
        max-width: 240px;
        height: 150px;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 3px 10px rgba(0,0,0,0.06);
        border: 1px solid #E2EFEA;
    }
    
    .image-landscape-wrapper img {
        width: 100% !important;
        height: 100% !important;
        object-fit: cover !important;
    }

    .disease-box-title {
        color: #1F6F5F;
        font-weight: 700;
        font-size: 1.25rem;
        margin-bottom: 8px;
        text-align: left;
    }

    .disease-box-desc {
        color: #2F3E3A;
        font-size: 0.98rem;
        line-height: 1.6;
        text-align: justify;
    }
    
    /* GAYA VISUAL LANGKAH PENGGUNAAN SATU KOTAK UTUH */
    .single-step-box {
        background-color: #FFFFFF;
        border-radius: 14px;
        padding: 22px;
        border: 1px solid #E2EFEA;
        border-left: 5px solid #1F6F5F;
        box-shadow: 0 4px 15px rgba(31, 111, 95, 0.04);
        margin-bottom: 30px;
    }
    
    .step-title-text {
        color: #1F6F5F;
        font-weight: 700;
        font-size: 1.15rem;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .step-list {
        margin: 0;
        padding-left: 20px;
    }

    .step-list li {
        margin-bottom: 12px;
        color: #2F3E3A;
        font-size: 0.98rem;
        line-height: 1.5;
    }
    
    /* Tombol Eksekusi */
    div.stButton > button:first-child {
        background-color: #2FA084 !important;
        color: white !important;
        border-radius: 10px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        border: none !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
        box-shadow: 0 4px 12px rgba(47, 160, 132, 0.2) !important;
    }
    div.stButton > button:first-child:hover {
        background-color: #1F6F5F !important;
    }
    
    /* Badge Hasil */
    .detection-badge {
        background-color: #EEF5F2;
        border-left: 5px solid #2FA084;
        padding: 14px;
        border-radius: 8px;
        margin-bottom: 10px;
        font-weight: 600;
        color: #1F6F5F;
        font-size: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# 3. NAVBAR HORIZONTAL ATAS WEBSITE
tab_home, tab_upload, tab_realtime = st.tabs([
    "Halaman Utama", 
    "Deteksi via Upload Gambar", 
    "Deteksi Kamera Real-Time"
])

# 4. CORE MODEL LOADING WITH CACHING
@st.cache_resource
def load_model():
    try:
        model = YOLO("best.pt")
        return model
    except:
        return None

model = load_model()

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

class YoloVideoTransformer(VideoTransformerBase):
    def __init__(self): self.model = model
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        return self.model(img, conf=0.25)[0].plot(boxes=False)


if model is not None:
    # ==================== TAB 1: HALAMAN UTAMA ====================
    with tab_home:
        st.markdown('<h1 class="hero-title">SoyLeaf-Guard</h1>', unsafe_allow_html=True)
        st.markdown('<p class="hero-sub">Sistem Komputasi Pakar Identifikasi Dini Penyakit Daun Kedelai</p>', unsafe_allow_html=True)
        
        st.markdown('<div class="bg-skripsi-box" style="margin-top: 20px;">', unsafe_allow_html=True)
        st.markdown("""
        <p class="info-text">
        Kedelai (<i>Glycine max L.</i>) merupakan tanaman kacang-kacangan kaya nutrisi yang berfungsi sebagai sumber protein nabati utama bagi masyarakat, 
        serta memiliki sifat adaptif tinggi dan berperan sebagai pupuk hijau penyubur tanah. 
        Meskipun bernilai strategis, produktivitas komoditas ini kerap terancam oleh keberadaan patogen makroskopis yang menyerang organ vegetatif, 
        khususnya pada area permukaan daun.
        </p>
        <p class="info-text">
        Secara umum, terdapat empat jenis infeksi penyakit daun utama yang merugikan di lapangan, yaitu Karat Daun (<i>Phakopsora pachyrhizi</i>), 
        Pustul Bakteri (<i>Xanthomonas axonopodis</i>), Embun Bulu (<i>Peronospora manshurica</i>), dan Bercak Target (<i>Corynespora cassiicola</i>). 
        Serangan patogen ini dapat memicu klorosis jaringan, kerusakan klorofil, hingga keguguran daun pramatang yang berpotensi menurunkan hasil panen secara signifikan.
        </p>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-header">Jenis Penyakit Daun Kedelai</div>', unsafe_allow_html=True)
        
        diseases = [
            {"title": "1. Karat Daun (Soybean Rust)", "desc": "Disebabkan oleh infeksi jamur patogen Phakopsora pachyrhizi...", "filename": "karatdaun.jpg"},
            {"title": "2. Pustul Bakteri (Bacterial Pustule)", "desc": "Disebabkan oleh agen infeksi bakteri Xanthomonas axonopodis...", "filename": "pustulbakteri.jpg"},
            {"title": "3. Embun Bulu (Downy Mildew)", "desc": "Disebabkan oleh cendawan oomycete Peronospora manshurica...", "filename": "embunbulu.jpg"},
            {"title": "4. Bercak Target (Target Spot)", "desc": "Disebabkan oleh jamur nekrotrofik Corynespora cassiicola...", "filename": "bercaktarget.jpg"},
            {"title": "5. Daun Sehat (Healthy Leaf)", "desc": "Kondisi kontrol pembanding dimana organ daun memiliki pigmen klorofil...", "filename": "healthy.jpg"}
        ]

        for d in diseases:
            st.markdown('<div class="disease-vertical-box">', unsafe_allow_html=True)
            inner_col1, inner_col2 = st.columns([1, 4])
            
            with inner_col1:
                st.markdown('<div class="image-landscape-wrapper">', unsafe_allow_html=True)
                if os.path.exists(d["filename"]):
                    st.image(d["filename"], width="stretch")
                else:
                    st.error("No Image")
                st.markdown('</div>', unsafe_allow_html=True)
                
            with inner_col2:
                st.markdown(f'<div class="disease-box-title">{d["title"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="disease-box-desc">{d["desc"]}</div>', unsafe_allow_html=True)
                
            st.markdown('</div>', unsafe_allow_html=True)

    # ==================== TAB 2: DETEKSI UPLOAD ====================
    with tab_upload:
        st.markdown('<div class="section-header">Deteksi Upload Gambar</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="single-step-box"><div class="step-title-text">📋 Langkah Penggunaan:</div><ul class="step-list"><li><b>Unggah Citra Daun:</b> Klik area berkas di bawah untuk memasukkan foto daun kedelai.</li><li><b>Jalankan Inferensi:</b> Klik tombol hijau <b>"Deteksi Penyakit Daun"</b>.</li></ul></div>', unsafe_allow_html=True)
            
        uploaded_file = st.file_uploader("Pilih file gambar daun kedelai:", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            col1, col2 = st.columns(2, gap="large")
            
            with col1:
                st.subheader("Gambar Asli")
                st.image(image, width="stretch")
                btn_trigger = st.button("Deteksi Penyakit Daun")
            
            with col2:
                st.subheader("Hasil Deteksi Penyakit Daun")
                if btn_trigger:
                    img_array = np.array(image)
                    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    results = model(img_bgr)
                    result_img = results[0].plot(boxes=False)
                    result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                    st.image(result_img_rgb, width="stretch")
                else:
                    st.info("Silakan klik tombol di samping untuk memproses gambar.")
            
            if btn_trigger:
                st.write("")
                st.subheader("Detail Klasifikasi & Kepercayaan")
                if len(results[0].boxes) > 0:
                    for i, box in enumerate(results[0].boxes):
                        class_id = int(box.cls[0])
                        class_name = results[0].names[class_id]
                        confidence = float(box.conf[0])
                        st.markdown(f'<div class="detection-badge">Objek #{i+1} : <b>{class_name}</b> — Confidence Score: <b>{(confidence*100):.2f}%</b></div>', unsafe_allow_html=True)
                else:
                    st.warning("Model tidak mendeteksi adanya gejala penyakit pada sampel daun ini.")

    # ==================== TAB 3: DETEKSI REAL-TIME ====================
    with tab_realtime:
        st.markdown('<div class="section-header">Deteksi Kamera Real-Time</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="single-step-box"><div class="step-title-text">📋 Langkah Penggunaan Kamera Real-Time:</div><ul class="step-list"><li><b>Mulai Deteksi:</b> Tekan tombol <b>"START"</b> pada jendela pemutar WebRTC.</li></ul></div>', unsafe_allow_html=True)
            
        col_cam1, col_cam2 = st.columns([2, 1], gap="large")
        with col_cam1:
            st.subheader("Aliran Video WebRTC")
            webrtc_streamer(
                key="yolov9-clean-final",
                video_processor_factory=YoloVideoTransformer,
                rtc_configuration=RTC_CONFIGURATION,
                media_stream_constraints={"video": True, "audio": False}
            )
            
        with col_cam2:
            st.subheader("Catatan Teknis Stabilitas")
            st.caption("- Pastikan jarak objek daun tidak terlalu dekat agar tidak blur.")

else:
    st.error("Gagal menginisialisasi file bobot model 'best.pt'.")

# 5. FOOTER
st.markdown("<br><br><hr><p style='text-align: center; color: #7A8A80; font-size: 0.85rem;'>© 2026 | SoyLeaf-Guard | Universitas Islam Indonesia</p>", unsafe_allow_html=True)
