import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# 1. SET CONFIG UTAMA
st.set_page_config(
    page_title="SoyLeaf-Guard | UII",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed" 
)

# 2. SUNTIKAN CSS PREMIUM (Forest Green Theme)
st.markdown("""
    <style>
    .stApp {
        background-color: #F8FAF9;
        font-family: 'Inter', sans-serif;
    }
    
    /* Navbar Atas */
    [data-testid="collapsedControl"] { display: none !important; }
    [data-testid="stSidebar"] { display: none !important; }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 15px;
        background-color: #FFFFFF;
        padding: 10px 20px;
        border-radius: 12px;
        box-shadow: 0 2px 15px rgba(0,0,0,0.05);
        justify-content: center;
        margin-bottom: 40px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        background-color: #F2F6F3;
        border-radius: 8px;
        color: #3D4E44;
        font-weight: 600;
        padding: 0px 25px;
        border: none !important;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #0F4C2A !important;
        color: #FFFFFF !important;
    }

    .stTabs [data-baseweb="tab-border"] { display: none !important; }

    /* Judul & Teks */
    .hero-title {
        color: #0F4C2A;
        font-weight: 800;
        font-size: 3rem;
        text-align: center;
        margin-top: 10px;
    }
    .hero-sub {
        color: #5A6E62;
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 40px;
    }
    
    .section-header {
        color: #0F4C2A;
        font-weight: 700;
        font-size: 1.6rem;
        margin-top: 30px;
        margin-bottom: 15px;
    }

    .info-text {
        color: #3D4E44;
        line-height: 1.7;
        font-size: 1.05rem;
    }

    /* Layout Penyakit */
    .disease-box {
        margin-bottom: 40px;
        padding-bottom: 20px;
        border-bottom: 1px solid #E1EBE5;
    }
    .disease-title {
        color: #0F4C2A;
        font-weight: 700;
        font-size: 1.3rem;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# 3. NAVBAR
tab_home, tab_upload, tab_realtime = st.tabs([
    "Halaman Utama", 
    "Deteksi via Upload", 
    "Deteksi Real-Time"
])

# 4. LOAD MODEL
@st.cache_resource
def load_model():
    try:
        model = YOLO("best.pt")
        return model
    except:
        return None

model = load_model()

# Config WebRTC
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

class YoloVideoTransformer(VideoTransformerBase):
    def __init__(self): self.model = model
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        return self.model(img, conf=0.25)[0].plot()

if model is not None:
    # ==================== TAB 1: HALAMAN UTAMA ====================
    with tab_home:
        st.markdown('<h1 class="hero-title">SoyLeaf-Guard</h1>', unsafe_allow_html=True)
        st.markdown('<p class="hero-sub">Sistem Komputasi Pakar Identifikasi Dini Penyakit Daun Kedelai</p>', unsafe_allow_html=True)
        
        # Bagian 1: Informasi Sistem
        st.markdown('<div class="section-header">Informasi Sistem</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-text">
        Menu informasi sistem berfungsi untuk memberikan penjelasan kepada pengguna mengenai sistem yang digunakan. 
        Informasi yang ditampilkan meliputi:
        <ul>
            <li><b>Model yang digunakan:</b> YOLOv9 (You Only Look Once versi 9).</li>
            <li><b>Framework pengembangan:</b> Streamlit (Python-based web framework).</li>
            <li><b>Fitur utama sistem:</b> Deteksi berbasis unggah gambar dan kamera real-time.</li>
            <li><b>Tentang aplikasi:</b> SoyLeaf-Guard dirancang untuk membantu identifikasi penyakit daun kedelai secara instan guna mendukung produktivitas pertanian.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        # Bagian 2: Karakteristik Penyakit (Dengan Gambar)
        st.markdown('<div class="section-header">Karakteristik Visual Penyakit Daun Kedelai</div>', unsafe_allow_html=True)
        
        # List Penyakit untuk dilooping
        diseases = [
            {
                "title": "1. Karat Daun (Soybean Rust)",
                "desc": "Disebabkan oleh jamur <i>Phakopsora pachyrhizi</i>. Gejala ditandai dengan bercak pustul kecil berwarna cokelat kelabu atau kemerahan di permukaan bawah daun, menyebabkan daun menguning dan gugur.",
                "img": "https://pustaka.setjen.pertanian.go.id/media/k2/items/cache/8d94e21a8d8e1d5a7d6e6a1e2f3d4c5b_L.jpg" # Ganti dengan path lokal jika ada
            },
            {
                "title": "2. Pustul Bakteri (Bacterial Pustule)",
                "desc": "Disebabkan oleh bakteri <i>Xanthomonas axonopodis pv. glycines</i>. Gejala berupa bintik kecil kemerahan yang menonjol (pustul) dan dikelilingi oleh warna kuning (halo).",
                "img": "https://pest.ceris.purdue.edu/images/pest/soybean_bacterial_pustule.jpg" # Ganti path
            },
            {
                "title": "3. Embun Bulu (Downy Mildew)",
                "desc": "Disebabkan oleh cendawan <i>Peronospora manshurica</i>. Menghasilkan bercak hijau pucat di bagian atas daun dan kumpulan bulu halus abu-abu di bagian bawah daun.",
                "img": "https://cropprotectionnetwork.org/api/v1/images/2237" # Ganti path
            },
            {
                "title": "4. Bercak Target (Target Spot)",
                "desc": "Disebabkan oleh jamur <i>Corynespora cassiicola</i>. Gejala berupa bercak cokelat melingkar dengan pola lingkaran konsentris menyerupai sasaran tembak.",
                "img": "https://soybeandiseases.cropsci.illinois.edu/wp-content/uploads/2015/07/Target-spot-leaf-symptoms.jpg" # Ganti path
            },
            {
                "title": "5. Daun Sehat (Healthy Leaf)",
                "desc": "Kondisi kontrol dimana daun memiliki warna hijau merata, tekstur mulus, dan tidak menunjukkan adanya bercak atau gejala nekrosis.",
                "img": "https://img.freepik.com/premium-photo/healthy-soybean-leaf-soy-leaves-field-farm-background_166116-4314.jpg" # Ganti path
            }
        ]

        for d in diseases:
            col_img, col_txt = st.columns([1, 2])
            with col_img:
                st.image(d['img'], use_container_width=True)
            with col_txt:
                st.markdown(f'<div class="disease-title">{d["title"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<p class="info-text">{d["desc"]}</p>', unsafe_allow_html=True)
            st.markdown('<div class="disease-box"></div>', unsafe_allow_html=True)

    # ==================== TAB 2: DETEKSI UPLOAD ====================
    with tab_upload:
        st.markdown('<div class="section-header">Deteksi via File Gambar</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Pilih gambar daun kedelai:", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Gambar Asli")
                st.image(image, use_container_width=True)
                btn = st.button("Jalankan Deteksi")
            with c2:
                st.subheader("Hasil YOLOv9")
                if btn:
                    res = model(np.array(image))
                    st.image(cv2.cvtColor(res[0].plot(), cv2.COLOR_BGR2RGB), use_container_width=True)
                    for box in res[0].boxes:
                        st.markdown(f'<div class="detection-badge">{res[0].names[int(box.cls[0])]} - Confidence: {float(box.conf[0])*100:.1f}%</div>', unsafe_allow_html=True)

    # ==================== TAB 3: DETEKSI REAL-TIME ====================
    with tab_realtime:
        st.markdown('<div class="section-header">Deteksi Kamera Real-Time</div>', unsafe_allow_html=True)
        webrtc_streamer(key="yolov9", video_transformer_factory=YoloVideoTransformer, rtc_configuration=RTC_CONFIGURATION)

else:
    st.error("Model best.pt tidak ditemukan.")

# 5. FOOTER
st.markdown("<br><hr><p style='text-align: center; color: #7A8A80; font-size: 0.8rem;'>© 2026 | SoyLeaf-Guard | FMIPA Universitas Islam Indonesia</p>", unsafe_allow_html=True)
