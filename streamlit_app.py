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

# 2. SUNTIKAN CSS PREMIUM (Kombinasi Warna 1F6F5F, 2FA084, 6FCF97)
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
    
    /* Efek Hover Menu (Menggunakan Warna 6FCF97) */
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #6FCF97;
        color: #1F6F5F;
        transform: translateY(-1px);
    }
    
    /* Menu Aktif Terpilih (Menggunakan Warna 1F6F5F) */
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
    
    /* Subjudul Bagian (Menggunakan Warna 2FA084) */
    .section-header {
        color: #2FA084;
        font-weight: 700;
        font-size: 1.6rem;
        margin-top: 35px;
        margin-bottom: 20px;
        padding-left: 5px;
    }

    .info-text {
        color: #2F3E3A;
        line-height: 1.7;
        font-size: 1.05rem;
    }
    .info-text ul {
        margin-top: 8px;
        padding-left: 20px;
    }
    .info-text li {
        margin-bottom: 6px;
    }

    /* KOTAK TIMBUL DENGAN BAYANGAN HALUS (CARD DISEASES) */
    .disease-card-box {
        background-color: #FFFFFF;
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 10px 25px rgba(31, 111, 95, 0.07); /* Efek Timbul Bayangan */
        border: 1px solid #E2EFEA;
        border-left: 5px solid #2FA084; /* Aksen warna tepi kiri */
        margin-bottom: 30px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .disease-card-box:hover {
        transform: translateY(-3px); /* Efek melayang saat kursor di atas kotak */
        box-shadow: 0 14px 30px rgba(31, 111, 95, 0.12);
    }
    .disease-title {
        color: #1F6F5F;
        font-weight: 700;
        font-size: 1.35rem;
        margin-bottom: 12px;
    }
    
    /* Tombol Eksekusi Deteksi */
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
        box-shadow: 0 6px 15px rgba(31, 111, 95, 0.3) !important;
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

# 3. DEKLARASI NAVBAR HORIZONTAL ATAS WEBSITE
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

# Konfigurasi WebRTC STUN Server
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
        
        # Bagian A: Informasi Sistem
        st.markdown('<div class="section-header">Informasi Sistem</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-text">
        Menu informasi sistem berfungsi untuk memberikan penjelasan kepada pengguna mengenai sistem yang digunakan. 
        Informasi yang ditampilkan pada menu ini meliputi:
        <ul>
            <li><b>Model yang digunakan:</b> YOLOv9.</li>
            <li><b>Framework pengembangan:</b> Streamlit.</li>
            <li><b>Fitur utama sistem:</b> Deteksi berbasis unggah gambar dan kamera real-time.</li>
            <li><b>Tentang aplikasi:</b> SoyLeaf-Guard dirancang untuk membantu identifikasi penyakit daun kedelai secara instan guna mendukung produktivitas pertanian.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        # Bagian B: Karakteristik Penyakit (Kotak Timbul + Link Gambar Lokal)
        st.markdown('<div class="section-header">Karakteristik Visual Penyakit Daun Kedelai</div>', unsafe_allow_html=True)
        
        # Struktur Data Pemanggilan File Gambar Lokal Anda
        diseases = [
            {
                "title": "1. Karat Daun (Soybean Rust)",
                "desc": "Disebabkan oleh jamur <i>Phakopsora pachyrhizi</i>. Gejala ditandai dengan bercak pustul kecil berwarna cokelat kelabu atau kemerahan di permukaan bawah daun, menyebabkan daun menguning dan gugur pra-matang.",
                "filename": "karat.jpg"
            },
            {
                "title": "2. Pustul Bakteri (Bacterial Pustule)",
                "desc": "Disebabkan oleh bakteri <i>Xanthomonas axonopodis pv. glycines</i>. Ditandai bintik kecil berwarna kemerahan yang menonjol di bagian tengah, umumnya dikelilingi oleh area kuning (halo) di sekitarnya.",
                "filename": "pustul.jpg"
            },
            {
                "title": "3. Embun Bulu (Downy Mildew)",
                "desc": "Disebabkan oleh cendawan <i>Peronospora manshurica</i>. Permukaan atas daun menunjukkan bercak hijau pucat atau kuning, sedangkan permukaan bawah daun ditumbuhi kapang halus berwarna abu-abu.",
                "filename": "embun.jpg"
            },
            {
                "title": "4. Bercak Target (Target Spot)",
                "desc": "Disebabkan oleh jamur <i>Corynespora cassiicola</i>. Gejala berupa bercak cokelat melingkar besar dengan pola lingkaran konsentris berlapis menyerupai sasaran tembak.",
                "filename": "bercak.jpg"
            },
            {
                "title": "5. Daun Sehat (Healthy Leaf)",
                "desc": "Kondisi daun kontrol pembanding dengan permukaan bersih, pigmen hijau merata, dan bebas dari gejala infeksi patogen maupun serangan mikroorganisme merugikan.",
                "filename": "sehat.jpg"
            }
        ]

        # Menampilkan per penyakit ke dalam Kotak Timbul berpola Kolom Horizontal
        for d in diseases:
            st.markdown('<div class="disease-card-box">', unsafe_allow_html=True)
            col_img, col_txt = st.columns([1, 2.5], gap="large")
            
            with col_img:
                # Cek ketersediaan file gambar lokal di folder berkas Anda
                if os.path.exists(d["filename"]):
                    st.image(d["filename"], use_container_width=True)
                else:
                    # Menampilkan area penampung rapi jika berkas gambar belum dimasukkan ke folder
                    st.warning(f"File '{d['filename']}' tidak ditemukan di direktori.")
                    
            with col_txt:
                st.markdown(f'<div class="disease-title">{d["title"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<p class="info-text" style="margin:0;">{d["desc"]}</p>', unsafe_allow_html=True)
                
            st.markdown('</div>', unsafe_allow_html=True)

    # ==================== TAB 2: DETEKSI UPLOAD ====================
    with tab_upload:
        st.markdown('<div class="section-header">Deteksi via File Gambar</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Unggah file gambar daun kedelai:", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            col1, col2 = st.columns(2, gap="large")
            
            with col1:
                st.subheader("Gambar Asli")
                st.image(image, use_container_width=True)
                btn_trigger = st.button("Jalankan Analisis Objek")
            
            with col2:
                st.subheader("Hasil Lokalisasi YOLOv9")
                if btn_trigger:
                    img_array = np.array(image)
                    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    results = model(img_bgr)
                    result_img = results[0].plot()
                    result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                    st.image(result_img_rgb, use_container_width=True)
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
                        st.markdown(
                            f'<div class="detection-badge">Objek #{i+1} : <b>{class_name}</b> — Confidence Score: <b>{(confidence*100):.2f}%</b></div>', 
                            unsafe_allow_html=True
                        )
                else:
                    st.warning("Model tidak mendeteksi adanya gejala penyakit pada sampel daun ini.")

    # ==================== TAB 3: DETEKSI REAL-TIME ====================
    with tab_realtime:
        st.markdown('<div class="section-header">Deteksi Kamera Real-Time</div>', unsafe_allow_html=True)
        
        col_cam1, col_cam2 = st.columns([2, 1], gap="large")
        with col_cam1:
            st.subheader("Aliran Video WebRTC")
            webrtc_streamer(
                key="yolov9-forest-theme",
                video_transformer_factory=YoloVideoTransformer,
                rtc_configuration=RTC_CONFIGURATION,
                media_stream_constraints={"video": True, "audio": False}
            )
            
        with col_cam2:
            st.subheader("Panduan Teknis")
            st.caption("""
            - Aktifkan fungsi kamera dengan menekan tombol START.
            - Posisikan sampel daun kedelai secara sejajar di depan lensa.
            - Akurasi inferensi real-time dipengaruhi oleh stabilitas perangkat dan intensitas cahaya sekitar.
            """)

else:
    st.error("Gagal menginisialisasi file bobot model 'best.pt'.")

# 5. FOOTER HALAMAN FORMAL SCIENTIFIC
st.markdown(
    """
    <br><br><hr>
    <p style='text-align: center; color: #7A8A80; font-size: 0.85rem;'>
        © 2026 | SoyLeaf-Guard | Jurusan Statistika FMIPA | Universitas Islam Indonesia
    </p>
    """, 
    unsafe_allow_html=True
)
