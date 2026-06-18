import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# 1. SET CONFIG UTAMA (Layout Wide & Sembunyikan Sidebar)
st.set_page_config(
    page_title="SoyLeaf-Guard | Sistem Pakar Kedelai",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed" 
)

# 2. SUNTIKAN CSS PREMIUM (Navbar Paling Atas, Warna Eksklusif & Elemen Elegan)
st.markdown("""
    <style>
    /* Background Global */
    .stApp {
        background-color: #F8FAF9;
        font-family: 'Inter', sans-serif;
    }
    
    /* MENYEMBUNYIKAN SIDEBAR TOTAL */
    [data-testid="collapsedControl"] { display: none !important; }
    [data-testid="stSidebar"] { display: none !important; }
    
    /* REKAYASA GAYA NAVBAR HORIZONTAL DI PALING ATAS */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: #FFFFFF;
        padding: 8px 16px;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(15, 76, 42, 0.05);
        border: 1px solid #EAEFEA;
        justify-content: center;
        margin-bottom: 30px;
        position: relative;
    }
    
    /* Mengubah Tampilan Tombol Menu Tab */
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        background-color: #F2F6F3;
        border-radius: 8px;
        color: #3D4E44;
        font-weight: 600;
        font-size: 0.95rem;
        padding: 0px 20px;
        transition: all 0.25s ease;
        border: none !important;
    }
    
    /* Efek Saat Menu Disorot (Hover) */
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #E1EBE5;
        color: #0F4C2A;
        transform: translateY(-1px);
    }
    
    /* Efek Saat Menu Aktif Terpilih (Deep Forest Green) */
    .stTabs [aria-selected="true"] {
        background-color: #0F4C2A !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 12px rgba(15, 76, 42, 0.2) ;
    }
    
    /* Hilangkan border bawah bawaan streamlit */
    .stTabs [data-baseweb="tab-border"] { display: none !important; }
    
    /* DESAIN CONTAINER KARTU (CARD LAYOUT) */
    .custom-card {
        background: white;
        padding: 28px;
        border-radius: 16px;
        box-shadow: 0 2px 15px rgba(0, 0, 0, 0.015);
        border: 1px solid #EAEAEA;
        margin-bottom: 20px;
    }
    
    /* DESAIN KARTU PENYAKIT MODERN */
    .disease-card {
        background: #FFFFFF;
        padding: 20px;
        border-radius: 12px;
        border-left: 4px solid #0F4C2A;
        box-shadow: 0 2px 10px rgba(0,0,0,0.01);
        margin-bottom: 15px;
    }
    .disease-title {
        color: #0F4C2A;
        font-weight: 700;
        font-size: 1.15rem;
        margin-bottom: 6px;
    }
    
    /* HERO SECTION TYPOGRAPHY */
    .hero-title {
        color: #0F4C2A;
        font-weight: 800;
        font-size: 2.8rem;
        text-align: center;
        margin-bottom: 4px;
        margin-top: 10px;
    }
    .hero-sub {
        color: #5A6E62;
        font-size: 1.15rem;
        text-align: center;
        margin-bottom: 35px;
    }
    
    /* SECTION SUBHEADER */
    .section-header {
        color: #0F4C2A;
        font-weight: 700;
        font-size: 1.4rem;
        margin-bottom: 15px;
        border-bottom: 2px solid #E1EBE5;
        padding-bottom: 6px;
    }
    
    /* TOMBOL INFERENCE INTERAKTIF */
    div.stButton > button:first-child {
        background-color: #0F4C2A !important;
        color: white !important;
        border-radius: 10px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        border: none !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
    }
    div.stButton > button:first-child:hover {
        background-color: #0B381F !important;
        transform: translateY(-1px);
    }
    
    /* BADGE NOTIFIKASI HASIL */
    .detection-badge {
        background-color: #F4F8F5;
        border-left: 5px solid #0F4C2A;
        padding: 14px;
        border-radius: 8px;
        margin-bottom: 10px;
        font-weight: 600;
        color: #0F4C2A;
        font-size: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# 3. DEKLARASI NAVBAR TAB HORIZONTAL DI PALING ATAS HALAMAN
menu_tab1, menu_tab2, menu_tab3 = st.tabs([
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
    except Exception as e:
        return None

model = load_model()

# Konfigurasi WebRTC STUN Server
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class YoloVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = model
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = self.model(img, conf=0.25)
        return results[0].plot()


if model is not None:
    
    # ==================== MENU TAB 1: HALAMAN UTAMA ====================
    with menu_tab1:
        # Judul Utama di dalam Halaman Utama
        st.markdown('<h1 class="hero-title">SoyLeaf-Guard</h1>', unsafe_allow_html=True)
        st.markdown('<p class="hero-sub">Sistem Komputasi Pakar Identifikasi Dini Penyakit Daun Kedelai Berbasis YOLOv9</p>', unsafe_allow_html=True)
        
        # KARTU A: INFORMASI SISTEM & METRIK
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Parameter & Spesifikasi Teknis Komputasi</div>', unsafe_allow_html=True)
        
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric(label="Akurasi Model (mAP50)", value="97.8%", delta="Optimal")
        with col_m2:
            st.metric(label="Total Dataset Penelitian", value="3.600 Citra", delta="Augmentasi Berhasil")
        with col_m3:
            st.metric(label="Kecepatan Pemrosesan", value="15-30 FPS", delta="WebRTC Terintegrasi")
            
        st.write("")
        st.markdown("""
        Sistem ini dikembangkan sebagai bagian dari penelitian Tugas Akhir Statistika UII yang mengintegrasikan model deep learning 
        YOLOv9 untuk mengenali dan melokalisasi gejala patogen pada daun tanaman kedelai secara otomatis dan efisien.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        # KARTU B: PENJELASAN PENYAKIT DAUN KEDELAI (DI BAWAHNYA)
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Karakteristik Visual Penyakit Daun Kedelai</div>', unsafe_allow_html=True)
        st.write("Klasifikasi penyakit daun kedelai yang dapat diidentifikasi oleh sistem:")
        
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            st.markdown("""
            <div class="disease-card">
                <div class="disease-title">1. Karat Daun (Soybean Rust)</div>
                <p style="color: #4A554F; font-size: 0.95rem; margin-bottom:0;">Disebabkan oleh jamur <i>Phakopsora pachyrhizi</i>. Gejala ditandai dengan bercak pustul kecil berwarna cokelat kelabu atau kemerahan di permukaan bawah daun, menyebabkan daun menguning dan gugur pra-matang.</p>
            </div>
            <div class="disease-card">
                <div class="disease-title">2. Pustul Bakteri (Bacterial Pustule)</div>
                <p style="color: #4A554F; font-size: 0.95rem; margin-bottom:0;">Disebabkan oleh bakteri <i>Xanthomonas axonopodis pv. glycines</i>. Ditandai bintik kecil berwarna kemerahan yang menonjol di bagian tengah, umumnya dikelilingi oleh area kuning (halo) di sekitarnya.</p>
            </div>
            """, unsafe_allow_html=True)
            
        with c2:
            st.markdown("""
            <div class="disease-card">
                <div class="disease-title">3. Embun Bulu (Downy Mildew)</div>
                <p style="color: #4A554F; font-size: 0.95rem; margin-bottom:0;">Disebabkan oleh cendawan <i>Peronospora manshurica</i>. Permukaan atas daun menunjukkan bercak hijau pucat atau kuning, sedangkan permukaan bawah daun ditumbuhi kapang halus berwarna abu-abu.</p>
            </div>
            <div class="disease-card">
                <div class="disease-title">4. Bercak Target (Target Spot)</div>
                <p style="color: #4A554F; font-size: 0.95rem; margin-bottom:0;">Disebabkan oleh jamur <i>Corynespora cassiicola</i>. Gejala berupa bercak cokelat melingkar besar dengan pola lingkaran konsentris berlapis menyerupai sasaran tembak.</p>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("""
        <div class="disease-card" style="border-left: 4px solid #3E7A5A;">
            <div class="disease-title" style="color: #3E7A5A;">5. Daun Sehat (Healthy Leaf)</div>
            <p style="color: #4A554F; font-size: 0.95rem; margin-bottom:0;">Kondisi daun kontrol pembanding dengan permukaan bersih, pigmen hijau merata, dan bebas dari gejala infeksi mikroorganisme.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


    # ==================== MENU TAB 2: DETEKSI UPLOAD ====================
    with menu_tab2:
        st.markdown('<h2 style="color: #0F4C2A; font-weight:700; margin-bottom:20px;">Deteksi via File Gambar</h2>', unsafe_allow_html=True)
        
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Unggah file gambar daun kedelai:", 
            type=["jpg", "jpeg", "png"]
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            col1, col2 = st.columns(2, gap="large")
            
            with col1:
                st.markdown('<div class="custom-card">', unsafe_allow_html=True)
                st.subheader("Gambar Asli")
                st.image(image, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                btn_trigger = st.button("Jalankan Analisis Objek")
            
            with col2:
                st.markdown('<div class="custom-card">', unsafe_allow_html=True)
                st.subheader("Hasil Lokalisasi YOLOv9")
                
                if btn_trigger:
                    img_array = np.array(image)
                    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    
                    results = model(img_bgr)
                    result_img = results[0].plot()
                    result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                    st.image(result_img_rgb, use_container_width=True)
                else:
                    st.info("Silakan klik tombol di atas untuk memproses gambar.")
                st.markdown('</div>', unsafe_allow_html=True)
            
            if btn_trigger:
                st.markdown('<div class="custom-card">', unsafe_allow_html=True)
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
                st.markdown('</div>', unsafe_allow_html=True)


    # ==================== MENU TAB 3: DETEKSI REAL-TIME ====================
    with menu_tab3:
        st.markdown('<h2 style="color: #0F4C2A; font-weight:700; margin-bottom:20px;">Deteksi Kamera Real-Time</h2>', unsafe_allow_html=True)
        
        col_cam1, col_cam2 = st.columns([2, 1], gap="large")
        with col_cam1:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.subheader("Aliran Video WebRTC")
            webrtc_streamer(
                key="yolov9-final-topbar",
                video_transformer_factory=YoloVideoTransformer,
                rtc_configuration=RTC_CONFIGURATION,
                media_stream_constraints={"video": True, "audio": False}
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_cam2:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.subheader("Panduan Teknis")
            st.caption("""
            - Aktifkan fungsi kamera dengan menekan tombol START.
            - Posisikan sampel daun kedelai secara sejajar di depan lensa.
            - Akurasi inferensi real-time dipengaruhi oleh stabilitas perangkat dan intensitas cahaya sekitar.
            """)
            st.markdown('</div>', unsafe_allow_html=True)

else:
    st.error("Gagal menginisialisasi file bobot model 'best.pt'.")

# 5. FOOTER HALAMAN FORMAL
st.markdown(
    """
    <br><br><hr>
    <p style='text-align: center; color: #7A8A80; font-size: 0.85rem;'>
        © 2026 | SoyLeaf-Guard | Jurusan Statistika FMIPA | Universitas Islam Indonesia
    </p>
    """, 
    unsafe_allow_html=True
)
