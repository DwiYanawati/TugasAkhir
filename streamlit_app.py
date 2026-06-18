import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# 1. SET CONFIG UTAMA (Premium Layout - Wide Mode)
st.set_page_config(
    page_title="SoyLeaf-Guard | Sistem Pakar Kedelai",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded" 
)

# 2. SUNTIKAN CSS MODERN & INTERAKTIF (Custom Dashboard UI)
st.markdown("""
    <style>
    /* Background Global */
    .stApp {
        background-color: #F4F7F5;
        font-family: 'Inter', sans-serif;
    }
    
    /* Desain Sidebar yang Minimalis & Clean */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E3EAE4;
    }
    [data-testid="stSidebar"] h2 {
        color: #1B5E20 !important;
        font-weight: 700;
        font-size: 1.3rem;
        margin-bottom: 20px;
    }
    
    /* Radio Button Custom Styling di Sidebar */
    div[data-testid="stSidebarUserContent"] .stRadio > div {
        gap: 10px;
    }
    
    /* DESAIN CONTAINER KARTU (CARD LAYOUT) */
    .custom-card {
        background: white;
        padding: 30px;
        border-radius: 24px;
        box-shadow: 0 4px 25px rgba(0, 0, 0, 0.02);
        border: 1px solid #EAEAEA;
        margin-bottom: 25px;
    }
    
    /* DESAIN KARTU PENYAKIT */
    .disease-card {
        background: #FFFFFF;
        padding: 22px;
        border-radius: 16px;
        border-left: 5px solid #2E7D32;
        box-shadow: 0 4px 15px rgba(0,0,0,0.01);
        margin-bottom: 18px;
        transition: all 0.3s ease;
    }
    .disease-card:hover {
        transform: translateX(5px);
        box-shadow: 0 6px 20px rgba(46,125,50,0.08);
    }
    .disease-title {
        color: #1B5E20;
        font-weight: 700;
        font-size: 1.25rem;
        margin-bottom: 8px;
    }
    
    /* HERO SECTION TYPOGRAPHY */
    .hero-title {
        color: #1B5E20;
        font-weight: 800;
        font-size: 3.4rem;
        text-align: center;
        margin-bottom: 5px;
        letter-spacing: -1px;
    }
    .hero-sub {
        color: #556B58;
        font-size: 1.25rem;
        text-align: center;
        margin-bottom: 40px;
    }
    
    /* SECTION SUBHEADER */
    .section-header {
        color: #1B5E20;
        font-weight: 700;
        font-size: 1.6rem;
        margin-top: 10px;
        margin-bottom: 20px;
        border-bottom: 2px solid #C8E6C9;
        padding-bottom: 8px;
    }
    
    /* TOMBOL INFERENCE INTERAKTIF */
    div.stButton > button:first-child {
        background-color: #2E7D32 !important;
        color: white !important;
        border-radius: 14px !important;
        padding: 14px 28px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        border: none !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
        box-shadow: 0 4px 12px rgba(46, 125, 50, 0.2) !important;
    }
    div.stButton > button:first-child:hover {
        background-color: #1B5E20 !important;
        box-shadow: 0 6px 20px rgba(46, 125, 50, 0.35) !important;
        transform: translateY(-2px);
    }
    
    /* BADGE NOTIFIKASI HASIL */
    .detection-badge {
        background-color: #F1F8F2;
        border-left: 6px solid #2E7D32;
        padding: 16px;
        border-radius: 12px;
        margin-bottom: 12px;
        font-weight: 600;
        color: #1B5E20;
        font-size: 1.05rem;
    }
    </style>
""", unsafe_allow_html=True)

# 3. CORE MODEL LOADING WITH CACHING
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


# 4. SIDEBAR SEBAGAI NAVIGATOR UTAMA (Sangat Bersih & Elegan)
with st.sidebar:
    st.markdown("## 🧭 Menu Navigasi")
    app_mode = st.radio(
        "Pilih Layanan Sistem:",
        ["🏠 Halaman Utama (Informasi & Edukasi)", "📤 Fitur Deteksi via Upload", "🎥 Fitur Deteksi Real-Time"]
    )
    st.markdown("---")
    if model is not None:
        st.success("✅ Model YOLOv9s Aktif")
    else:
        st.error("❌ Model Gagal Dimuat")


# ==================== MODE A: HALAMAN UTAMA ====================
if app_mode == "🏠 Halaman Utama (Informasi & Edukasi)":
    # Hero Title Tengah Halaman
    st.markdown('<h1 class="hero-title">🌱 SoyLeaf-Guard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Sistem Komputasi Cerdas Identifikasi Dini Penyakit Daun Kedelai Berbasis <b>YOLOv9</b></p>', unsafe_allow_html=True)
    
    # KARTU 1: INFORMASI SISTEM & METRIK
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">📊 Parameter & Spesifikasi Teknis Komputasi</div>', unsafe_allow_html=True)
    
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric(label="Akurasi Model (mAP50)", value="97.8%", delta="Performa Optimal")
    with col_m2:
        st.metric(label="Total Dataset Penelitian", value="3.600 Citra", delta="Augmentasi Sinkron")
    with col_m3:
        st.metric(label="Kecepatan Pemrosesan", value="15-30 FPS", delta="Teknologi WebRTC")
        
    st.write("")
    st.markdown("""
    Aplikasi ini merupakan produk hilirisasi dari penelitian Tugas Akhir Statistika UII yang bertujuan untuk mempermudah petani 
    maupun penyuluh lapangan dalam mengidentifikasi gejala patogen pada daun kedelai secara instan tanpa batasan waktu.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # KARTU 2: PENJELASAN JENIS PENYAKIT DAUN KEDELAI (DI BAWAHNYA)
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">🔬 Ensiklopedia Gejala Visual Penyakit Daun Kedelai</div>', unsafe_allow_html=True)
    st.write("Berikut merupakan 4 jenis penyakit daun kedelai beserta karakteristik daun sehat yang mampu diidentifikasi oleh model kecerdasan buatan YOLOv9s ini:")
    
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown("""
        <div class="disease-card">
            <div class="disease-title">🍂 1. Karat Daun (Soybean Rust)</div>
            <p style="color: #555; font-size: 0.95rem; margin-bottom:0;">Disebabkan oleh jamur <i>Phakopsora pachyrhizi</i>. Gejala ditandai dengan munculnya bintik/pustul kecil berwarna cokelat kelabu atau kemerahan seperti karat di permukaan bawah daun, menyebabkan daun menguning dan gugur sebelum waktunya.</p>
        </div>
        <div class="disease-card">
            <div class="disease-title">🦠 2. Pustul Bakteri (Bacterial Pustule)</div>
            <p style="color: #555; font-size: 0.95rem; margin-bottom:0;">Disebabkan oleh bakteri <i>Xanthomonas axonopodis pv. glycines</i>. Ditandai bintik kecil berwarna kemerahan yang menonjol (pustul) di bagian tengah, biasanya dikelilingi oleh lingkaran kuning halus (halo) terang di sekelilingnya.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.markdown("""
        <div class="disease-card">
            <div class="disease-title">☁️ 3. Embun Bulu (Downy Mildew)</div>
            <p style="color: #555; font-size: 0.95rem; margin-bottom:0;">Disebabkan oleh cendawan <i>Peronospora manshurica</i>. Permukaan atas daun memperlihatkan bercak hijau pucat atau kuning kelabu, sedangkan pada permukaan bawah daun tumbuh kumpulan benang jamur halus menyerupai bulu berwarna abu-abu keunguan.</p>
        </div>
        <div class="disease-card">
            <div class="disease-title">🎯 4. Bercak Target (Target Spot)</div>
            <p style="color: #555; font-size: 0.95rem; margin-bottom:0;">Disebabkan oleh jamur <i>Corynespora cassiicola</i>. Gejala berupa bercak cokelat melingkar besar yang menyerupai papan sasaran tembak (mempunyai lingkaran-linggaran konsentris yang berlapis di area bercaknya).</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("""
    <div class="disease-card" style="border-left: 5px solid #4CAF50;">
        <div class="disease-title" style="color: #2E7D32;">🌿 5. Daun Sehat (Healthy Soybean Leaf)</div>
        <p style="color: #555; font-size: 0.95rem; margin-bottom:0;">Kondisi daun pembanding dengan permukaan yang bersih, berwarna hijau segar homogen, bertekstur mulus, serta bebas dari segala bentuk bercak, nekrosis, maupun serangan mikroorganisme merugikan.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ==================== MODE B: DETEKSI UPLOAD ====================
elif app_mode == "📤 Fitur Deteksi via Upload":
    st.markdown('<h1 class="hero-title">📤 Deteksi via File Gambar</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Metode Analisis Statis Citra Daun Kedelai</p>', unsafe_allow_html=True)
    
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Pilih file gambar dengan format JPG, JPEG, atau PNG:", 
        type=["jpg", "jpeg", "png"]
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.subheader("📷 Gambar Asli Pengguna")
            st.image(image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            btn_trigger = st.button("🚀 Mulai Analisis Deteksi")
        
        with col2:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.subheader("🎯 Hasil Lokalisasi Objek (YOLOv9s)")
            
            if btn_trigger:
                img_array = np.array(image)
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                
                results = model(img_bgr)
                result_img = results[0].plot()
                result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                st.image(result_img_rgb, use_container_width=True)
            else:
                st.info("Silakan tekan tombol 'Mulai Analisis Deteksi' di bawah kolom Gambar Asli.")
            st.markdown('</div>', unsafe_allow_html=True)
        
        if btn_trigger:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.subheader("📊 Analisis Tingkat Keparahan & Klasifikasi")
            if len(results[0].boxes) > 0:
                for i, box in enumerate(results[0].boxes):
                    class_id = int(box.cls[0])
                    class_name = results[0].names[class_id]
                    confidence = float(box.conf[0])
                    st.markdown(
                        f'<div class="detection-badge">Klasifikasi #{i+1} : <b>{class_name}</b> — Confidence Score: <b>{(confidence*100):.2f}%</b></div>', 
                        unsafe_allow_html=True
                    )
            else:
                st.warning("Model tidak mendeteksi gejala penyakit makroskopis pada sampel citra ini.")
            st.markdown('</div>', unsafe_allow_html=True)


# ==================== MODE C: DETEKSI REAL-TIME ====================
else:
    st.markdown('<h1 class="hero-title">🎥 Deteksi Kamera Real-Time</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Metode Analisis Dinamis Berbasis Aliran Video WebRTC</p>', unsafe_allow_html=True)
    
    col_cam1, col_cam2 = st.columns([2, 1], gap="large")
    with col_cam1:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("🎥 Video Stream")
        webrtc_streamer(
            key="yolov9-hybrid-detection",
            video_transformer_factory=YoloVideoTransformer,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False}
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_cam2:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("💡 Petunjuk Lapangan")
        st.caption("""
        - Izinkan browser untuk mengakses kamera perangkat Anda (*Allow Camera*).
        - Atur fokus daun kedelai secara tenang agar kotak pembatas tidak bergetar berlebihan.
        - Perubahan pencahayaan eksternal dan kecepatan gerak objek berpengaruh langsung pada fluktuasi skor akurasi model di lapangan.
        """)
        st.markdown('</div>', unsafe_allow_html=True)


# 5. FOOTER HALAMAN PREMIUM
st.markdown(
    """
    <br><br><hr>
    <p style='text-align: center; color: #888888; font-size: 0.85rem;'>
        © 2026 | SoyLeaf-Guard | Jurusan Statistika FMIPA | Universitas Islam Indonesia
    </p>
    """, 
    unsafe_allow_html=True
)
