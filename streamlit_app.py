import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# 1. SET CONFIG UTAMA (Aesthetic setup)
st.set_page_config(
    page_title="SoyLeaf-Guard | Deteksi Penyakit Kedelai",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. SUNTIKAN CSS UNTUK TAMPILAN MODERN & INTERAKTIF
st.markdown("""
    <style>
    /* Global Styles */
    .stApp {
        background-color: #F8F9FA;
        font-family: 'Inter', sans-serif;
    }
    
    /* Mempercantik Sidebar */
    [data-testid="stSidebar"] {
        background-color: #E8F5E9;
        border-right: 1px solid #C8E6C9;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #1B5E20 !important;
        font-weight: 700;
    }
    
    /* Desain Kartu Kontainer (Card Layout) */
    .custom-card {
        background: white;
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04);
        border: 1px solid #E0E0E0;
        margin-bottom: 25px;
    }
    
    /* Judul Utama */
    .main-title {
        color: #1B5E20;
        font-weight: 800;
        font-size: 2.6rem;
        margin-bottom: 0px;
        padding-top: 10px;
    }
    .sub-title {
        color: #555555;
        font-size: 1.1rem;
        margin-bottom: 25px;
    }
    
    /* Tombol Utama yang Interaktif */
    div.stButton > button:first-child {
        background-color: #2E7D32 !important;
        color: white !important;
        border-radius: 12px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        border: none !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
        box-shadow: 0 4px 10px rgba(46, 125, 50, 0.2) !important;
    }
    div.stButton > button:first-child:hover {
        background-color: #1B5E20 !important;
        box-shadow: 0 6px 15px rgba(46, 125, 50, 0.3) !important;
        transform: translateY(-2px);
    }
    
    /* Badge Hasil Deteksi yang Modern */
    .detection-badge {
        background-color: #E8F5E9;
        border-left: 5px solid #2E7D32;
        padding: 12px 18px;
        border-radius: 8px;
        margin-bottom: 10px;
        font-weight: 600;
        color: #1B5E20;
    }
    .no-detection-badge {
        background-color: #FFF3E0;
        border-left: 5px solid #FF9800;
        padding: 12px 18px;
        border-radius: 8px;
        font-weight: 600;
        color: #E65100;
    }
    </style>
""", unsafe_allow_html=True)

# 3. HEADER UTAMA
st.markdown('<h1 class="main-title">🌱 SoyLeaf-Guard</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Sistem Cerdas Deteksi Dini Penyakit Daun Kedelai Berbasis <b>YOLOv9</b></p>', unsafe_allow_html=True)

# 4. LOAD MODEL DENGAN CACHING
@st.cache_resource
def load_model():
    try:
        model = YOLO("best.pt")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

with st.spinner("Menginisialisasi Arsitektur YOLOv9..."):
    model = load_model()

# Konfigurasi WebRTC STUN Server (Agar kamera jalan di server cloud online)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Class Transformer untuk memproses Video Frame demi Frame secara Real-Time
class YoloVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = model

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Proses deteksi frame lewat YOLOv9
        results = self.model(img, conf=0.25) # Nilai thresholds bisa disesuaikan
        annotated_frame = results[0].plot()
        
        return annotated_frame

# 5. SIDEBAR MENU (Sekarang ada 3 Pilihan Menu Sesuai Skripsi!)
with st.sidebar:
    st.header("📌 Menu Navigasi")
    menu = st.radio(
        "Pilih Mode Aplikasi:", 
        ["📤 Upload Gambar", "🎥 Kamera Real-Time", "ℹ️ Informasi Sistem"], 
        index=0
    )
    
    st.markdown("---")
    st.subheader("💡 Panduan Pengguna")
    if menu == "📤 Upload Gambar":
        st.info("Unggah foto daun kedelai (.jpg/.png), lalu klik tombol '🔍 Deteksi Penyakit'.")
    elif menu == "🎥 Kamera Real-Time":
        st.info("Klik tombol 'START' untuk mengaktifkan kamera perangkat. Dekatkan daun kedelai ke arah lensa kamera.")
    else:
        st.info("Halaman ini memuat informasi parameter komputasi data penelitian.")

if model is not None:
    st.sidebar.success("✅ Model YOLOv9 Berhasil Dimuat!")
    
    # ==================== MODE 1: UPLOAD GAMBAR ====================
    if menu == "📤 Upload Gambar":
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("📤 Panel Unggah Citra Daun")
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
                st.subheader("📷 Gambar Asli")
                st.image(image, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                btn_trigger = st.button("🔍 Deteksi Penyakit")
            
            if btn_trigger:
                with st.spinner("Model sedang mendeteksi karakteristik visual daun..."):
                    img_array = np.array(image)
                    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    
                    results = model(img_bgr)
                    result_img = results[0].plot()
                    result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                    
                    with col2:
                        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
                        st.subheader("🎯 Hasil Analisis Sistem")
                        st.image(result_img_rgb, use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
                    st.subheader("📊 Detail Objek Terdeteksi (Instance-level)")
                    
                    if len(results[0].boxes) > 0:
                        for i, box in enumerate(results[0].boxes):
                            class_id = int(box.cls[0])
                            class_name = results[0].names[class_id]
                            confidence = float(box.conf[0])
                            st.markdown(
                                f'<div class="detection-badge">Objek #{i+1} : {class_name} — Confidence Score: {(confidence*100):.2f}%</div>', 
                                unsafe_allow_html=True
                            )
                    else:
                        st.markdown(
                            '<div class="no-detection-badge">⚠️ Sistem Tidak Mendeteksi Adanya Gejala Penyakit (Daun Sehat/Objek Tidak Dikenali)</div>', 
                            unsafe_allow_html=True
                        )
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                with col2:
                    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
                    st.subheader("🎯 Hasil Analisis Sistem")
                    st.info("Silakan klik tombol 'Deteksi Penyakit' untuk melihat hasil.")
                    st.markdown('</div>', unsafe_allow_html=True)
    
    # ==================== MODE 2: KAMERA REAL-TIME (SUDAH ADA!) ====================
    elif menu == "🎥 Kamera Real-Time":
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("🎥 Deteksi Real-Time via WebRTC")
        st.write("Fitur ini menggunakan kamera perangkat Anda untuk melakukan inferensi secara langsung (*on-the-fly*).")
        
        # Integrasi komponen streaming video WebRTC Streamlit
        webrtc_streamer(
            key="yolov9-detection",
            video_transformer_factory=YoloVideoTransformer,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False}
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.caption("**Catatan Teknis Lapangan:** Deteksi mode *real-time* memiliki fluktuasi *confidence score* (rata-rata 65%) akibat faktor pergerakan (*motion blur*), *frame rate* kamera, dan pencahayaan alami di area lahan pertanian.")
        st.markdown('</div>', unsafe_allow_html=True)
        
    # ==================== MODE 3: INFORMASI SISTEM ====================
    else:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("ℹ️ Spesifikasi Teknis & Informasi Sistem")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Arsitektur Model", "YOLOv9s", "Object Detection")
        with col2:
            st.metric("Framework Web", "Streamlit Cloud", "Python-based")
        with col3:
            st.metric("Metode Deteksi", "Upload & WebRTC", "Otomatis")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("🔬 Tentang Penelitian")
        st.write("""
        Aplikasi ini dikembangkan untuk mendeteksi 4 jenis penyakit utama pada daun tanaman kedelai 
        (*Karat Daun, Pustul Bakteri, Embun Bulu, Bercak Target*) serta mampu mengidentifikasi kondisi daun sehat sebagai pembanding.
        
        **Karakteristik Pelatihan Model:**
        - **Dataset Awal:** 1.500 citra (dikembangkan menjadi 3.600 citra melalui metode augmentasi).
        - **Konfigurasi:** 30 *Epoch*, *Imgsize* 640, *Batch size* 16.
        - **Validasi Kestabilan:** *5-Fold Cross Validation*.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.error("❌ Gagal memuat model. Pastikan file 'best.pt' berada pada direktori repositori aplikasi Anda.")

# 6. FOOTER HALAMAN
st.markdown(
    """
    <br><hr>
    <p style='text-align: center; color: #777777; font-size: 0.9rem;'>
        © 2026 | Aplikasi Deteksi Dini Penyakit Daun Kedelai | Jurusan Statistika FMIPA | Universitas Islam Indonesia
    </p>
    """, 
    unsafe_allow_html=True
)
