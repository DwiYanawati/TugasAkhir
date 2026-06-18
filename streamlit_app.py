import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# 1. SET CONFIG UTAMA (Aesthetic setup - Wide Mode)
st.set_page_config(
    page_title="SoyLeaf-Guard | Deteksi Penyakit Kedelai",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed" # Menyembunyikan sidebar bawaan secara default
)

# 2. SUNTIKAN CSS PREMIUM (Menghilangkan Sidebar & Merombak Tab Menjadi Navbar Atas)
st.markdown("""
    <style>
    /* Global Background */
    .stApp {
        background-color: #F4F7F5;
        font-family: 'Inter', sans-serif;
    }
    
    /* MENYEMBUNYIKAN SIDEBAR TOTAL (Tombol panah kiri atas dihilangkan) */
    [data-testid="collapsedControl"] {
        display: none !important;
    }
    
    /* REKAYASA GAYA TAB MENJADI NAVBAR PREMIUM */
    .stTabs [data-baseweb="tab-list"] {
        gap: 15px;
        background-color: #FFFFFF;
        padding: 12px 20px;
        border-radius: 16px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.03);
        border: 1px solid #E3EAE4;
        justify-content: center; /* Membuat menu berada tepat di tengah halaman */
        margin-bottom: 30px;
    }
    
    /* Mengubah Tampilan Tombol Tab */
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F0F4F1;
        border-radius: 10px;
        color: #4A5D4E;
        font-weight: 600;
        font-size: 1rem;
        padding: 0px 25px;
        transition: all 0.3s ease;
        border: none !important;
    }
    
    /* Efek Saat Tab Disorot (Hover) */
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #C8E6C9;
        color: #1B5E20;
        transform: translateY(-2px);
    }
    
    /* Efek Saat Tab Aktif Terpilih */
    .stTabs [aria-selected="true"] {
        background-color: #2E7D32 !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 12px rgba(46, 125, 50, 0.25);
    }
    
    /* Garis bawah bawaan tab dihilangkan */
    .stTabs [data-baseweb="tab-border"] {
        display: none !important;
    }
    
    /* Desain Kartu Kontainer (Card Layout) */
    .custom-card {
        background: white;
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 4px 25px rgba(0, 0, 0, 0.03);
        border: 1px solid #EAEAEA;
        margin-bottom: 25px;
    }
    
    /* Tipografi Judul */
    .hero-title {
        color: #1B5E20;
        font-weight: 800;
        font-size: 3rem;
        text-align: center;
        margin-bottom: 0px;
    }
    .hero-sub {
        color: #556B58;
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 35px;
    }
    
    /* Tombol Unggah & Deteksi */
    div.stButton > button:first-child {
        background-color: #2E7D32 !important;
        color: white !important;
        border-radius: 12px !important;
        padding: 14px 28px !important;
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
    
    /* Hasil Badge */
    .detection-badge {
        background-color: #F1F8F2;
        border-left: 6px solid #2E7D32;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 12px;
        font-weight: 600;
        color: #1B5E20;
        font-size: 1.05rem;
    }
    </style>
""", unsafe_allow_html=True)

# 3. HERO SECTION (Header Atas Tengah)
st.markdown('<h1 class="hero-title">🌱 SoyLeaf-Guard</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Sistem Komputasi Cerdas Identifikasi Penyakit Daun Kedelai Berbasis <b>YOLOv9</b></p>', unsafe_allow_html=True)

# 4. MEMBUAT TABS HORIZONTAL DI TENGAH SEBAGAI PENGGANTI SIDEBAR
menu_tab1, menu_tab2, menu_tab3 = st.tabs([
    "📤  Upload File Citra", 
    "🎥  Kamera Real-Time", 
    "ℹ️  Informasi Sistem & Parameter"
])

# 5. LOAD MODEL DENGAN CACHING (Logika Utama Anda)
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
    
    # ==================== HALAMAN TAB 1: UPLOAD GAMBAR ====================
    with menu_tab1:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("📁 Unggah Gambar Daun Kedelai")
        uploaded_file = st.file_uploader(
            "Seret atau pilih file gambar (.jpg, .jpeg, .png):", 
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed" # Menyembunyikan teks label bawaan agar clean
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            # Membagi layout visualisasi gambar dengan proporsi kolom seimbang
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
                    st.info("Menunggu perintah analisis... Silakan tekan tombol hijau di bawah Gambar Asli.")
                st.markdown('</div>', unsafe_allow_html=True)
            
            if btn_trigger:
                # Panel Detail Karakteristik Penyakit
                st.markdown('<div class="custom-card">', unsafe_allow_html=True)
                st.subheader("📊 Analisis Tingkat Keparahan & Klasifikasi")
                
                if len(results[0].boxes) > 0:
                    for i, box in enumerate(results[0].boxes):
                        class_id = int(box.cls[0])
                        class_name = results[0].names[class_id]
                        confidence = float(box.conf[0])
                        st.markdown(
                            f'<div class="detection-badge">Klasifikasi #{i+1} : <b>{class_name}</b> — Tingkat Kepercayaan (Confidence): <b>{(confidence*100):.2f}%</b></div>', 
                            unsafe_allow_html=True
                        )
                else:
                    st.warning("Model tidak mendeteksi adanya bercak kelainan penyakit visual pada sampel daun ini.")
                st.markdown('</div>', unsafe_allow_html=True)

    # ==================== HALAMAN TAB 2: KAMERA REAL-TIME ====================
    with menu_tab2:
        col_cam1, col_cam2 = st.columns([2, 1], gap="large")
        
        with col_cam1:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.subheader("🎥 Pengambilan Citra Berbasis Streaming WebRTC")
            st.write("Arahkan kamera perangkat langsung ke sampel daun kedelai untuk analisis otomatis.")
            
            webrtc_streamer(
                key="yolov9-navbar-detection",
                video_transformer_factory=YoloVideoTransformer,
                rtc_configuration=RTC_CONFIGURATION,
                media_stream_constraints={"video": True, "audio": False}
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_cam2:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.subheader("💡 Petunjuk Lapangan")
            st.caption("""
            - Pastikan izin akses kamera pada browser sudah disetujui (*Allow*).
            - Jaga stabilitas jarak kamera dengan objek daun sekitar 15-30 cm.
            - Fluktuasi nilai kestabilan deteksi real-time dipengaruhi oleh intensitas cahaya matahari dan *motion blur*.
            """)
            st.markdown('</div>', unsafe_allow_html=True)

    # ==================== HALAMAN TAB 3: INFORMASI SISTEM ====================
    with menu_tab3:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("📊 Parameter Evaluasi Akhir Penelitian")
        
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Akurasi Model (mAP50)", "97.8%", "Sangat Unggul")
        with col_m2:
            st.metric("Total Dataset", "3.600 Citra", "Setelah Augmentasi")
        with col_m3:
            st.metric("Kecepatan Komputasi", "15-30 FPS", "Real-Time Tracking")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("🔬 Ringkasan Metodologi Komputasi")
        st.write("""
        Aplikasi ini merupakan bagian akhir dari implementasi Tugas Akhir Statistika UII yang mengintegrasikan model deep learning **YOLOv9s** ke dalam arsitektur sistem berbasis web.
        
        **Spesifikasi Teknis:**
        - **Model Back-end:** Ultralytics YOLOv9s (`best.pt`)
        - **Arsitektur Validasi:** *5-Fold Cross Validation & Retraining*
        - **Kelas Target:** *Karat Daun, Pustul Bakteri, Embun Bulu, Bercak Target,* dan *Daun Sehat*.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.error("Gagal memuat arsitektur model 'best.pt'. Periksa repositori Anda.")

# 6. FOOTER PREMIUM
st.markdown(
    """
    <br><br>
    <p style='text-align: center; color: #888888; font-size: 0.85rem;'>
        © 2026 | SoyLeaf-Guard Kedelai | Jurusan Statistika FMIPA | Universitas Islam Indonesia
    </p>
    """, 
    unsafe_allow_html=True
)
