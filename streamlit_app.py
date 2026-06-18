import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# 1. SET CONFIG UTAMA (Clean & Wide Mode tanpa Sidebar)
st.set_page_config(
    page_title="SoyLeaf-Guard | Sistem Pakar Kedelai",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. SUNTIKAN CSS PREMIUM (Navbar Tab Atas, Kartu Dashboard, & Tipografi Modern)
st.markdown("""
    <style>
    /* Background Global */
    .stApp {
        background-color: #F4F7F5;
        font-family: 'Inter', sans-serif;
    }
    
    /* Sembunyikan Sidebar Bawaan */
    [data-testid="collapsedControl"] {
        display: none !important;
    }
    
    /* DESAIN NAVBAR HORIZONTAL JADI ELEGAN & INTERAKTIF */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: #FFFFFF;
        padding: 12px 24px;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(46, 125, 50, 0.05);
        border: 1px solid #E3EAE4;
        justify-content: center;
        margin-bottom: 35px;
    }
    
    /* Desain Tombol Menu */
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        background-color: #F0F4F1;
        border-radius: 12px;
        color: #4A5D4E;
        font-weight: 600;
        font-size: 0.95rem;
        padding: 0px 22px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border: none !important;
    }
    
    /* Efek Hover Menu */
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #C8E6C9;
        color: #1B5E20;
        transform: translateY(-2px);
    }
    
    /* Efek Saat Menu Aktif Terpilih */
    .stTabs [aria-selected="true"] {
        background-color: #2E7D32 !important;
        color: #FFFFFF !important;
        box-shadow: 0 6px 15px rgba(46, 125, 50, 0.3) !important;
    }
    
    /* Hilangkan border bawah bawaan streamlit */
    .stTabs [data-baseweb="tab-border"] {
        display: none !important;
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
    
    /* DESAIN KARTU PENYAKIT (PADA TAB 2) */
    .disease-card {
        background: #FFFFFF;
        padding: 20px;
        border-radius: 16px;
        border-left: 5px solid #2E7D32;
        box-shadow: 0 4px 15px rgba(0,0,0,0.02);
        margin-bottom: 15px;
    }
    .disease-title {
        color: #1B5E20;
        font-weight: 700;
        font-size: 1.2rem;
        margin-bottom: 5px;
    }
    
    /* HERO SECTION TYPOGRAPHY */
    .hero-title {
        color: #1B5E20;
        font-weight: 800;
        font-size: 3.2rem;
        text-align: center;
        margin-bottom: 0px;
        letter-spacing: -1px;
    }
    .hero-sub {
        color: #556B58;
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 35px;
    }
    
    /* TOMBOL INFERENCE GREEN INTERAKTIF */
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
    
    /* CALL TO ACTION CARD (TAB 5) */
    .cta-container {
        background: linear-gradient(135deg, #1B5E20 0%, #2E7D32 100%);
        color: white;
        padding: 40px;
        border-radius: 24px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(46,125,50,0.2);
    }
    </style>
""", unsafe_allow_html=True)

# 3. HEADER IDENTITAS UTAMA (HERO)
st.markdown('<h1 class="hero-title">🌱 SoyLeaf-Guard</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Sistem Komputasi Pakar Identifikasi Dini Penyakit Daun Kedelai Berbasis <b>YOLOv9</b></p>', unsafe_allow_html=True)

# 4. DEKLARASI 5 HALAMAN INTERAKTIF SESUAI STRUKTUR SKRIPSI
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "ℹ️ 1. Informasi Sistem",
    "🔬 2. Karakteristik Penyakit",
    "📤 3. Deteksi via Upload",
    "🎥 4. Deteksi Real-Time",
    "🚀 5. Kenali Penyakit Sekarang"
])

# 5. LOADING MODEL CORE LOGIC WITH CACHING
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
    
    # ==================== HALAMAN 1: INFORMASI SISTEM ====================
    with tab1:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("📊 Parameter & Spesifikasi Teknis Komputasi")
        
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Akurasi Model (mAP50)", "97.8%", "Performa Optimal")
        with col_m2:
            st.metric("Total Dataset Penelitian", "3.600 Citra", "Augmentasi Sinkron")
        with col_m3:
            st.metric("Kecepatan Pemrosesan", "15-30 FPS", "Teknologi WebRTC")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("🔬 Latar Belakang & Metodologi Aplikasi")
        st.write("""
        Aplikasi ini merupakan produk hilirisasi dari penelitian Tugas Akhir Statistika UII yang bertujuan untuk mempermudah petani 
        maupun penyuluh lapangan dalam mengidentifikasi gejala patogen pada daun kedelai secara instan tanpa batasan waktu.
        
        **Kelebihan Integrasi Sistem:**
        - **Arsitektur Pengujian Back-end:** Divalidasi menggunakan struktur ketat *5-Fold Cross Validation* serta *Retraining* parameter.
        - **Antarmuka Kompatibel:** Menggunakan *framework* Streamlit Cloud tanpa membebani memori lokal perangkat keras milik pengguna (*Client-side efficiency*).
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    # ==================== HALAMAN 2: JENIS PENYAKIT DAUN KEDELAI ====================
    with tab2:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("🔬 Ensiklopedia Gejala Visual Penyakit Daun Kedelai (Dataset Skripsi)")
        st.write("Berikut merupakan 4 jenis penyakit daun kedelai beserta karakteristik daun sehat yang mampu diidentifikasi oleh model kecerdasan buatan YOLOv9s ini:")
        
        # Grid 2 Kolom untuk menampilkan Card Informasi Penyakit
        c1, c2 = st.columns(2, gap="medium")
        
        with c1:
            st.markdown("""
            <div class="disease-card">
                <div class="disease-title">🍂 1. Karat Daun (Soybean Rust)</div>
                <p style="color: #555; font-size: 0.95rem;">Disebabkan oleh jamur <i>Phakopsora pachyrhizi</i>. Gejala ditandai dengan munculnya bintik/pustul kecil berwarna cokelat kelabu atau kemerahan seperti karat di permukaan bawah daun, menyebabkan daun menguning dan gugur sebelum waktunya.</p>
            </div>
            <div class="disease-card">
                <div class="disease-title">🦠 2. Pustul Bakteri (Bacterial Pustule)</div>
                <p style="color: #555; font-size: 0.95rem;">Disebabkan oleh bakteri <i>Xanthomonas axonopodis pv. glycines</i>. Ditandai bintik kecil berwarna kemerahan yang menonjol (pustul) di bagian tengah, biasanya dikelilingi oleh lingkaran kuning halis (halo) terang di sekelilingnya.</p>
            </div>
            """, unsafe_allow_html=True)
            
        with c2:
            st.markdown("""
            <div class="disease-card">
                <div class="disease-title">☁️ 3. Embun Bulu (Downy Mildew)</div>
                <p style="color: #555; font-size: 0.95rem;">Disebabkan oleh cendawan <i>Peronospora manshurica</i>. Permukaan atas daun memperlihatkan bercak hijau pucat atau kuning kelabu, sedangkan pada permukaan bawah daun tumbuh kumpulan benang jamur halus menyerupai bulu berwarna abu-abu keunguan.</p>
            </div>
            <div class="disease-card">
                <div class="disease-title">🎯 4. Bercak Target (Target Spot)</div>
                <p style="color: #555; font-size: 0.95rem;">Disebabkan oleh jamur <i>Corynespora cassiicola</i>. Gejala berupa bercak cokelat melingkar besar yang menyerupai papan sasaran tembak (mempunyai lingkaran-linggaran konsentris yang berlapis di area bercaknya).</p>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("""
        <div class="disease-card" style="border-left: 5px solid #4CAF50;">
            <div class="disease-title" style="color: #2E7D32;">🌿 5. Daun Sehat (Healthy Soybean Leaf)</div>
            <p style="color: #555; font-size: 0.95rem;">Kondisi daun pembanding dengan permukaan yang bersih, berwarna hijau segar homogen, bertekstur mulus, serta bebas dari segala bentuk bercak, nekrosis, maupun serangan mikroorganisme merugikan.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ==================== HALAMAN 3: DETEKSI UPLOAD ====================
    with tab3:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("📁 Panel Unggah Citra Daun")
        uploaded_file = st.file_uploader(
            "Pilih file gambar dengan format JPG, JPEG, atau PNG:", 
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
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
                    st.info("Silakan tekan tombol hijau di bawah kolom Gambar Asli untuk memulai proses.")
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

    # ==================== HALAMAN 4: DETEKSI REAL-TIME ====================
    with tab4:
        col_cam1, col_cam2 = st.columns([2, 1], gap="large")
        
        with col_cam1:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.subheader("🎥 Pengambilan Citra Berbasis Streaming WebRTC")
            
            webrtc_streamer(
                key="yolov9-5page-detection",
                video_transformer_factory=YoloVideoTransformer,
                rtc_configuration=RTC_CONFIGURATION,
                media_stream_constraints={"video": True, "audio": False}
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_cam2:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.subheader("💡 Petunjuk Lapangan")
            st.caption("""
            - Pastikan telah memberikan izin akses kamera pada jendela pop-up penjelajah web (*browser*).
            - Atur sudut pencahayaan daun agar bercak patogen terlihat kontras.
            - Fluktuasi performa komputasi di lapangan (rata-rata nilai keandalan 65%) dipengaruhi oleh keterbatasan resolusi video dan *motion blur*.
            """)
            st.markdown('</div>', unsafe_allow_html=True)

    # ==================== HALAMAN 5: PERINTAH SEGERA KENALI PENYAKIT ====================
    with tab5:
        st.markdown("""
        <div class="cta-container">
            <h2 style='color: white; font-weight: 700; margin-bottom: 10px;'>📢 PERINTAH SEGERA IDENTIFIKASI!</h2>
            <p style='font-size: 1.15rem; max-width: 800px; margin: 0 auto 25px auto; opacity: 0.9;'>
                Jangan biarkan infeksi patogen merusak produktivitas komoditas lahan kedelai Anda. Lakukan langkah preventif penanganan dini secara cepat dan presisi menggunakan modul kecerdasan buatan komputer.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="custom-card" style="margin-top: 25px;">', unsafe_allow_html=True)
        st.subheader("🛠️ Langkah Cepat Mulai Pemeriksaan Sekarang:")
        
        col_step1, col_step2 = st.columns(2)
        with col_step1:
            st.info("""
            **Metode Statis (Sangat Direkomendasikan untuk Akurasi Tinggi):**
            1. Pindahkan tab menu di atas ke **3. Deteksi via Upload**.
            2. Masukkan berkas foto daun kedelai yang Anda ambil lewat ponsel pintar.
            3. Tekan tombol **Mulai Analisis Deteksi** untuk melihat letak kotak *bounding box*.
            """)
        with col_step2:
            st.success("""
            **Metode Dinamis (Praktis & Instan Langsung di Area Lahan):**
            1. Pindahkan tab menu di atas ke **4. Deteksi Real-Time**.
            2. Klik tombol **START** hitam pada jendela pemutar media.
            3. Dekatkan daun kedelai di depan kamera perangkat untuk pemindaian otomatis.
            """)
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.error("Gagal memuat arsitektur model 'best.pt'. Periksa direktori repositori file aplikasi Anda.")

# 6. FOOTER HALAMAN PREMIUM
st.markdown(
    """
    <br><br>
    <p style='text-align: center; color: #888888; font-size: 0.85rem;'>
        © 2026 | SoyLeaf-Guard | Jurusan Statistika FMIPA | Universitas Islam Indonesia
    </p>
    """, 
    unsafe_allow_html=True
)
