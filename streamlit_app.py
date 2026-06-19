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

# 2. SUNTIKAN CSS PREMIUM (Kombinasi Warna Modern & Pembatasan Gambar Supaya Tidak Kegedean)
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

    .info-text {
        color: #2F3E3A;
        line-height: 1.7;
        font-size: 1.05rem;
        text-align: justify;
    }

    /* KOTAK TIMBUL PER PENYAKIT (VERTIKAL KE BAWAH) */
    .disease-vertical-box {
        background-color: #FFFFFF;
        border-radius: 16px;
        box-shadow: 0 10px 25px rgba(31, 111, 95, 0.08);
        border: 1px solid #E2EFEA;
        padding: 24px;
        margin-bottom: 35px;
        text-align: center;
        transition: transform 0.3s ease;
    }
    
    .disease-vertical-box:hover {
        transform: translateY(-3px);
    }

    /* WADAH FOTO LANDSCAPE SERAGAM - TIDAK KEGEDEAN */
    .image-landscape-wrapper {
        max-width: 480px; /* Membatasi lebar gambar biar ga kegedean jir */
        height: 270px;    /* Mengunci tinggi gambar proporsional 16:9 */
        margin: 0 auto 20px auto; /* Ketengah dan beri jarak ke bawah */
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border: 2px solid #EEF5F2;
    }
    
    .image-landscape-wrapper img {
        width: 100% !important;
        height: 100% !important;
        object-fit: cover !important; /* Potong gambar otomatis biar ga gepeng */
    }

    .disease-box-title {
        color: #1F6F5F;
        font-weight: 700;
        font-size: 1.35rem;
        margin-bottom: 12px;
    }

    .disease-box-desc {
        color: #2F3E3A;
        font-size: 1rem;
        line-height: 1.6;
        text-align: justify;
        max-width: 800px;
        margin: 0 auto;
        padding: 10px 15px;
        background-color: #FAFCFB;
        border-radius: 8px;
    }
    
    /* GAYA VISUAL LANGKAH PENGGUNAAN (STEP BOX) */
    .step-container {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 18px;
        border: 1px solid #E2EFEA;
        border-left: 4px solid #1F6F5F;
        margin-bottom: 25px;
    }
    
    .step-title {
        color: #1F6F5F;
        font-weight: 700;
        font-size: 1.1rem;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 8px;
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
        
        # 1. Kasus Penyakit Daun Kedelai
        st.markdown('<div class="section-header">Kasus & Urgensi Penyakit Daun Kedelai</div>', unsafe_allow_html=True)
        st.markdown("""
        <p class="info-text">
        Tanaman kedelai (<i>Glycine max L.</i>) merupakan salah satu komoditas pangan strategis yang memiliki peran penting dalam pemenuhan 
        kebutuhan protein nabati masyarakat. Namun, dalam proses budidayanya, produktivitas tanaman seringkali mengalami penurunan signifikan 
        akibat serangan berbagai patogen penyakit yang menyerang organ daun. Infeksi makroskopis seperti bercak, karat, hingga kerusakan jaringan 
        klorofil dapat menurunkan laju fotosintesis, menghambat pengisian polong, hingga memicu risiko gagal panen masal.
        </p>
        <p class="info-text">
        Metode identifikasi konvensional yang mengandalkan pengamatan visual manual oleh petani seringkali subjektif dan membutuhkan waktu yang 
        cukup lama, sehingga penanganan sering terlambat dilakukan. Oleh karena itu, pendekatan berbasis teknologi komputasi cerdas deep learning 
        objek deteksi dirancang dalam penelitian ini untuk mengenali ragam spasial dan melokalisasi tingkat serangan penyakit secara instan, 
        presisi, dan efisien.
        </p>
        """, unsafe_allow_html=True)

        # 2. Jenis Penyakit Daun Kedelai (Kotak Timbul, Foto Landscape Terkontrol, Teks Masuk Kotak)
        st.markdown('<div class="section-header">Jenis Penyakit Daun Kedelai</div>', unsafe_allow_html=True)
        
        diseases = [
            {
                "title": "1. Karat Daun (Soybean Rust)",
                "desc": "Disebabkan oleh infeksi jamur patogen Phakopsora pachyrhizi. Gejala awal ditandai dengan munculnya bercak pustul kecil berwarna cokelat kelabu atau kemerahan di permukaan bawah daun, mengakibatkan klorosis jaringan sekitar hingga daun gugur pra-matang.",
                "filename": "karatdaun.jpg"
            },
            {
                "title": "2. Pustul Bakteri (Bacterial Pustule)",
                "desc": "Disebabkan oleh agen infeksi bakteri Xanthomonas axonopodis pv. glycines. Karakteristik visual dicirikan oleh bintik kecil berwarna kemerahan yang mengalami elevasi menonjol di bagian tengah, umumnya dikelilingi oleh cincin kuning halus (halo) di sekeliling area infeksi.",
                "filename": "pustulbakteri.jpg"
            },
            {
                "title": "3. Embun Bulu (Downy Mildew)",
                "desc": "Disebabkan oleh cendawan oomycete Peronospora manshurica. Permukaan atas helaian daun memperlihatkan sebaran bercak hijau pucat atau kuning kelabu, sedangkan pada area permukaan bawah daun ditumbuhi oleh kumpulan massa konidia halus berwarna abu-abu keunguan.",
                "filename": "embunbulu.jpg"
            },
            {
                "title": "4. Bercak Target (Target Spot)",
                "desc": "Disebabkan oleh jamur nekrotrofik Corynespora cassiicola. Gejala ditandai dengan pembentukan lesi atau bercak cokelat melingkar berdiameter besar yang menampilkan pola struktur lingkaran konsentris berlapis menyerupai bentuk papan sasaran tembak.",
                "filename": "bercaktarget.jpg"
            },
            {
                "title": "5. Daun Sehat (Healthy Leaf)",
                "desc": "Kondisi kontrol pembanding dimana organ daun memiliki pigmen klorofil hijau merata yang homogen, bertekstur mulus, serta bersih sepenuhnya dari segala jenis bentuk nekrosis, klorosis, maupun degradasi akibat serangan organisme pengganggu tanaman.",
                "filename": "healthy.jpg"
            }
        ]

        # Loop Berjejer ke Bawah secara Bersih
        for d in diseases:
            st.markdown('<div class="disease-vertical-box">', unsafe_allow_html=True)
            
            # Membungkus gambar kedalam wrapper CSS pembatas ukuran
            st.markdown('<div class="image-landscape-wrapper">', unsafe_allow_html=True)
            if os.path.exists(d["filename"]):
                st.image(d["filename"], use_container_width=True)
            else:
                st.error(f"File '{d['filename']}' tidak ditemukan.")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Teks Judul dan Deskripsi Berada di dalam kotak yang sama
            st.markdown(f'<div class="disease-box-title">{d["title"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="disease-box-desc">{d["desc"]}</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

    # ==================== TAB 2: DETEKSI UPLOAD ====================
    with tab_upload:
        st.markdown('<div class="section-header">Deteksi via File Gambar</div>', unsafe_allow_html=True)
        
        # Penambahan Panduan Langkah Penggunaan Berjejer Samping (3 Kolom)
        st.markdown('<div class="step-title">📋 Langkah-Langkah Penggunaan Panduan Analisis:</div>', unsafe_allow_html=True)
        step_cols = st.columns(3)
        with step_cols[0]:
            st.markdown('<div class="step-container"><b>1. Unggah Citra Daun</b><br><span style="font-size:0.85rem; color:#555;">Klik area drag-and-drop di bawah untuk memasukkan file foto daun kedelai (Format PNG/JPG/JPEG).</span></div>', unsafe_allow_html=True)
        with step_cols[1]:
            st.markdown('<div class="step-container"><b>2. Jalankan Inferensi</b><br><span style="font-size:0.85rem; color:#555;">Klik tombol hijau "Jalankan Analisis Objek" untuk memicu pemindaian koordinat model YOLOv9.</span></div>', unsafe_allow_html=True)
        with step_cols[2]:
            st.markdown('<div class="step-container"><b>3. Evaluasi Prediksi</b><br><span style="font-size:0.85rem; color:#555;">Sistem memunculkan kotak lokalisasi (*bounding box*) dan nilai tingkat akurasi presentase patogen.</span></div>', unsafe_allow_html=True)
            
        uploaded_file = st.file_uploader("Pilih file gambar daun kedelai:", type=["jpg", "jpeg", "png"])
        
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
        
        # Penambahan Panduan Langkah Penggunaan Real-Time
        st.markdown('<div class="step-title">📋 Langkah-Langkah Penggunaan Deteksi Kamera:</div>', unsafe_allow_html=True)
        step_cam_cols = st.columns(3)
        with step_cam_cols[0]:
            st.markdown('<div class="step-container"><b>1. Izinkan Akses Kamera</b><br><span style="font-size:0.85rem; color:#555;">Berikan izin browser untuk mengakses modul hardware kamera laptop / webcam eksternal Anda.</span></div>', unsafe_allow_html=True)
        with step_cam_cols[1]:
            st.markdown('<div class="step-container"><b>2. Jalankan Tombol Start</b><br><span style="font-size:0.85rem; color:#555;">Klik tombol "START" pada jendela WebRTC untuk mengaktifkan pemindaian streaming video beruntun.</span></div>', unsafe_allow_html=True)
        with step_cam_cols[2]:
            st.markdown('<div class="step-container"><b>3. Dekatkan Daun</b><br><span style="font-size:0.85rem; color:#555;">Arahkan sampel fisik daun kedelai ke depan lensa kamera. Model akan melakukan inferensi otomatis.</span></div>', unsafe_allow_html=True)
            
        col_cam1, col_cam2 = st.columns([2, 1], gap="large")
        with col_cam1:
            st.subheader("Aliran Video WebRTC")
            webrtc_streamer(
                key="yolov9-clean-final",
                video_transformer_factory=YoloVideoTransformer,
                rtc_configuration=RTC_CONFIGURATION,
                media_stream_constraints={"video": True, "audio": False}
            )
            
        with col_cam2:
            st.subheader("Catatan Teknis Stabilitas")
            st.caption("""
            - Akurasi deteksi objek *real-time* sangat bergantung pada kondisi pencahayaan ruangan dan kestabilan fokus lensa.
            - Pastikan jarak objek daun tidak terlalu dekat agar piksel warna tidak mengalami distorsi *blur*.
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
