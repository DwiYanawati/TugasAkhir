import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import base64
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# 1. SET CONFIG UTAMA
st.set_page_config(
    page_title="SoyLeaf-Guard",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed" 
)

# Fungsi Helper untuk mengubah gambar lokal menjadi Base64 agar bisa dibaca HTML Streamlit
def get_image_base64(path):
    if os.path.exists(path):
        with open(path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    return None

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

    /* KOTAK PUTIH PANJANG PREMIUM TEMPAT PENYAKIT */
    .disease-container-box {
        background-color: #FFFFFF;
        border-radius: 16px;
        box-shadow: 0 4px 15px rgba(31, 111, 95, 0.05);
        border: 1px solid #E2EFEA;
        padding: 20px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 25px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .disease-container-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 22px rgba(31, 111, 95, 0.1);
    }
    
    .disease-img-wrapper {
        flex: 1;
        min-width: 160px;
        max-width: 220px;
    }

    .disease-img-wrapper img {
        width: 100%;
        height: auto;
        border-radius: 10px;
        object-fit: cover;
        display: block;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }

    .disease-text-wrapper {
        flex: 4;
    }

    .disease-box-title {
        color: #1F6F5F;
        font-weight: 700;
        font-size: 1.3rem;
        margin-bottom: 8px;
        margin-top: 0px;
    }

    .disease-box-desc {
        color: #2F3E3A;
        font-size: 0.98rem;
        line-height: 1.6;
        text-align: justify;
        margin: 0px;
    }
    
    /* GAYA VISUAL LANGKAH PENGGUNAAN */
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
    "Deteksi Upload Gambar", 
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
        return self.model(img, conf=0.25)[0].plot(boxes=True)


if model is not None:
    # ==================== TAB 1: HALAMAN UTAMA ====================
    with tab_home:
        st.markdown('<h1 class="hero-title">🌿SoyLeaf-Guard</h1>', unsafe_allow_html=True)
        st.markdown('<p class="hero-sub">Sistem Identifikasi Dini Penyakit Daun Kedelai</p>', unsafe_allow_html=True)
        
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

        st.markdown('<div class="section-header">Jenis Penyakit Daun Kedelai</div>', unsafe_allow_html=True)

        # 2. Data Jenis Penyakit Daun Kedelai
        diseases = [
            {
                "title": "1. Penyakit Karat Daun (Phakopsora pachyrhizi)",
                "desc": "Merupakan penyakit jamur terpenting pada tanaman kedelai, khususnya saat musim kemarau, yang mampu memicu risiko kehilangan hasil panen sebesar 30–60% serta menurunkan kualitas biji. Jamur ini berkembang dengan membentuk struktur uredinia (berwarna cokelat tua) dan telia (berwarna lebih gelap) pada helaian jaringan daun.",
                "filename": "karatdaun.jpg"
            },
            {
                "title": "2. Penyakit Pustul Bakteri (Xanthomonas axonopodis pv. glycines)",
                "desc": "Penyakit infeksi bakteri yang menjadi faktor utama penurunan produktivitas kedelai nasional. Gejalanya dicirikan oleh munculnya bintik kecil hijau pucat di kedua sisi daun, yang kemudian berkembang menjadi benjolan (pustul) cokelat muda di permukaan bawah daun. Pada stadium lanjut, bercak nekrotik ini menyatu, mudah robek oleh angin hingga daun berlubang dan gugur.",
                "filename": "pustulbakteri.jpg"
            },
            {
                "title": "3. Penyakit Embun Bulu (Peronospora manshurica)",
                "desc": "Penyakit endemik global yang di Indonesia tersebar luas di Pulau Jawa dan Sumatera dengan potensi kerugian mencapai 6–18%. Gejala awal diidentifikasi melalui munculnya bercak kuning kehujauan pada permukaan atas helaian daun, yang perlahan bertransisi menjadi kelabu hingga cokelat tua serta mengganggu integritas mutu benih.",
                "filename": "embunbulu.jpg"
            },
            {
                "title": "4. Penyakit Bercak Target (Corynespora cassiicola)",
                "desc": "Disebabkan oleh jamur yang menyerang sejak fase awal pertumbuhan vegetatif hingga pembentukan polong. Memiliki karakteristik lesi bercak cokelat kemerahan berdiameter 10–15 mm yang membentuk zonasi melingkar konsentris menyerupai papan target. Patogen tangguh ini bersifat kosmopolitan dan mampu bertahan hidup di sisa tanaman maupun tanah selama lebih dari dua tahun.",
                "filename": "bercaktarget.jpg"
            },
            {
                "title": "5. Daun Sehat (Healthy Leaf) - Kontrol Pembanding",
                "desc": "Kondisi kontrol pembanding dimana organ vegetatif daun kedelai memiliki pigmen klorofil hijau merata yang homogen, berstruktur mulus, serta bersih sepenuhnya dari tanda-tanda nekrosis, klorosis, bintik pustul, maupun kerusakan jaringan akibat paparan patogen.",
                "filename": "healthy.jpg"
            }
        ]

        # PROSES PEMBUNGKUSAN TOTAL (Gambar base64 + Teks di dalam satu div)
        for d in diseases:
            img_base64 = get_image_base64(d["filename"])
            
            if img_base64:
                # Jika gambar ditemukan, render langsung dengan string base64 data URI
                img_tag = f"<img src='data:image/jpeg;base64,{img_base64}' alt='{d['title']}' />"
            else:
                # Fallback jika file gambar tidak ditemukan di root folder proyek Anda
                img_tag = "<div style='background:#EEF5F2; padding:30px 10px; border-radius:10px; color:#1F6F5F; font-weight:bold; text-align:center; font-size:0.9rem;'>File Gambar<br>Tidak Ketemu</div>"
            
            disease_html = f"""
            <div class="disease-container-box">
                <div class="disease-img-wrapper">
                    {img_tag}
                </div>
                <div class="disease-text-wrapper">
                    <h3 class="disease-box-title">{d["title"]}</h3>
                    <p class="disease-box-desc">{d["desc"]}</p>
                </div>
            </div>
            """
            st.markdown(disease_html, unsafe_allow_html=True)

    # ==================== TAB 2: DETEKSI UPLOAD ====================
    with tab_upload:
        st.markdown('<div class="section-header">Deteksi Upload Gambar</div>', unsafe_allow_html=True)
        st.markdown('<div class="single-step-box"><div class="step-title-text">Langkah Penggunaan:</div><ul class="step-list"><li><b>Unggah Citra Daun:</b> Klik area berkas di bawah untuk memasukkan foto daun kedelai (Format: JPG, PNG).</li><li><b>Jalankan Inferensi:</b> Klik tombol hijau <b>"Deteksi Penyakit Daun Kedelai"</b> untuk memicu pemindaian model kecerdasan buatan YOLOv9.</li><li><b>Evaluasi Prediksi:</b> Tinjau area kanan untuk melihat hasil lokalisasi bercak serta akurasi presentase (Confidence Score).</li></ul></div>', unsafe_allow_html=True)
            
        uploaded_file = st.file_uploader("Pilih file gambar daun kedelai:", type=["jpg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            col1, col2 = st.columns(2, gap="large")
            
            with col1:
                st.subheader("Gambar Asli")
                st.image(image, use_container_width=True)
                btn_trigger = st.button("Deteksi Penyakit Daun")
            
            with col2:
                st.subheader("Hasil Deteksi Penyakit Daun")
                if btn_trigger:
                    img_array = np.array(image)
                    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    results = model(img_bgr)
                    result_img = results[0].plot(boxes=True)
                    result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                    st.image(result_img_rgb, use_container_width=True)
                else:
                    st.info("Silakan klik tombol di samping untuk memproses gambar.")
            
            if btn_trigger:
                st.write("")
                st.subheader("Detail Klasifikasi & Tingkat Kepercayaan")
                if len(results[0].boxes) > 0:
                    for i, box in enumerate(results[0].boxes):
                        class_id = int(box.cls[0])
                        class_name = results[0].names[class_id]
                        confidence = float(box.conf[0])
                        st.markdown(
                            f'<div class="detection-badge">Jenis Penyakit #{i+1} : <b>{class_name}</b> — Confidence Score: <b>{(confidence*100):.2f}%</b></div>', 
                            unsafe_allow_html=True
                        )
                else:
                    st.warning("Model tidak mendeteksi adanya gejala penyakit pada sampel daun ini.")

    # ==================== TAB 3: DETEKSI REAL-TIME ====================
    with tab_realtime:
        st.markdown('<div class="section-header">Deteksi Kamera Real-Time</div>', unsafe_allow_html=True)
        st.markdown('<div class="single-step-box"><div class="step-title-text">Langkah Penggunaan:</div><ul class="step-list"><li><b>Izinkan Akses Perangkat:</b> Pastikan browser diberikan izin penuh untuk mengakses hardware webcam laptop Anda.</li><li><b>Mulai Deteksi:</b> Tekan tombol <b>"START"</b> pada jendela pemutar WebRTC untuk mengaktifkan pemindaian video langsung.</li><li><b>Deteksi Otomatis:</b> Hadapkan sisi fisik daun kedelai secara sejajar di depan lensa kamera hingga deteksi otomatis muncul tanpa kotak pembatas.</li></ul></div>', unsafe_allow_html=True)
            
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
            - Akurasi deteksi objek real-time sangat bergantung pada kondisi pencahayaan ruangan dan kestabilan fokus lensa.
            - Pastikan jarak objek daun tidak terlalu dekat agar piksel warna tidak mengalami distorsi blur.
            """)

else:
    st.error("Gagal menginisialisasi file bobot model 'best.pt'.")

# 5. FOOTER HALAMAN FORMAL SCIENTIFIC
st.markdown(
    """
    <br><br><hr>
    <p style='text-align: center; color: #7A8A80; font-size: 0.85rem;'>
        © 2026 | SoyLeaf-Guard | Universitas Islam Indonesia
    </p>
    """, 
    unsafe_allow_html=True
)
