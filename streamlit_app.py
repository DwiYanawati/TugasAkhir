import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from ultralytics import YOLO
import time

# 1. SET CONFIG UTAMA (Taruh ini di baris paling atas setelah import!)
st.set_page_config(
    page_title="SoyLeaf-Guard | Deteksi Penyakit Kedelai",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. SUNTIKAN CSS UNTUK TAMPILAN AESTHETIC
st.markdown("""
    <style>
    /* Mengubah font utama dan background */
    .stApp {
        background-color: #F8F9FA;
        font-family: 'Inter', sans-serif;
    }
    
    /* Mempercantik Sidebar */
    [data-testid="stSidebar"] {
        background-color: #E8F5E9;
        border-right: 1px solid #C8E6C9;
    }
    
    /* Desain Kartu (Card Layout) untuk Informasi & Hasil */
    .custom-card {
        background: white;
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        border: 1px solid #E0E0E0;
        margin-bottom: 20px;
    }
    
    /* Judul Utama yang Eye-catching */
    .main-title {
        color: #1B5E20;
        font-weight: 800;
        font-size: 2.5rem;
        margin-bottom: 5px;
    }
    .sub-title {
        color: #666666;
        font-size: 1.1rem;
        margin-bottom: 30px;
    }
    
    /* Desain Tombol Deteksi yang Modern */
    div.stButton > button:first-child {
        background-color: #2E7D32;
        color: white;
        border-radius: 12px;
        padding: 10px 24px;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
        width: 100%;
        box-shadow: 0 4px 10px rgba(46, 125, 50, 0.2);
    }
    div.stButton > button:first-child:hover {
        background-color: #1B5E20;
        box-shadow: 0 6px 15px rgba(46, 125, 50, 0.3);
        transform: translateY(-2px);
    }
    </style>
""", unsafe_allow_html=True)

# 3. CONTOH IMPLEMENTASI STRUKTUR HALAMAN UTAMA
st.markdown('<h1 class="main-title">🌱 SoyLeaf-Guard</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Sistem Cerdas Identifikasi Dini Penyakit Tanaman Kedelai (YOLOv9s)</p>', unsafe_allow_html=True)

# Contoh membagi Layout jadi 2 kolom seimbang (Kiri: Input, Kanan: Hasil)
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.subheader("📁 Unggah Citra Daun")
    # File uploader Anda ditaruh di sini
    uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "png", "jpeg"])
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.subheader("📊 Hasil Analisis Model")
    # Tampilan output visualisasi bounding box ditaruh di sini
    st.info("Menunggu gambar diunggah untuk memulai inferensi...")
    st.markdown('</div>', unsafe_allow_html=True)


