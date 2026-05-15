import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Steam Matchmaker | Kelompok 12",
    page_icon="🕹️",
    layout="wide"
)

# --- 2. INJEKSI CSS KUSTOM (DARK MODE, SIDEBAR, & FONT) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');

    /* Warna Dasar Aplikasi */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }

    /* CSS UNTUK SIDEBAR: Mengubah warna putih menjadi Biru Gelap Steam */
    [data-testid="stSidebar"] {
        background-color: #161b22 !important;
        border-right: 1px solid #30363d;
    }
    
    /* Menyesuaikan warna teks dan label di sidebar agar kontras */
    [data-testid="stSidebar"] .stMarkdown, 
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] .stSelectbox div {
        color: #58a6ff !important;
    }

    /* Font Arcade untuk Header */
    .arcade-font {
        font-family: 'Press Start 2P', cursive;
        color: #58a6ff;
        text-align: center;
        text-shadow: 2px 2px #ff7b72;
        padding: 20px;
        font-size: 2.5rem;
    }

    /* Card Hasil Rekomendasi */
    .game-card {
        background-color: #161b22;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        border: 1px solid #30363d;
        transition: transform 0.2s;
    }
    
    .game-card:hover {
        transform: scale(1.01);
        border-color: #58a6ff;
    }

    .match-percent {
        color: #3fb950;
        font-weight: bold;
        font-size: 1.1em;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. LOGIKA DATA & ENGINE ---
@st.cache_resource(show_spinner=False)
def load_assets():
    # Load dataset utama
    df = pd.read_pickle('steam_games_clean.pkl')
    
    # Perbaikan Performa: Log Transformation pada Popularitas (Score Weight)
    # Ini menangani data yang skewed agar game besar tidak terlalu mendominasi
    df['norm_popularity'] = np.log1p(df['score_weight'])
    df['norm_popularity'] = (df['norm_popularity'] - df['norm_popularity'].min()) / \
                             (df['norm_popularity'].max() - df['norm_popularity'].min())
    
    # Load data mentah untuk deskripsi lengkap
    try:
        df_raw = pd.read_parquet('data.parquet')
        df_raw = df_raw[['name', 'short_description', 'genres']].drop_duplicates('name')
    except:
        df_raw = pd.DataFrame()
        
    # Load TF-IDF Matrix
    with open('tfidf_matrix.pkl', 'rb') as f:
        tfidf_matrix = pickle.load(f)
        
    return df, df_raw, tfidf_matrix

# Inisialisasi Data
with st.empty():
    st.markdown("<h3 style='text-align: center; color: #58a6ff;'>SYSTEM BOOTING...</h3>", unsafe_allow_html=True)
    df, df_raw, tfidf_matrix = load_assets()
    st.empty()

def get_recommendations(title, os_pref, age_limit, alpha):
    try:
        idx = df[df['name'] == title].index[0]
        # Hitung Similarity
        sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
        
        results_df = df.copy()
        # Rumus Hybrid: (Similarity * Alpha) + (Popularity * (1 - Alpha))
        results_df['match_score'] = (sim_scores * alpha) + (results_df['norm_popularity'] * (1 - alpha))
        
        # Filtering OS dan Umur
        os_col = {'windows': 'os_windows', 'mac': 'os_mac', 'linux': 'os_linux'}[os_pref]
        results_df = results_df[(results_df[os_col] == 1) & (results_df['required_age'] <= age_limit)]
        
        # Ambil Top 5
        final = results_df[results_df['name'] != title].sort_values('match_score', ascending=False).head(5)
        
        if not final.empty and not df_raw.empty:
            final = final.drop(columns=['short_description', 'genres']).merge(df_raw, on='name', how='left')
        return final
    except Exception:
        return pd.DataFrame()

# --- 4. STRUKTUR UI DASHBOARD ---

# Judul Utama Arcade
st.markdown("<h1 class='arcade-font'>STEAM MATCHMAKER</h1>", unsafe_allow_html=True)

# Sidebar Area
with st.sidebar:
    st.markdown("<h3 style='color:#58a6ff;'>MODEL TUNING</h3>", unsafe_allow_html=True)
    alpha_val = st.slider("Content vs Popularity", 0.0, 1.0, 0.7)
    st.caption("1.0 = Fokus Deskripsi | 0.0 = Fokus Popularitas")
    
    st.divider()
    st.markdown("<h3 style='color:#58a6ff;'>PREFERENCES</h3>", unsafe_allow_html=True)
    selected_os = st.selectbox("Pilih Platform OS:", ["Windows", "Mac", "Linux"]).lower()
    age_limit = st.select_slider("Batas Usur (Age Rating):", options=[0, 7, 12, 16, 18], value=18)
    
    st.divider()
    st.info("Algoritma menggunakan Content-Based Filtering dengan pembobotan Log-Popularitas.")

# Konten Utama (Centered Search Bar)
_, col_center, _ = st.columns([1, 2, 1])
with col_center:
    target_game = st.selectbox(
        "APA GAME YANG TERAKHIR KAMU MAINKAN?", 
        df['name'].values, 
        index=None,
        placeholder="Ketik judul game..."
    )
    
    st.write("") # Spasi vertikal
    btn_generate = st.button("🚀 TEMUKAN MATCH!", use_container_width=True)

# Tampilan Hasil Rekomendasi
if btn_generate:
    if target_game:
        with st.status("🛸 Menganalisis Database...", expanded=False) as status:
            recomm_list = get_recommendations(target_game, selected_os, age_limit, alpha_val)
            status.update(label="Analisis Selesai!", state="complete")
        
        if not recomm_list.empty:
            st.balloons()
            st.subheader(f"Rekomendasi Berdasarkan {target_game}:")
            
            for _, row in recomm_list.iterrows():
                # Menghitung persentase kecocokan untuk tampilan
                display_pct = min(int(row['match_score'] * 100), 99)
                
                # Render Game Card
                st.markdown(f"""
                <div class="game-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 1.3em; font-weight: bold; color: #58a6ff;">{row['name']}</span>
                        <span class="match-percent">{display_pct}% MATCH</span>
                    </div>
                    <div style="color: #8b949e; font-size: 0.85em; margin-bottom: 8px;">
                        Rated: {row['required_age']}+
                    </div>
                    <div style="font-size: 0.95em; line-height: 1.4;">{row['short_description']}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Maaf, tidak ada game yang cocok dengan kriteria OS atau Umurmu.")
    else:
        st.error("Silakan pilih judul game terlebih dahulu!")

# Footer Identitas Kelompok
st.divider()
st.markdown("<center><small>Institut Teknologi Sepuluh Nopember | ETS DATA MINING | Kelompok 12</small></center>", unsafe_allow_html=True)