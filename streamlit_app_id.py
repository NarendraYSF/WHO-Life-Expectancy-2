# Mencegah masalah PyArrow dengan mengatur variabel lingkungan sebelum import
import os
os.environ['PANDAS_USE_PYARROW'] = 'false'
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
# Nonaktifkan PyArrow sepenuhnya
os.environ['ARROW_DISABLE'] = 'true'

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from datetime import datetime
import warnings

# Menekan peringatan dan error PyArrow
warnings.filterwarnings('ignore', category=UserWarning, module='pyarrow')
warnings.filterwarnings('ignore', category=FutureWarning, module='pyarrow')

# Import modul kustom
from data_preprocessing import DataPreprocessor
from model_training import LifeExpectancyModel
from visualization import LifeExpectancyVisualizer
from outlier_detection import OutlierDetectionSystem
from pca_analysis import PCAAnalysis
from pca_visualization import PCAVisualizer
from pca_feature_selection import PCAFeatureSelector

# Import modul sklearn untuk regresi linear
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Import library pemetaan
import folium
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster, HeatMap

# Konfigurasi pandas untuk menangani tipe data bermasalah dengan lebih baik
pd.options.mode.use_inf_as_na = True

# Nonaktifkan PyArrow di pandas sepenuhnya
try:
    import pyarrow
    # Ini akan mencegah pandas menggunakan PyArrow
    pd.options.io.parquet.engine = 'fastparquet'
except ImportError:
    pass

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Harapan Hidup",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tambahkan catatan tentang bypass PyArrow
st.markdown("""
<div style="background-color: #e8f4fd; padding: 10px; border-radius: 5px; margin-bottom: 20px;">
    <strong>Catatan:</strong> Aplikasi ini menggunakan metode tampilan alternatif untuk menghindari masalah kompatibilitas PyArrow. 
    Data ditampilkan menggunakan tabel HTML untuk kompatibilitas yang lebih baik. Setiap error terkait PyArrow di konsol dapat diabaikan dengan aman.
</div>
""", unsafe_allow_html=True)

# CSS Kustom
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .sidebar-header {
        font-size: 1.5rem;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .outlier-info {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Muat dan cache dataset."""
    preprocessor = DataPreprocessor()
    data_file = "Dataset/WHO Life Expectancy Descriptive Statistics - Raw Data.csv"
    
    if os.path.exists(data_file):
        try:
            df = preprocessor.load_and_clean_data(data_file)
            
            # Sanitasi tambahan untuk mencegah masalah PyArrow
            df_clean = df.copy()
            
            # Tangani tipe data bermasalah yang tersisa
            for col in df_clean.columns:
                try:
                    # Konversi kolom numerik bermasalah ke float, menangani error
                    if df_clean[col].dtype in ['object', 'string']:
                        # Coba konversi ke numerik terlebih dahulu
                        numeric_series = pd.to_numeric(df_clean[col], errors='coerce')
                        if not numeric_series.isna().all():  # Jika konversi berhasil
                            df_clean[col] = numeric_series
                        else:
                            # Tetap sebagai string tapi bersihkan
                            df_clean[col] = df_clean[col].astype(str)
                            df_clean[col] = df_clean[col].replace(['nan', 'None', 'NULL', ''], 'N/A')
                    elif df_clean[col].dtype in ['float64', 'int64']:
                        # Tangani nilai tak terbatas
                        df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
                        # Pastikan tipe numerik yang tepat
                        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                except Exception as e:
                    # Jika konversi gagal, konversi ke string
                    df_clean[col] = df_clean[col].astype(str)
                    df_clean[col] = df_clean[col].replace(['nan', 'None', 'NULL', ''], 'N/A')
            
            return df_clean, preprocessor
            
        except Exception as e:
            st.error(f"Error memuat data: {e}")
            return None, None
    else:
        st.error(f"File data tidak ditemukan di {data_file}")
        return None, None

@st.cache_resource
def train_models(X, y, feature_names):
    """Latih dan cache model."""
    model_trainer = LifeExpectancyModel()
    results = model_trainer.train_models(X, y, feature_names)
    return results, model_trainer

def show_detailed_outlier_info(df, outlier_analysis):
    """Tampilkan informasi detail tentang outlier yang terdeteksi."""
    
    st.subheader("📋 Informasi Detail Outlier")
    
    # Buat tab untuk berbagai jenis informasi outlier
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Outlier Target", "📊 Outlier Fitur", "🌐 Outlier Multivariat", "📈 Analisis Dampak"])
    
    with tab1:
        st.write("**Outlier Harapan Hidup berdasarkan Metode Deteksi**")
        
        for method, results in outlier_analysis['target_outliers'].items():
            with st.expander(f"{method.upper()} - {results['outlier_count']} outlier ({results['outlier_percentage']:.2f}%)"):
                if results['outlier_count'] > 0:
                    # Ambil data outlier
                    target_data = df['Life expectancy'].dropna()
                    outlier_indices = results['outlier_indices']
                    outlier_values = target_data.iloc[outlier_indices]
                    outlier_scores = results['outlier_scores'][outlier_indices]
                    
                    # Buat tabel detail outlier
                    outlier_details = []
                    for i, (idx, value, score) in enumerate(zip(outlier_indices, outlier_values, outlier_scores)):
                        # Temukan baris asli
                        original_idx = df.index[df['Life expectancy'] == value].tolist()
                        if original_idx:
                            row = df.loc[original_idx[0]]
                            outlier_details.append({
                                'Index': i + 1,
                                'Negara': row['Country'],
                                'Tahun': row['Year'],
                                'Status': row['Status'],
                                'Wilayah': row['Region'],
                                'Harapan Hidup': f"{value:.1f}",
                                'Skor Outlier': f"{score:.3f}",
                                'Index Asli': original_idx[0]
                            })
                    
                    if outlier_details:
                        outlier_df = pd.DataFrame(outlier_details)
                        safe_display_dataframe(outlier_df, use_container_width=True)
                        
                        # Tampilkan statistik
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Outlier Min", f"{outlier_values.min():.1f}")
                        with col2:
                            st.metric("Outlier Max", f"{outlier_values.max():.1f}")
                        with col3:
                            st.metric("Outlier Rata-rata", f"{outlier_values.mean():.1f}")
                        
                        # Tampilkan distribusi outlier
                        fig = px.histogram(outlier_values, title=f"Distribusi Outlier {method}")
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Tidak ada outlier yang terdeteksi dengan metode ini.")
    
    with tab2:
        st.write("**Analisis Outlier per Fitur**")
        
        # Pilih fitur untuk dianalisis
        feature_options = list(outlier_analysis['feature_outliers'].keys())
        selected_feature = st.selectbox("Pilih fitur:", feature_options)
        
        if selected_feature:
            feature_methods = outlier_analysis['feature_outliers'][selected_feature]
            
            for method, results in feature_methods.items():
                with st.expander(f"{selected_feature} - {method.upper()}"):
                    if results['outlier_count'] > 0:
                        # Ambil data fitur
                        feature_data = df[selected_feature].dropna()
                        outlier_indices = results['outlier_indices']
                        outlier_values = feature_data.iloc[outlier_indices]
                        
                        st.write(f"**Outlier terdeteksi:** {results['outlier_count']} ({results['outlier_percentage']:.2f}%)")
                        
                        # Tampilkan statistik outlier
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Rentang Outlier", f"{outlier_values.min():.3f} - {outlier_values.max():.3f}")
                        with col2:
                            st.metric("Rentang Keseluruhan", f"{feature_data.min():.3f} - {feature_data.max():.3f}")
                        with col3:
                            st.metric("Rata-rata Outlier", f"{outlier_values.mean():.3f}")
                        
                        # Tampilkan distribusi outlier
                        fig = px.histogram(outlier_values, title=f"Distribusi Outlier {selected_feature}")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Tidak ada outlier terdeteksi untuk fitur ini.")
    
    with tab3:
        st.write("**Analisis Outlier Multivariat**")
        
        for method, results in outlier_analysis['multivariate_outliers'].items():
            with st.expander(f"{method.upper()} - {results['outlier_count']} outlier ({results['outlier_percentage']:.2f}%)"):
                if results['outlier_count'] > 0:
                    # Ambil indeks outlier
                    outlier_indices = results['outlier_indices']
                    
                    # Buat tabel detail
                    multivariate_details = []
                    for i, idx in enumerate(outlier_indices[:20]):  # Tampilkan 20 pertama
                        row = df.iloc[idx]
                        multivariate_details.append({
                            'Index': i + 1,
                            'Negara': row['Country'],
                            'Tahun': row['Year'],
                            'Status': row['Status'],
                            'Wilayah': row['Region'],
                            'Harapan Hidup': f"{row['Life expectancy']:.1f}",
                            'Skor Outlier': f"{results['outlier_scores'][idx]:.3f}",
                            'Index Asli': idx
                        })
                    
                    if multivariate_details:
                        multivariate_df = pd.DataFrame(multivariate_details)
                        safe_display_dataframe(multivariate_df, use_container_width=True)
                        
                        if len(outlier_indices) > 20:
                            st.info(f"... dan {len(outlier_indices) - 20} outlier lainnya")
                        
                        # Tampilkan plot sebar 2D dari dua fitur numerik pertama
                        numerical_cols = df.select_dtypes(include=[np.number]).columns[:2]
                        if len(numerical_cols) >= 2:
                            fig = px.scatter(
                                df, 
                                x=numerical_cols[0], 
                                y=numerical_cols[1],
                                title=f"Outlier Multivariat: {numerical_cols[0]} vs {numerical_cols[1]}"
                            )
                            
                            # Sorot outlier
                            outlier_df = df.iloc[outlier_indices]
                            fig.add_trace(go.Scatter(
                                x=outlier_df[numerical_cols[0]],
                                y=outlier_df[numerical_cols[1]],
                                mode='markers',
                                marker=dict(color='red', size=8),
                                name='Outlier'
                            ))
                            
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Tidak ada outlier multivariat yang terdeteksi.")
    
    with tab4:
        st.write("**Analisis Dampak Penghapusan Outlier**")
        
        # Tampilkan dampak untuk setiap metode
        for method, results in outlier_analysis['target_outliers'].items():
            if results['outlier_count'] > 0:
                with st.expander(f"Dampak penghapusan outlier {method}"):
                    # Hitung statistik
                    target_data = df['Life expectancy'].dropna()
                    
                    stats_before = {
                        'count': len(target_data),
                        'mean': target_data.mean(),
                        'std': target_data.std(),
                        'median': target_data.median(),
                        'min': target_data.min(),
                        'max': target_data.max()
                    }
                    
                    outlier_indices = results['outlier_indices']
                    clean_data = np.delete(target_data.values, outlier_indices)
                    
                    stats_after = {
                        'count': len(clean_data),
                        'mean': np.mean(clean_data),
                        'std': np.std(clean_data),
                        'median': np.median(clean_data),
                        'min': np.min(clean_data),
                        'max': np.max(clean_data)
                    }
                    
                    # Tampilkan perbandingan
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Record", f"{stats_before['count']}", f"{stats_after['count'] - stats_before['count']}")
                    
                    with col2:
                        mean_change = stats_after['mean'] - stats_before['mean']
                        st.metric("Rata-rata", f"{stats_before['mean']:.2f}", f"{mean_change:+.2f}")
                    
                    with col3:
                        std_change = stats_after['std'] - stats_before['std']
                        st.metric("Std Dev", f"{stats_before['std']:.2f}", f"{std_change:+.2f}")
                    
                    # Tampilkan grafik perbandingan sebelum/sesudah
                    comparison_data = pd.DataFrame({
                        'Metrik': ['Rata-rata', 'Median', 'Std Dev', 'Min', 'Max'],
                        'Sebelum': [stats_before['mean'], stats_before['median'], stats_before['std'], 
                                 stats_before['min'], stats_before['max']],
                        'Sesudah': [stats_after['mean'], stats_after['median'], stats_after['std'], 
                                stats_after['min'], stats_after['max']]
                    })
                    
                    fig = px.bar(comparison_data, x='Metrik', y=['Sebelum', 'Sesudah'], 
                               title=f"Statistik Sebelum vs Sesudah Menghapus Outlier {method}",
                               barmode='group')
                    st.plotly_chart(fig, use_container_width=True)

def sanitize_dataframe_for_display(df):
    """
    Sanitasi DataFrame untuk mencegah error konversi PyArrow.
    
    Args:
        df (pd.DataFrame): DataFrame input
        
    Returns:
        pd.DataFrame: DataFrame yang sudah disanitasi aman untuk ditampilkan
    """
    try:
        df_clean = df.copy()
        
        # Tangani tipe data bermasalah
        for col in df_clean.columns:
            try:
                # Periksa apakah kolom berisi nilai bermasalah
                if df_clean[col].dtype == 'object':
                    # Konversi kolom object ke string, menangani nilai NaN
                    df_clean[col] = df_clean[col].astype(str)
                    # Ganti representasi string bermasalah
                    df_clean[col] = df_clean[col].replace(['nan', 'None', 'NULL', ''], 'N/A')
                elif df_clean[col].dtype == 'float64':
                    # Tangani nilai tak terbatas di kolom float
                    df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
                    # Konversi ke string jika masih ada masalah
                    if df_clean[col].isnull().any():
                        df_clean[col] = df_clean[col].astype(str)
                        df_clean[col] = df_clean[col].replace('nan', 'N/A')
                elif df_clean[col].dtype == 'int64':
                    # Tangani kolom integer dengan nilai NaN
                    if df_clean[col].isnull().any():
                        df_clean[col] = df_clean[col].astype(str)
                        df_clean[col] = df_clean[col].replace('nan', 'N/A')
            except Exception as e:
                # Jika konversi gagal, konversi ke string
                df_clean[col] = df_clean[col].astype(str)
                df_clean[col] = df_clean[col].replace(['nan', 'None', 'NULL', ''], 'N/A')
        
        return df_clean
    except Exception as e:
        # Jika semua gagal, kembalikan representasi string dasar
        st.error(f"Error kritis dalam sanitasi data: {e}")
        return pd.DataFrame({'Error': ['Data tidak dapat disanitasi untuk ditampilkan']})

def alternative_display_dataframe(df, title="Data"):
    """
    Tampilkan DataFrame menggunakan metode alternatif ketika PyArrow gagal sepenuhnya.
    
    Args:
        df (pd.DataFrame): DataFrame yang akan ditampilkan
        title (str): Judul untuk tampilan
    """
    st.write(f"**{title}**")
    st.write(f"Ukuran: {df.shape[0]} baris × {df.shape[1]} kolom")
    
    # Tampilkan tipe data
    st.write("**Tipe Data:**")
    dtype_info = df.dtypes.value_counts()
    for dtype, count in dtype_info.items():
        st.write(f"- {dtype}: {count} kolom")
    
    # Tampilkan beberapa baris pertama sebagai tabel HTML
    if len(df) > 0:
        st.write("**10 baris pertama:**")
        
        # Tampilkan menggunakan pendekatan yang ramah Streamlit alih-alih HTML
        first_10_df = df.head(10)
        
        # Tampilkan data menggunakan st.write yang menangani DataFrame dengan baik
        st.write(first_10_df)
        
        if len(df) > 10:
            st.info(f"... dan {len(df) - 10} baris lainnya")
            
        # Tampilkan statistik ringkasan untuk kolom numerik
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            st.write("**Statistik Ringkasan (Kolom Numerik):**")
            summary_stats = df[numeric_cols].describe()
            
            # Tampilkan statistik ringkasan menggunakan st.write
            st.write(summary_stats)
    else:
        st.info("Tidak ada data untuk ditampilkan")

def safe_display_dataframe(df, use_container_width=True):
    """
    Tampilkan DataFrame dengan aman dengan sepenuhnya melewati PyArrow.
    
    Args:
        df (pd.DataFrame): DataFrame yang akan ditampilkan
        use_container_width (bool): Apakah menggunakan lebar container (diabaikan untuk tampilan alternatif)
    """
    # Selalu gunakan metode tampilan alternatif untuk sepenuhnya menghindari PyArrow
    alternative_display_dataframe(df, "Konten DataFrame")

def get_country_coordinates(country_name):
    """
    Dapatkan koordinat negara menggunakan dictionary pemetaan.
    
    Args:
        country_name (str): Nama negara
        
    Returns:
        tuple: (latitude, longitude) atau None jika tidak ditemukan
    """
    # Pemetaan koordinat negara
    country_coordinates = {
        'Afghanistan': (33.9391, 67.7100),
        'Albania': (41.1533, 20.1683),
        'Algeria': (28.0339, 1.6596),
        'Andorra': (42.5063, 1.5218),
        'Angola': (-11.2027, 17.8739),
        'Antigua and Barbuda': (17.0608, -61.7964),
        'Argentina': (-38.4161, -63.6167),
        'Armenia': (40.0691, 45.0382),
        'Australia': (-25.2744, 133.7751),
        'Austria': (47.5162, 14.5501),
        'Azerbaijan': (40.1431, 47.5769),
        'Bahamas': (25.0343, -77.3963),
        'Bahrain': (26.0667, 50.5577),
        'Bangladesh': (23.6850, 90.3563),
        'Barbados': (13.1939, -59.5432),
        'Belarus': (53.7098, 27.9534),
        'Belgium': (50.8503, 4.3517),
        'Belize': (17.1899, -88.4976),
        'Benin': (9.3077, 2.3158),
        'Bhutan': (27.5142, 90.4336),
        'Bolivia': (-16.2902, -63.5887),
        'Bosnia and Herzegovina': (43.9159, 17.6791),
        'Botswana': (-22.3285, 24.6849),
        'Brazil': (-14.2350, -51.9253),
        'Brunei': (4.5353, 114.7277),
        'Bulgaria': (42.7339, 25.4858),
        'Burkina Faso': (12.2383, -1.5616),
        'Burundi': (-3.3731, 29.9189),
        'Cambodia': (12.5657, 104.9910),
        'Cameroon': (7.3697, 12.3547),
        'Canada': (56.1304, -106.3468),
        'Cape Verde': (16.5388, -23.0418),
        'Central African Republic': (6.6111, 20.9394),
        'Chad': (15.4542, 18.7322),
        'Chile': (-35.6751, -71.5430),
        'China': (35.8617, 104.1954),
        'Colombia': (4.5709, -74.2973),
        'Comoros': (-11.6455, 43.3333),
        'Congo': (-0.2280, 15.8277),
        'Costa Rica': (9.9281, -84.0907),
        'Cuba': (23.1162, -82.3862),
        'Cyprus': (35.1264, 33.4299),
        'Czech Republic': (49.8175, 15.4730),
        'Denmark': (56.2639, 9.5018),
        'Djibouti': (11.8251, 42.5903),
        'Dominica': (15.4150, -61.3710),
        'Dominican Republic': (18.7357, -70.1627),
        'Ecuador': (-1.8312, -78.1834),
        'Egypt': (26.8206, 30.8025),
        'El Salvador': (13.7942, -88.8965),
        'Equatorial Guinea': (1.6508, 10.2679),
        'Eritrea': (15.1794, 39.7823),
        'Estonia': (59.4360, 24.7424),
        'Ethiopia': (9.1450, 40.4897),
        'Fiji': (-16.5782, 179.4144),
        'Finland': (65.5696, 25.0505),
        'France': (46.2276, 2.2137),
        'Gabon': (-0.8037, 11.6094),
        'Gambia': (13.4432, -15.3101),
        'Georgia': (41.7151, 44.8271),
        'Germany': (51.1657, 10.4515),
        'Ghana': (7.9465, -1.0232),
        'Greece': (39.0742, 21.8243),
        'Grenada': (12.2628, -61.6041),
        'Guatemala': (15.7835, -90.2308),
        'Guinea': (9.9456, -9.6966),
        'Guinea-Bissau': (11.8037, -15.1804),
        'Guyana': (4.8604, -58.9301),
        'Haiti': (18.9712, -72.2852),
        'Holy See': (41.6426, 12.4151),
        'Honduras': (15.1999, -86.2419),
        'Hungary': (47.1625, 19.5033),
        'Iceland': (64.9631, -19.0208),
        'India': (20.5937, 78.9629),
        'Indonesia': (-0.7893, 113.9213),
        'Iran': (32.4279, 53.6880),
        'Iraq': (33.2238, 43.6793),
        'Ireland': (53.4129, -8.2439),
        'Israel': (31.0461, 34.8516),
        'Italy': (41.8719, 12.5674),
        'Jamaica': (18.1096, -77.2975),
        'Japan': (36.2048, 138.2529),
        'Jordan': (31.2451, 36.5881),
        'Kazakhstan': (48.0196, 66.9237),
        'Kenya': (-0.0236, 37.9062),
        'Kiribati': (-3.3233, 176.7267),
        'Kuwait': (29.3375, 47.6581),
        'Kyrgyzstan': (41.2043, 74.7661),
        'Laos': (19.8563, 102.4955),
        'Latvia': (56.8796, 24.6032),
        'Lebanon': (33.8547, 35.8623),
        'Lesotho': (-29.6098, 28.2355),
        'Liberia': (6.4281, -9.4295),
        'Libya': (26.3351, 17.2283),
        'Liechtenstein': (47.1660, 9.5554),
        'Lithuania': (55.1694, 23.8813),
        'Luxembourg': (49.8153, 6.1296),
        'Madagascar': (-18.7669, 46.8691),
        'Malawi': (-13.2543, 34.3015),
        'Malaysia': (4.2105, 101.9758),
        'Maldives': (3.2028, 73.2207),
        'Mali': (17.5707, -3.9962),
        'Malta': (35.8897, 14.5147),
        'Marshall Islands': (7.1315, 171.1845),
        'Mauritania': (21.0079, -10.9408),
        'Mauritius': (-20.3484, 57.5521),
        'Mexico': (23.6345, -102.5528),
        'Micronesia': (7.4256, 150.5508),
        'Moldova': (47.4116, 28.3699),
        'Monaco': (43.7384, 7.4246),
        'Mongolia': (46.8625, 103.8467),
        'Montenegro': (42.7339, 19.3744),
        'Morocco': (31.7917, -7.0926),
        'Mozambique': (-18.6657, 35.5296),
        'Myanmar': (21.9139, 95.9562),
        'Namibia': (-22.9576, 18.4904),
        'Nauru': (-0.5228, 166.9315),
        'Nepal': (28.3949, 84.1240),
        'Netherlands': (52.1326, 5.2913),
        'New Zealand': (-40.9006, 174.8860),
        'Nicaragua': (12.8654, -85.2072),
        'Niger': (17.6077, 8.0817),
        'Nigeria': (9.0820, 8.6753),
        'North Korea': (40.3399, 127.5101),
        'North Macedonia': (41.6086, 21.7453),
        'Norway': (60.4720, 8.4689),
        'Oman': (21.5126, 55.9233),
        'Pakistan': (30.3753, 69.3451),
        'Palau': (7.5149, 134.5825),
        'Palestine State': (31.9522, 35.2332),
        'Panama': (8.5380, -80.7821),
        'Papua New Guinea': (-6.3160, 143.9555),
        'Paraguay': (-23.4425, -58.4438),
        'Peru': (-9.1899, -75.0152),
        'Philippines': (12.8797, 121.7740),
        'Poland': (51.9194, 19.1451),
        'Portugal': (39.3999, -8.2245),
        'Qatar': (25.3548, 51.1839),
        'Romania': (45.9432, 24.9666),
        'Russia': (61.5240, 105.3188),
        'Rwanda': (-1.9403, 29.8739),
        'Saint Kitts and Nevis': (17.3579, -62.7830),
        'Saint Lucia': (13.9094, -60.9789),
        'Saint Vincent and the Grenadines': (12.9843, -61.2872),
        'Samoa': (-13.7590, -172.1046),
        'San Marino': (43.9424, 12.4578),
        'Sao Tome and Principe': (0.1864, 6.6131),
        'Saudi Arabia': (23.8859, 45.0792),
        'Senegal': (14.4974, -14.4524),
        'Serbia': (44.0165, 21.0059),
        'Seychelles': (-4.6796, 55.4920),
        'Sierra Leone': (8.4606, -11.7799),
        'Singapore': (1.3521, 103.8198),
        'Slovakia': (48.6690, 19.6990),
        'Slovenia': (46.1512, 14.9955),
        'Solomon Islands': (-9.6457, 160.1562),
        'Somalia': (5.1521, 46.1996),
        'South Africa': (-29.0468, 24.6849),
        'South Korea': (35.9078, 127.7669),
        'South Sudan': (6.8770, 31.3070),
        'Spain': (40.4637, -3.7492),
        'Sri Lanka': (6.9271, 80.7718),
        'Sudan': (15.5007, 30.5925),
        'Suriname': (3.9193, -56.0278),
        'Sweden': (60.1282, 18.6435),
        'Switzerland': (46.8182, 8.2275),
        'Syria': (34.8025, 38.9968),
        'Tajikistan': (38.8610, 71.2761),
        'Thailand': (15.8700, 100.9925),
        'Togo': (8.6195, 0.8248),
        'Tonga': (-17.5653, -149.5627),
        'Trinidad and Tobago': (10.6918, -61.2225),
        'Tunisia': (33.8869, 9.5375),
        'Turkey': (38.5653, 35.9496),
        'Turkmenistan': (38.9697, 59.5563),
        'Tuvalu': (-7.1095, 177.3930),
        'Uganda': (1.3733, 32.2903),
        'Ukraine': (48.3794, 31.1656),
        'United Arab Emirates': (23.4241, 53.8478),
        'United Kingdom': (55.3781, -3.4360),
        'United States': (37.0902, -95.7129),
        'Uruguay': (-32.5228, -55.7658),
        'Uzbekistan': (41.3775, 64.5859),
        'Vanuatu': (-15.3767, 166.9592),
        'Vatican City': (41.9029, 12.4534),
        'Venezuela': (6.4238, -66.5897),
        'Vietnam': (14.0583, 108.2772),
        'Yemen': (15.5527, 48.5164),
        'Zambia': (-13.1339, 27.8675),
        'Zimbabwe': (-19.0155, 29.1549)
    }
    
    # Pemetaan nama negara umum untuk dataset WHO
    country_mapping = {
        'United States of America': 'United States',
        'United Kingdom of Great Britain and Northern Ireland': 'United Kingdom',
        'Russian Federation': 'Russia',
        'Iran (Islamic Republic of)': 'Iran',
        'Venezuela (Bolivarian Republic of)': 'Venezuela',
        'Syrian Arab Republic': 'Syria',
        'Korea, Republic of': 'South Korea',
        'Korea, Democratic People\'s Republic of': 'North Korea',
        'Lao People\'s Democratic Republic': 'Laos',
        'Brunei Darussalam': 'Brunei',
        'Côte d\'Ivoire': 'Ivory Coast',
        'Congo': 'Republic of the Congo',
        'Congo, Democratic Republic of the': 'Democratic Republic of the Congo',
        'Tanzania, United Republic of': 'Tanzania',
        'Central African Republic': 'Central African Republic',
        'Equatorial Guinea': 'Equatorial Guinea',
        'Guinea-Bissau': 'Guinea-Bissau',
        'Sao Tome and Principe': 'Sao Tome and Principe',
        'Cabo Verde': 'Cape Verde',
        'Eswatini': 'Eswatini',
        'Timor-Leste': 'East Timor'
    }
    
    # Coba temukan koordinat negara
    # Pertama coba dengan nama asli
    if country_name in country_coordinates:
        return country_coordinates[country_name]
    
    # Coba dengan nama yang dipetakan
    mapped_name = country_mapping.get(country_name, country_name)
    if mapped_name in country_coordinates:
        return country_coordinates[mapped_name]
    
    # Coba pencocokan fuzzy untuk kecocokan yang dekat
    for known_country, coords in country_coordinates.items():
        if country_name.lower() in known_country.lower() or known_country.lower() in country_name.lower():
            return coords
    
    return None

def create_interactive_map(df, metric='Life expectancy', year=None, region=None):
    """
    Buat visualisasi peta interaktif.
    
    Args:
        df (pd.DataFrame): DataFrame input
        metric (str): Metrik untuk divisualisasikan di peta
        year (int): Tahun tertentu untuk filter (opsional)
        region (str): Wilayah tertentu untuk filter (opsional)
        
    Returns:
        folium.Map: Peta interaktif
    """
    # Filter data berdasarkan parameter
    map_data = df.copy()
    
    if year is not None:
        map_data = map_data[map_data['Year'] == year]
    
    if region is not None:
        map_data = map_data[map_data['Region'] == region]
    
    # Kelompokkan berdasarkan negara dan hitung rata-rata untuk metrik
    country_data = map_data.groupby('Country').agg({
        metric: 'mean',
        'Status': 'first',
        'Region': 'first',
        'Year': ['min', 'max', 'count']
    }).reset_index()
    
    # Ratakan nama kolom
    country_data.columns = ['Country', f'{metric}_avg', 'Status', 'Region', 'Year_min', 'Year_max', 'Year_count']
    
    # Hapus baris dengan nilai metrik yang hilang
    country_data = country_data.dropna(subset=[f'{metric}_avg'])
    
    if len(country_data) == 0:
        return None
    
    # Buat peta dasar
    center_lat, center_lon = 20, 0
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=2,
        tiles='OpenStreetMap'
    )
    
    # Tambahkan layer tile yang berbeda
    folium.TileLayer('OpenStreetMap', name='OpenStreetMap').add_to(m)
    folium.TileLayer('CartoDB positron', name='CartoDB Positron').add_to(m)
    folium.TileLayer('CartoDB dark_matter', name='CartoDB Dark').add_to(m)
    
    # Buat cluster marker
    marker_cluster = MarkerCluster().add_to(m)
    
    # Pemetaan warna untuk status
    status_colors = {
        'Developed': 'green',
        'Developing': 'orange'
    }
    
    # Tambahkan marker untuk setiap negara
    for idx, row in country_data.iterrows():
        country_name = row['Country']
        metric_value = row[f'{metric}_avg']
        status = row['Status']
        region = row['Region']
        year_range = f"{row['Year_min']}-{row['Year_max']}"
        data_points = row['Year_count']
        
        # Dapatkan koordinat
        coords = get_country_coordinates(country_name)
        
        if coords:
            lat, lon = coords
            
            # Buat konten popup
            popup_content = f"""
            <div style="width: 200px;">
                <h4>{country_name}</h4>
                <p><strong>{metric}:</strong> {metric_value:.2f}</p>
                <p><strong>Status:</strong> {status}</p>
                <p><strong>Wilayah:</strong> {region}</p>
                <p><strong>Tahun:</strong> {year_range}</p>
                <p><strong>Data Points:</strong> {data_points}</p>
            </div>
            """
            
            # Buat marker
            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(popup_content, max_width=300),
                tooltip=f"{country_name}: {metric_value:.2f}",
                icon=folium.Icon(
                    color=status_colors.get(status, 'blue'),
                    icon='info-sign'
                )
            ).add_to(marker_cluster)
    
    # Tambahkan kontrol layer
    folium.LayerControl().add_to(m)
    
    return m

def create_choropleth_map(df, metric='Life expectancy', year=None):
    """
    Buat peta choropleth menggunakan Plotly.
    
    Args:
        df (pd.DataFrame): DataFrame input
        metric (str): Metrik untuk divisualisasikan
        year (int): Tahun tertentu untuk filter (opsional)
        
    Returns:
        plotly.graph_objects.Figure: Peta choropleth
    """
    # Filter data berdasarkan parameter
    map_data = df.copy()
    
    if year is not None:
        map_data = map_data[map_data['Year'] == year]
    
    # Kelompokkan berdasarkan negara dan hitung rata-rata untuk metrik
    country_data = map_data.groupby('Country').agg({
        metric: 'mean',
        'Status': 'first',
        'Region': 'first'
    }).reset_index()
    
    # Hapus baris dengan nilai metrik yang hilang
    country_data = country_data.dropna(subset=[metric])
    
    if len(country_data) == 0:
        return None
    
    # Buat peta choropleth
    fig = px.choropleth(
        country_data,
        locations='Country',
        locationmode='country names',
        color=metric,
        hover_name='Country',
        hover_data=['Status', 'Region'],
        color_continuous_scale='Viridis',
        title=f'Distribusi Global {metric}',
        labels={metric: metric}
    )
    
    fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='equirectangular'
        ),
        height=600
    )
    
    return fig

def create_heatmap_data(df, metric='Life expectancy', year=None):
    """
    Buat data heatmap untuk visualisasi.
    
    Args:
        df (pd.DataFrame): DataFrame input
        metric (str): Metrik untuk divisualisasikan
        year (int): Tahun tertentu untuk filter (opsional)
        
    Returns:
        list: Daftar [lat, lon, value] untuk heatmap
    """
    # Filter data berdasarkan parameter
    map_data = df.copy()
    
    if year is not None:
        map_data = map_data[map_data['Year'] == year]
    
    # Kelompokkan berdasarkan negara dan hitung rata-rata untuk metrik
    country_data = map_data.groupby('Country').agg({
        metric: 'mean'
    }).reset_index()
    
    # Hapus baris dengan nilai metrik yang hilang
    country_data = country_data.dropna(subset=[metric])
    
    heatmap_data = []
    
    for idx, row in country_data.iterrows():
        country_name = row['Country']
        metric_value = row[metric]
        
        # Dapatkan koordinat
        coords = get_country_coordinates(country_name)
        
        if coords:
            lat, lon = coords
            heatmap_data.append([lat, lon, metric_value])
    
    return heatmap_data

def show_map_visualizations(df):
    """
    Tampilkan visualisasi peta komprehensif.
    
    Args:
        df (pd.DataFrame): DataFrame input
    """
    st.markdown("## 🌍 Visualisasi Peta Interaktif")
    
    # Kontrol sidebar
    st.sidebar.markdown("### Kontrol Peta")
    
    # Pemilihan metrik
    metric_options = ['Life expectancy'] + [col for col in df.select_dtypes(include=[np.number]).columns 
                                          if col not in ['Year', 'Life expectancy']]
    selected_metric = st.sidebar.selectbox("Pilih Metrik:", metric_options)
    
    # Filter tahun
    years = sorted(df['Year'].unique())
    year_filter = st.sidebar.selectbox("Pilih Tahun (Opsional):", ['Semua Tahun'] + list(years))
    selected_year = None if year_filter == 'Semua Tahun' else int(year_filter)
    
    # Filter wilayah
    regions = sorted(df['Region'].unique())
    region_filter = st.sidebar.selectbox("Pilih Wilayah (Opsional):", ['Semua Wilayah'] + list(regions))
    selected_region = None if region_filter == 'Semua Wilayah' else region_filter
    
    # Pemilihan tipe peta
    map_type = st.sidebar.selectbox("Tipe Peta:", ["Marker Interaktif", "Choropleth", "Heatmap"])
    
    # Buat tab untuk tampilan peta yang berbeda
    tab1, tab2, tab3 = st.tabs(["🌍 Peta Interaktif", "📊 Statistik", "📈 Tren"])
    
    with tab1:
        st.subheader(f"🌍 Distribusi {selected_metric}")
        
        if map_type == "Marker Interaktif":
            # Buat peta interaktif dengan marker
            map_obj = create_interactive_map(df, selected_metric, selected_year, selected_region)
            
            if map_obj:
                folium_static(map_obj, width=800, height=600)
            else:
                st.warning("Tidak ada data yang tersedia untuk filter yang dipilih.")
        
        elif map_type == "Choropleth":
            # Buat peta choropleth
            fig = create_choropleth_map(df, selected_metric, selected_year)
            
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Tidak ada data yang tersedia untuk filter yang dipilih.")
        
        elif map_type == "Heatmap":
            # Buat heatmap
            heatmap_data = create_heatmap_data(df, selected_metric, selected_year)
            
            if heatmap_data:
                # Buat peta dasar untuk heatmap
                center_lat, center_lon = 20, 0
                m = folium.Map(
                    location=[center_lat, center_lon],
                    zoom_start=2,
                    tiles='CartoDB dark_matter'
                )
                
                # Tambahkan layer heatmap
                HeatMap(heatmap_data, radius=15).add_to(m)
                
                folium_static(m, width=800, height=600)
            else:
                st.warning("Tidak ada data yang tersedia untuk filter yang dipilih.")
    
    with tab2:
        st.subheader("📊 Statistik Geografis")
        
        # Filter data berdasarkan pilihan
        filtered_data = df.copy()
        if selected_year:
            filtered_data = filtered_data[filtered_data['Year'] == selected_year]
        if selected_region:
            filtered_data = filtered_data[filtered_data['Region'] == selected_region]
        
        # Hitung statistik berdasarkan wilayah
        region_stats = filtered_data.groupby('Region').agg({
            selected_metric: ['mean', 'std', 'min', 'max', 'count']
        }).round(2)
        
        # Ratakan nama kolom
        region_stats.columns = [f'{col[0]}_{col[1]}' for col in region_stats.columns]
        region_stats = region_stats.reset_index()
        
        # Tampilkan statistik wilayah
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Statistik Wilayah:**")
            safe_display_dataframe(region_stats, use_container_width=True)
        
        with col2:
            # Buat grafik batang rata-rata wilayah
            fig = px.bar(
                region_stats, 
                x='Region', 
                y=f'{selected_metric}_mean',
                title=f'Rata-rata {selected_metric} berdasarkan Wilayah',
                error_y=f'{selected_metric}_std'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Negara dengan nilai tertinggi dan terendah
        country_stats = filtered_data.groupby('Country').agg({
            selected_metric: 'mean'
        }).reset_index()
        country_stats = country_stats.dropna()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**10 Negara Teratas:**")
            top_countries = country_stats.nlargest(10, selected_metric)
            safe_display_dataframe(top_countries, use_container_width=True)
        
        with col2:
            st.write("**10 Negara Terbawah:**")
            bottom_countries = country_stats.nsmallest(10, selected_metric)
            safe_display_dataframe(bottom_countries, use_container_width=True)
    
    with tab3:
        st.subheader("📈 Tren Temporal")
        
        # Analisis deret waktu
        if selected_year is None:  # Hanya tampilkan tren jika tidak ada tahun tertentu yang dipilih
            # Hitung tren berdasarkan wilayah
            trend_data = df.groupby(['Year', 'Region'])[selected_metric].mean().reset_index()
            
            # Buat plot tren
            fig = px.line(
                trend_data,
                x='Year',
                y=selected_metric,
                color='Region',
                title=f'Tren {selected_metric} berdasarkan Wilayah dari Waktu ke Waktu'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Hitung tingkat pertumbuhan
            st.write("**Analisis Pertumbuhan:**")
            
            # Hitung rata-rata pertumbuhan tahunan berdasarkan wilayah
            growth_data = []
            for region in df['Region'].unique():
                region_data = df[df['Region'] == region].groupby('Year')[selected_metric].mean()
                if len(region_data) > 1:
                    growth_rate = (region_data.iloc[-1] - region_data.iloc[0]) / (region_data.index[-1] - region_data.index[0])
                    growth_data.append({
                        'Wilayah': region,
                        'Tingkat Pertumbuhan': growth_rate,
                        'Nilai Awal': region_data.iloc[0],
                        'Nilai Akhir': region_data.iloc[-1]
                    })
            
            if growth_data:
                growth_df = pd.DataFrame(growth_data)
                growth_df = growth_df.sort_values('Tingkat Pertumbuhan', ascending=False)
                safe_display_dataframe(growth_df, use_container_width=True)
        else:
            st.info("Pilih 'Semua Tahun' untuk melihat tren temporal.")

def main():
    """Aplikasi Streamlit utama."""
    
    # Header
    st.markdown('<h1 class="main-header">📊 Prediksi Harapan Hidup</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    st.sidebar.markdown('<h2 class="sidebar-header">Navigasi</h2>', unsafe_allow_html=True)
    
    # Language chooser
    st.sidebar.markdown("### 🌐 Bahasa / Language")
    language_choice = st.sidebar.selectbox(
        "Pilih Bahasa / Choose Language:",
        ["🇮🇩 Indonesia", "🇺🇸 English"]
    )
    
    # Add link to English version if selected
    if language_choice == "🇺🇸 English":
        st.sidebar.markdown("---")
        st.sidebar.markdown("**English Version:**")
        st.sidebar.markdown("[Open English App](https://life-expectancy-linear-regression-app.streamlit.app/)")
        st.sidebar.markdown("---")
    
    page = st.sidebar.selectbox(
        "Pilih halaman:",
        ["🏠 Beranda", "📈 Analisis Data", "🌍 Visualisasi Peta", "🔍 Deteksi Outlier", "📋 Detail Outlier", "🔬 Analisis PCA", "🤖 Pelatihan Model", "📊 Hasil", "🔮 Prediksi", "📋 Tentang"]
    )
    
    # Muat data
    with st.spinner("Memuat dataset..."):
        df, preprocessor = load_data()
    
    if df is None:
        st.error("Gagal memuat dataset. Silakan periksa path file.")
        return
    
    # Halaman Beranda
    if page == "🏠 Beranda":
        st.markdown("## Selamat Datang di Prediksi Harapan Hidup")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            Aplikasi ini menggunakan machine learning untuk memprediksi harapan hidup berdasarkan berbagai indikator kesehatan, 
            ekonomi, dan sosial dari dataset Organisasi Kesehatan Dunia (WHO).
            
            ### Fitur:
            - **Preprocessing Data**: Menangani nilai yang hilang, pembersihan data, dan rekayasa fitur
            - **Deteksi Outlier Komprehensif**: Berbagai metode deteksi dengan analisis detail
            - **Analisis Komponen Utama (PCA)**: Reduksi dimensi, seleksi fitur, dan visualisasi komponen
            - **Model Beragam**: Regresi Linear, Ridge, Lasso, Elastic Net, Random Forest, Gradient Boosting
            - **Evaluasi Model**: Validasi silang, tuning hyperparameter, dan metrik performa
            - **Visualisasi Lanjutan**: Plot interaktif dan analisis komprehensif
            - **Prediksi**: Membuat prediksi pada data baru
            
            ### Ikhtisar Dataset:
            """)
            
            # Info dataset
            st.info(f"📊 **Ukuran Dataset**: {df.shape[0]} baris × {df.shape[1]} kolom")
            st.info(f"🌍 **Negara**: {df['Country'].nunique()}")
            st.info(f"📅 **Tahun**: {df['Year'].min()} - {df['Year'].max()}")
            st.info(f"🎯 **Variabel Target**: Harapan hidup")
        
        with col2:
            st.markdown("### Statistik Cepat")
            
            # Statistik harapan hidup
            life_exp = df['Life expectancy'].dropna()
            col2.metric("Rata-rata Harapan Hidup", f"{life_exp.mean():.1f} tahun")
            col2.metric("Harapan Hidup Min", f"{life_exp.min():.1f} tahun")
            col2.metric("Harapan Hidup Max", f"{life_exp.max():.1f} tahun")
    
    # Halaman Analisis Data
    elif page == "📈 Analisis Data":
        st.markdown("## Analisis Data")
        
        # Ikhtisar data
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Informasi Dataset")
            st.write(f"**Ukuran**: {df.shape}")
            st.write(f"**Kolom**: {list(df.columns)}")
            
            # Nilai yang hilang
            missing_data = df.isnull().sum()
            missing_df = pd.DataFrame({
                'Kolom': missing_data.index,
                'Nilai Hilang': missing_data.values,
                'Persentase': (missing_data.values / len(df)) * 100
            }).sort_values('Nilai Hilang', ascending=False)
            
            st.subheader("Analisis Nilai Hilang")
            safe_display_dataframe(missing_df, use_container_width=True)
        
        with col2:
            st.subheader("Tipe Data")
            st.write(df.dtypes.value_counts())
            
            # Distribusi target
            st.subheader("Distribusi Harapan Hidup")
            fig = px.histogram(df, x='Life expectancy', nbins=30, 
                             title="Distribusi Harapan Hidup")
            st.plotly_chart(fig, use_container_width=True)
        
        # Analisis korelasi
        st.subheader("Analisis Korelasi")
        
        # Pilih kolom numerik
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        corr_matrix = df[numerical_cols].corr()
        
        # Buat heatmap korelasi
        fig = px.imshow(corr_matrix, 
                       title="Matriks Korelasi",
                       color_continuous_scale='RdBu',
                       aspect='auto')
        st.plotly_chart(fig, use_container_width=True)
        
        # Analisis fitur vs target
        st.subheader("Analisis Fitur vs Target")
        
        # Pilih fitur untuk dianalisis
        feature_cols = [col for col in numerical_cols if col != 'Life expectancy']
        selected_feature = st.selectbox("Pilih fitur untuk dianalisis:", feature_cols)
        
        if selected_feature:
            # Buat plot sebar dengan informasi negara (tanpa garis tren)
            fig = px.scatter(
                df, 
                x=selected_feature, 
                y='Life expectancy',
                color='Country',  # Warna berdasarkan negara
                hover_data=['Country', 'Year', 'Status', 'Region'],  # Tampilkan info tambahan saat hover
                title=f"Harapan Hidup vs {selected_feature} (Diberi Warna berdasarkan Negara)"
            )
            
            # Tambahkan garis regresi linear untuk prediksi
            # Siapkan data untuk regresi linear
            X = df[[selected_feature]].dropna()
            y = df['Life expectancy'].dropna()
            
            # Sejajarkan data
            common_indices = X.index.intersection(y.index)
            X = X.loc[common_indices]
            y = y.loc[common_indices]
            
            if len(X) > 0:
                try:
                    # Fit regresi linear
                    lr = LinearRegression()
                    lr.fit(X, y)
                    
                    # Generate titik untuk garis regresi
                    x_min, x_max = X[selected_feature].min(), X[selected_feature].max()
                    x_range = np.linspace(x_min, x_max, 100)
                    y_pred = lr.predict(x_range.reshape(-1, 1))
                    
                    # Tambahkan garis regresi ke plot
                    fig.add_trace(go.Scatter(
                        x=x_range,
                        y=y_pred,
                        mode='lines',
                        name='Regresi Linear (Prediksi)',
                        line=dict(color='red', width=3, dash='dash'),
                        showlegend=True
                    ))
                    
                    # Hitung skor R²
                    y_pred_all = lr.predict(X)
                    r2_score = lr.score(X, y)
                    
                    # Tambahkan informasi R² ke plot
                    fig.add_annotation(
                        x=0.05, y=0.95,
                        xref='paper', yref='paper',
                        text=f'R² = {r2_score:.3f}',
                        showarrow=False,
                        font=dict(size=14, color='red'),
                        bgcolor='rgba(255,255,255,0.8)',
                        bordercolor='red',
                        borderwidth=1
                    )
                except Exception as e:
                    st.warning(f"⚠️ Tidak dapat menghitung regresi linear: {e}")
                    st.info("Plot sebar ditampilkan tanpa garis regresi.")
            
            # Update layout untuk keterbacaan lebih baik
            fig.update_layout(
                height=600,
                showlegend=True,  # Tampilkan legenda untuk garis regresi
                hovermode='closest'
            )
            
            # Tambahkan catatan tentang informasi hover
            st.info("💡 **Arahkan kursor ke titik untuk melihat negara, tahun, status, dan wilayah**")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistik negara untuk fitur terpilih
            st.subheader(f"Statistik Negara untuk {selected_feature}")
            
            # Hitung statistik per negara
            country_stats = df.groupby('Country').agg({
                selected_feature: ['mean', 'std', 'min', 'max'],
                'Life expectancy': ['mean', 'std', 'min', 'max']
            }).round(2)
            
            # Ratakan nama kolom
            country_stats.columns = [f"{col[0]}_{col[1]}" for col in country_stats.columns]
            country_stats = country_stats.reset_index()
            
            # Tampilkan negara dengan nilai fitur tertinggi dan terendah
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**10 Negara Teratas berdasarkan Rata-rata {selected_feature}:**")
                top_countries = country_stats.nlargest(10, f'{selected_feature}_mean')
                safe_display_dataframe(top_countries[['Country', f'{selected_feature}_mean', f'{selected_feature}_std']], use_container_width=True)
            
            with col2:
                st.write(f"**10 Negara Terbawah berdasarkan Rata-rata {selected_feature}:**")
                bottom_countries = country_stats.nsmallest(10, f'{selected_feature}_mean')
                safe_display_dataframe(bottom_countries[['Country', f'{selected_feature}_mean', f'{selected_feature}_std']], use_container_width=True)
    
    # Visualisasi peta
    elif page == "🌍 Visualisasi Peta":
        show_map_visualizations(df)
    
    # Deteksi outlier
    elif page == "🔍 Deteksi Outlier":
        st.markdown("## Deteksi Outlier Komprehensif")
        
        # Inisialisasi sistem deteksi outlier
        outlier_system = OutlierDetectionSystem()
        
        # Pemilihan metode deteksi
        st.subheader("Metode Deteksi")
        
        col1, col2 = st.columns(2)
        
        with col1:
            univariate_methods = st.multiselect(
                "Pilih metode deteksi univariat:",
                ['zscore', 'iqr', 'modified_zscore', 'isolation_forest'],
                default=['zscore', 'iqr']
            )
            
            zscore_threshold = st.slider("Threshold Z-score", 2.0, 5.0, 3.0, 0.1)
            iqr_multiplier = st.slider("Pengali IQR", 1.0, 3.0, 1.5, 0.1)
        
        with col2:
            multivariate_methods = st.multiselect(
                "Pilih metode deteksi multivariat:",
                ['isolation_forest', 'local_outlier_factor', 'elliptic_envelope'],
                default=['isolation_forest']
            )
            
            contamination = st.slider("Tingkat kontaminasi", 0.01, 0.2, 0.1, 0.01)
        
        # Jalankan analisis outlier
        if st.button("🔍 Jalankan Analisis Outlier", type="primary"):
            with st.spinner("Menjalankan analisis outlier komprehensif..."):
                # Lakukan analisis
                outlier_analysis = outlier_system.comprehensive_outlier_analysis(
                    df, 
                    target_column='Life expectancy',
                    methods=univariate_methods
                )
                
                # Simpan di session state
                st.session_state.outlier_analysis = outlier_analysis
                st.session_state.outlier_system = outlier_system
                
                st.success("✅ Analisis outlier selesai!")
        
        # Tampilkan hasil jika tersedia
        if 'outlier_analysis' in st.session_state:
            outlier_analysis = st.session_state.outlier_analysis
            outlier_system = st.session_state.outlier_system
            
            # Metrik ringkasan
            st.subheader("Ringkasan Deteksi Outlier")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_methods = len(outlier_analysis['target_outliers'])
                st.metric("Metode Deteksi", total_methods)
            
            with col2:
                total_outliers = sum([results['outlier_count'] for results in outlier_analysis['target_outliers'].values()])
                st.metric("Total Outlier Terdeteksi", total_outliers)
            
            with col3:
                avg_percentage = np.mean([results['outlier_percentage'] for results in outlier_analysis['target_outliers'].values()])
                st.metric("Rata-rata Outlier %", f"{avg_percentage:.2f}%")
            
            # Hasil detail
            st.subheader("Hasil Detail berdasarkan Metode")
            
            # Buat tabel hasil
            results_data = []
            for method, results in outlier_analysis['target_outliers'].items():
                results_data.append({
                    'Metode': method,
                    'Jumlah Outlier': results['outlier_count'],
                    'Persentase Outlier': f"{results['outlier_percentage']:.2f}%",
                    'Threshold': results['threshold']
                })
            
            results_df = pd.DataFrame(results_data)
            safe_display_dataframe(results_df, use_container_width=True)
            
            # Rekomendasi
            st.subheader("Rekomendasi")
            summary = outlier_analysis['summary']
            
            for rec in summary['recommendations']:
                if "High outlier percentage" in rec:
                    st.warning(rec)
                elif "Moderate outlier percentage" in rec:
                    st.info(rec)
                else:
                    st.success(rec)
            
            # Visualisasi
            st.subheader("Visualisasi Analisis Outlier")
            
            # Generate visualizations
            outlier_fig = outlier_system.visualize_outlier_analysis(
                df, 
                outlier_analysis, 
                target_column='Life expectancy'
            )
            
            st.plotly_chart(outlier_fig, use_container_width=True)
            
            # Penanganan outlier interaktif
            st.subheader("Penanganan Outlier Interaktif")
            
            handling_options = outlier_system.interactive_outlier_handling(
                df, 
                outlier_analysis, 
                target_column='Life expectancy'
            )
            
            # Tampilkan opsi penanganan
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Jumlah Outlier berdasarkan Metode:**")
                for method, stats in handling_options['outlier_counts'].items():
                    st.write(f"- {method}: {stats['count']} ({stats['percentage']:.2f}%)")
            
            with col2:
                st.write("**Analisis Dampak:**")
                for method, option in handling_options['removal_options'].items():
                    impact = option['impact']
                    st.write(f"- {method}: {impact['records_removed']} records, perubahan rata-rata: {impact['mean_change']:.3f}")
            
            # Opsi penghapusan outlier
            st.subheader("Opsi Penghapusan Outlier")
            
            removal_choice = st.selectbox(
                "Pilih strategi penanganan outlier:",
                ["Simpan semua outlier", "Hapus outlier menggunakan metode yang direkomendasikan", "Hapus outlier menggunakan metode tertentu"]
            )
            
            if removal_choice == "Hapus outlier menggunakan metode yang direkomendasikan":
                best_method = min(outlier_analysis['target_outliers'].keys(), 
                                 key=lambda x: outlier_analysis['target_outliers'][x]['outlier_count'])
                
                if st.button(f"Hapus outlier menggunakan {best_method}"):
                    outlier_indices = outlier_analysis['target_outliers'][best_method]['outlier_indices']
                    df_clean, df_removed, removal_summary = outlier_system.remove_outliers(
                        df, outlier_indices, best_method
                    )
                    
                    st.session_state.df_clean = df_clean
                    st.session_state.df_removed = df_removed
                    st.session_state.removal_summary = removal_summary
                    
                    st.success(f"✅ Menghapus {removal_summary['removed_count']} outlier menggunakan {best_method}")
                    
            elif removal_choice == "Hapus outlier menggunakan metode tertentu":
                method_choice = st.selectbox(
                    "Pilih metode:",
                    list(handling_options['detection_methods'])
                )
                
                if st.button(f"Hapus outlier menggunakan {method_choice}"):
                    outlier_indices = outlier_analysis['target_outliers'][method_choice]['outlier_indices']
                    df_clean, df_removed, removal_summary = outlier_system.remove_outliers(
                        df, outlier_indices, method_choice
                    )
                    
                    st.session_state.df_clean = df_clean
                    st.session_state.df_removed = df_removed
                    st.session_state.removal_summary = removal_summary
                    
                    st.success(f"✅ Menghapus {removal_summary['removed_count']} outlier menggunakan {method_choice}")
            
            # Tampilkan ringkasan penghapusan jika tersedia
            if 'removal_summary' in st.session_state:
                st.subheader("Ringkasan Penghapusan Outlier")
                
                summary = st.session_state.removal_summary
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Metode yang Digunakan", summary['method'])
                
                with col2:
                    st.metric("Record yang Dihapus", summary['removed_count'])
                
                with col3:
                    st.metric("Persentase Penghapusan", f"{summary['removal_percentage']:.2f}%")
                
                # Tampilkan perbandingan sebelum/sesudah
                if 'df_clean' in st.session_state:
                    st.subheader("Sebelum vs Sesudah Penghapusan Outlier")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Sebelum Penghapusan:**")
                        st.write(f"Records: {summary['original_count']}")
                        st.write(f"Rata-rata Harapan Hidup: {df['Life expectancy'].mean():.2f}")
                        st.write(f"Std Harapan Hidup: {df['Life expectancy'].std():.2f}")
                    
                    with col2:
                        st.write("**Sesudah Penghapusan:**")
                        st.write(f"Records: {summary['remaining_count']}")
                        st.write(f"Rata-rata Harapan Hidup: {st.session_state.df_clean['Life expectancy'].mean():.2f}")
                        st.write(f"Std Harapan Hidup: {st.session_state.df_clean['Life expectancy'].std():.2f}")
    
    # Detail outlier
    elif page == "📋 Detail Outlier":
        if 'outlier_analysis' in st.session_state:
            show_detailed_outlier_info(df, st.session_state['outlier_analysis'])
        else:
            st.warning("⚠️ Jalankan deteksi outlier terlebih dahulu di halaman 'Deteksi Outlier'.")
    
    # Analisis PCA
    elif page == "🔬 Analisis PCA":
        st.markdown("## Analisis Komponen Utama (PCA)")
        
        # Periksa apakah data yang sudah dibersihkan tersedia
        if 'df_clean' in st.session_state:
            df_to_use = st.session_state.df_clean
            st.info("Menggunakan dataset yang sudah dibersihkan dari outlier untuk analisis PCA")
        else:
            df_to_use = df
            st.info("Menggunakan dataset asli untuk analisis PCA")
        
        # Inisialisasi analisis PCA
        pca_analyzer = PCAAnalysis()
        pca_visualizer = PCAVisualizer()
        pca_feature_selector = PCAFeatureSelector()
        
        # Konfigurasi PCA
        st.subheader("Konfigurasi PCA")
        
        col1, col2 = st.columns(2)
        
        with col1:
            variance_threshold = st.slider(
                "Ambang Batas Varians yang Dijelaskan", 
                min_value=0.8, 
                max_value=0.99, 
                value=0.95, 
                step=0.01,
                help="Varians minimum yang akan dijelaskan oleh komponen yang dipilih"
            )
            
            max_components = st.number_input(
                "Komponen Maksimum", 
                min_value=2, 
                max_value=20, 
                value=10,
                help="Jumlah maksimum komponen yang akan dipertimbangkan"
            )
        
        with col2:
            top_features = st.number_input(
                "Fitur Teratas untuk Ditampilkan", 
                min_value=5, 
                max_value=20, 
                value=10,
                help="Jumlah fitur teratas yang akan ditampilkan dalam visualisasi"
            )
            
            include_target = st.checkbox(
                "Sertakan Target dalam Analisis", 
                value=True,
                help="Sertakan harapan hidup dalam analisis PCA untuk pewarnaan"
            )
        
        # Tombol Jalankan Analisis PCA
        if st.button("🔬 Jalankan Analisis PCA", type="primary"):
            with st.spinner("Melakukan analisis PCA..."):
                try:
                    # Siapkan data untuk PCA
                    X_scaled, feature_names, target_values = pca_analyzer.prepare_data_for_pca(
                        df_to_use, 
                        target_column='Life expectancy'
                    )
                    
                    # Lakukan analisis PCA
                    pca_results = pca_analyzer.perform_pca_analysis(
                        X_scaled, 
                        explained_variance_threshold=variance_threshold
                    )
                    
                    # Simpan di session state
                    st.session_state.pca_analyzer = pca_analyzer
                    st.session_state.pca_results = pca_results
                    st.session_state.feature_names = feature_names
                    st.session_state.target_values = target_values
                    
                    st.success("✅ Analisis PCA berhasil diselesaikan!")
                    
                except Exception as e:
                    st.error(f"❌ Error dalam analisis PCA: {e}")
                    st.info("Silakan periksa data Anda dan coba lagi.")
        
        # Tampilkan hasil PCA jika tersedia
        if 'pca_results' in st.session_state:
            pca_results = st.session_state.pca_results
            pca_analyzer = st.session_state.pca_analyzer
            
            # Ringkasan PCA
            st.subheader("Ringkasan PCA")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Fitur Asli", len(pca_results['feature_names']))
            
            with col2:
                st.metric("Komponen Terpilih", pca_results['n_components'])
            
            with col3:
                reduction_ratio = (1 - pca_results['n_components'] / len(pca_results['feature_names'])) * 100
                st.metric("Pengurangan Fitur", f"{reduction_ratio:.1f}%")
            
            with col4:
                st.metric("Varians Dijelaskan", f"{pca_results['total_variance_explained']:.1%}")
            
            # Buat tab untuk berbagai visualisasi PCA
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Scree Plot", 
                "🔥 Heatmap Loadings", 
                "🎯 Visualisasi 2D", 
                "📈 Pentingnya Fitur", 
                "📋 Detail Komponen"
            ])
            
            with tab1:
                st.subheader("Scree Plot - Analisis Varians yang Dijelaskan")
                
                scree_fig = pca_analyzer.create_scree_plot()
                if scree_fig:
                    st.plotly_chart(scree_fig, use_container_width=True)
                    
                    # Tambahkan interpretasi
                    st.markdown("""
                    **Interpretasi:**
                    - **Varians Individual**: Menunjukkan berapa banyak varians yang dijelaskan oleh setiap komponen utama
                    - **Varians Kumulatif**: Menunjukkan total varians yang dijelaskan saat komponen ditambahkan
                    - **Titik Siku**: Cari titik di mana menambahkan komponen lebih lanjut memberikan hasil yang menurun
                    - **Ambang Batas 95%**: Garis putus-putus oranye menunjukkan ambang batas varians 95%
                    """)
                else:
                    st.error("Tidak dapat membuat scree plot.")
            
            with tab2:
                st.subheader("Heatmap Loadings Komponen")
                
                loadings_fig = pca_analyzer.create_component_loadings_heatmap(top_features=top_features)
                if loadings_fig:
                    st.plotly_chart(loadings_fig, use_container_width=True)
                    
                    # Tambahkan interpretasi
                    st.markdown("""
                    **Interpretasi:**
                    - **Warna merah**: Loadings positif (korelasi positif dengan komponen)
                    - **Warna biru**: Loadings negatif (korelasi negatif dengan komponen)
                    - **Warna lebih gelap**: Hubungan yang lebih kuat
                    - **Warna lebih terang**: Hubungan yang lebih lemah
                    """)
                else:
                    st.error("Tidak dapat membuat heatmap loadings.")
            
            with tab3:
                st.subheader("Visualisasi PCA 2D")
                
                # Opsi warna
                color_option = st.selectbox(
                    "Warna berdasarkan:",
                    ["Harapan Hidup", "Komponen", "Tidak Ada"],
                    help="Pilih cara mewarnai titik data"
                )
                
                if color_option == "Harapan Hidup" and include_target:
                    target_for_plot = st.session_state.target_values
                    color_by = 'target'
                elif color_option == "Komponen":
                    target_for_plot = None
                    color_by = 'component'
                else:
                    target_for_plot = None
                    color_by = 'none'
                
                pca_2d_fig = pca_analyzer.create_2d_pca_visualization(
                    target_values=target_for_plot, 
                    color_by=color_by
                )
                
                if pca_2d_fig:
                    st.plotly_chart(pca_2d_fig, use_container_width=True)
                    
                    # Tambahkan interpretasi
                    st.markdown("""
                    **Interpretasi:**
                    - **Cluster**: Kelompok titik mungkin menunjukkan negara atau pola yang serupa
                    - **Outlier**: Titik yang jauh dari cluster utama mungkin merupakan kasus yang tidak biasa
                    - **Tren**: Jika diberi warna berdasarkan harapan hidup, cari pola dalam distribusi
                    - **Varians**: Persentase pada setiap sumbu menunjukkan berapa banyak varians yang dijelaskan komponen tersebut
                    """)
                else:
                    st.error("Tidak dapat membuat visualisasi PCA 2D.")
            
            with tab4:
                st.subheader("Pentingnya Fitur dalam PCA")
                
                importance_fig, importance_df = pca_analyzer.create_feature_importance_plot(
                    top_features=top_features
                )
                
                if importance_fig:
                    st.plotly_chart(importance_fig, use_container_width=True)
                    
                    # Tampilkan tabel pentingnya
                    st.write("**Tabel Pentingnya Fitur:**")
                    safe_display_dataframe(importance_df, use_container_width=True)
                    
                    # Tambahkan interpretasi
                    st.markdown("""
                    **Interpretasi:**
                    - **Pentingnya lebih tinggi**: Fitur yang berkontribusi lebih banyak pada komponen utama
                    - **Pentingnya lebih rendah**: Fitur yang berkontribusi lebih sedikit dan berpotensi dapat dihapus
                    - **Seleksi fitur**: Pertimbangkan untuk mempertahankan fitur dengan skor penting yang tinggi
                    """)
                else:
                    st.error("Tidak dapat membuat plot pentingnya fitur.")
            
            with tab5:
                st.subheader("Detail Komponen dan Interpretasi")
                
                # Dapatkan interpretasi komponen
                interpretation_fig = pca_visualizer.create_component_interpretation_table(
                    pca_analyzer, 
                    top_features=5
                )
                
                if interpretation_fig:
                    st.plotly_chart(interpretation_fig, use_container_width=True)
                else:
                    st.error("Tidak dapat membuat tabel interpretasi komponen.")
                
                # Ringkasan varians
                variance_fig = pca_visualizer.create_variance_summary_plot(pca_analyzer)
                if variance_fig:
                    st.plotly_chart(variance_fig, use_container_width=True)
                
                # Informasi komponen detail
                st.subheader("Informasi Komponen Detail")
                
                try:
                    for i in range(pca_results['n_components']):
                        with st.expander(f"Komponen {i+1} (PC{i+1})"):
                            # Dapatkan fitur yang berkontribusi teratas untuk komponen ini
                            loadings = pca_results['loadings'][i]
                            feature_names = pca_results['feature_names']
                            
                            # Urutkan fitur berdasarkan nilai loading absolut
                            feature_loading_pairs = list(zip(feature_names, loadings))
                            feature_loading_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
                            
                            # Tampilkan statistik komponen
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Varians Dijelaskan", f"{pca_results['explained_variance_ratio'][i]:.3f}")
                            with col2:
                                st.metric("Varians Kumulatif", f"{pca_results['cumulative_variance'][i]:.3f}")
                            
                            # Tampilkan fitur teratas dengan cara yang lebih terorganisir
                            st.write("**Fitur yang Berkontribusi Teratas:**")
                            
                            # Buat DataFrame untuk tampilan yang lebih baik
                            top_features_data = []
                            for j, (feature, loading) in enumerate(feature_loading_pairs[:top_features]):
                                top_features_data.append({
                                    'Peringkat': j + 1,
                                    'Fitur': feature,
                                    'Loading': f"{loading:.3f}",
                                    'Loading Absolut': f"{abs(loading):.3f}",
                                    'Arah': "🟢 Positif" if loading > 0 else "🔴 Negatif"
                                })
                            
                            if top_features_data:
                                top_features_df = pd.DataFrame(top_features_data)
                                safe_display_dataframe(top_features_df, use_container_width=True)
                                
                                # Tambahkan interpretasi
                                st.markdown("""
                                **Interpretasi:**
                                - **Loadings positif**: Fitur yang meningkat dengan komponen ini
                                - **Loadings negatif**: Fitur yang menurun dengan komponen ini
                                - **Nilai absolut lebih tinggi**: Fitur yang berkontribusi lebih banyak pada komponen ini
                                """)
                            else:
                                st.warning("Tidak ada fitur ditemukan untuk komponen ini.")
                                
                except Exception as e:
                    st.error(f"Error menampilkan detail komponen: {e}")
                    st.info("Silakan coba jalankan analisis PCA lagi.")
            
            # Seleksi Fitur PCA dan Perbandingan Model
            st.subheader("Seleksi Fitur PCA dan Perbandingan Model")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔍 Bandingkan Fitur Asli vs PCA", type="secondary"):
                    with st.spinner("Melatih model dengan fitur PCA..."):
                        try:
                            # Dapatkan fitur PCA
                            X_pca = pca_results['X_pca']
                            
                            # Latih model dengan fitur PCA
                            pca_results_models, pca_model_trainer = train_models(
                                X_pca, 
                                st.session_state.target_values, 
                                [f'PC{i+1}' for i in range(pca_results['n_components'])]
                            )
                            
                            # Simpan hasil model PCA
                            st.session_state.pca_results_models = pca_results_models
                            st.session_state.pca_model_trainer = pca_model_trainer
                            
                            st.success("✅ Model PCA berhasil dilatih!")
                            
                        except Exception as e:
                            st.error(f"❌ Error melatih model PCA: {e}")
            
            with col2:
                if st.button("📊 Tampilkan Perbandingan Performa", type="secondary"):
                    if 'pca_results_models' in st.session_state and 'results' in st.session_state:
                        # Buat plot perbandingan
                        comparison_fig = pca_visualizer.create_pca_performance_comparison(
                            st.session_state.results,
                            st.session_state.pca_results_models
                        )
                        
                        if comparison_fig:
                            st.plotly_chart(comparison_fig, use_container_width=True)
                            
                            # Tampilkan tabel perbandingan
                            st.subheader("Tabel Perbandingan Performa")
                            
                            comparison_data = []
                            for model_name in st.session_state.results.keys():
                                if model_name in st.session_state.pca_results_models:
                                    comparison_data.append({
                                        'Model': model_name,
                                        'R² Asli': f"{st.session_state.results[model_name]['test_r2']:.4f}",
                                        'R² PCA': f"{st.session_state.pca_results_models[model_name]['test_r2']:.4f}",
                                        'RMSE Asli': f"{st.session_state.results[model_name]['test_rmse']:.4f}",
                                        'RMSE PCA': f"{st.session_state.pca_results_models[model_name]['test_rmse']:.4f}",
                                        'Perubahan R²': f"{st.session_state.pca_results_models[model_name]['test_r2'] - st.session_state.results[model_name]['test_r2']:+.4f}",
                                        'Perubahan RMSE': f"{st.session_state.pca_results_models[model_name]['test_rmse'] - st.session_state.results[model_name]['test_rmse']:+.4f}"
                                    })
                            
                            if comparison_data:
                                comparison_df = pd.DataFrame(comparison_data)
                                safe_display_dataframe(comparison_df, use_container_width=True)
                                
                                # Berikan rekomendasi
                                st.subheader("Rekomendasi")
                                
                                for row in comparison_data:
                                    r2_change = float(row['Perubahan R²'])
                                    if r2_change > 0.01:
                                        st.success(f"✅ {row['Model']}: PCA meningkatkan performa secara signifikan")
                                    elif r2_change > -0.01:
                                        st.info(f"ℹ️ {row['Model']}: PCA mempertahankan performa yang serupa")
                                    else:
                                        st.warning(f"⚠️ {row['Model']}: PCA mengurangi performa")
                    else:
                        st.warning("Silakan latih model asli dan PCA terlebih dahulu.")
    
    # Pelatihan model
    elif page == "🤖 Pelatihan Model":
        st.markdown("## Pelatihan Model")
        
        # Periksa apakah data yang sudah dibersihkan tersedia
        if 'df_clean' in st.session_state:
            df_to_use = st.session_state.df_clean
            st.info("Menggunakan dataset yang sudah dibersihkan dari outlier untuk pelatihan model")
        else:
            df_to_use = df
            st.info("Menggunakan dataset asli untuk pelatihan model")
        
        # Opsi preprocessing
        st.subheader("Opsi Preprocessing")
        
        col1, col2 = st.columns(2)
        
        with col1:
            imputation_strategy = st.selectbox(
                "Strategi Imputasi",
                ["median", "mean", "most_frequent"]
            )
        
        with col2:
            test_size = st.slider("Ukuran Test", 0.1, 0.5, 0.2, 0.05)
            random_state = st.number_input("Random State", value=42, min_value=0)
        
        # Tombol train models
        if st.button("🚀 Latih Model", type="primary"):
            with st.spinner("Melatih model..."):
                # Preprocess data
                X_scaled, y, feature_names, df_processed = preprocessor.prepare_final_dataset(df_to_use)
                
                # Train models
                results, model_trainer = train_models(X_scaled, y, feature_names)
                
                # Simpan di session state
                st.session_state.results = results
                st.session_state.model_trainer = model_trainer
                st.session_state.feature_names = feature_names
                st.session_state.X_scaled = X_scaled
                st.session_state.y = y
                st.session_state.df_processed = df_processed
                st.session_state.preprocessor = preprocessor
                
                st.success("✅ Model berhasil dilatih!")
        
        # Tampilkan hasil jika tersedia
        if 'results' in st.session_state:
            st.subheader("Performa Model")
            
            # Buat perbandingan performa
            results = st.session_state.results
            
            # Tabel metrik performa
            performance_data = []
            for name, result in results.items():
                performance_data.append({
                    'Model': name,
                    'Test R²': f"{result['test_r2']:.4f}",
                    'Test RMSE': f"{result['test_rmse']:.4f}",
                    'CV R²': f"{result['cv_mean']:.4f} (±{result['cv_std']:.4f})"
                })
            
            performance_df = pd.DataFrame(performance_data)
            safe_display_dataframe(performance_df, use_container_width=True)
            
            # Visualisasi performa
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Skor R²', 'Skor RMSE'),
                specs=[[{"type": "bar"}, {"type": "bar"}]]
            )
            
            model_names = list(results.keys())
            test_r2 = [results[name]['test_r2'] for name in model_names]
            test_rmse = [results[name]['test_rmse'] for name in model_names]
            
            fig.add_trace(
                go.Bar(x=model_names, y=test_r2, name='Test R²'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(x=model_names, y=test_rmse, name='Test RMSE'),
                row=1, col=2
            )
            
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    # Hasil model
    elif page == "📊 Hasil":
        st.markdown("## Hasil Model")
        
        if 'results' not in st.session_state:
            st.warning("⚠️ Latih model terlebih dahulu di halaman 'Pelatihan Model'.")
            return
        
        results = st.session_state.results
        model_trainer = st.session_state.model_trainer
        
        # Info model terbaik
        best_model_name = model_trainer.best_model_name
        st.success(f"🏆 **Model Terbaik**: {best_model_name}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Test R² Terbaik", f"{results[best_model_name]['test_r2']:.4f}")
        
        with col2:
            st.metric("Test RMSE Terbaik", f"{results[best_model_name]['test_rmse']:.4f}")
        
        with col3:
            st.metric("CV R² Terbaik", f"{results[best_model_name]['cv_mean']:.4f}")
        
        # Plot aktual vs prediksi
        st.subheader("Nilai Aktual vs Prediksi")
        
        y_test = results[best_model_name]['y_test']
        y_pred = results[best_model_name]['y_pred_test']
        
        fig = px.scatter(x=y_test, y=y_pred, 
                        title=f"Aktual vs Prediksi - {best_model_name}",
                        labels={'x': 'Nilai Aktual', 'y': 'Nilai Prediksi'})
        
        # Tambahkan garis prediksi sempurna
        min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
        fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], 
                                mode='lines', name='Prediksi Sempurna',
                                line=dict(color='red', dash='dash')))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Pentingnya fitur
        st.subheader("Pentingnya Fitur")
        
        importance_df = model_trainer.analyze_feature_importance()
        
        if importance_df is not None:
            # Top 10 fitur
            top_10 = importance_df.head(10)
            
            fig = px.bar(top_10, x='Importance', y='Feature', orientation='h',
                        title="10 Fitur Terpenting")
            st.plotly_chart(fig, use_container_width=True)
            
            # Tabel pentingnya fitur
            safe_display_dataframe(importance_df, use_container_width=True)
    
    # Prediksi
    elif page == "🔮 Prediksi":
        st.markdown("## Buat Prediksi")
        
        if 'model_trainer' not in st.session_state:
            st.warning("⚠️ Latih model terlebih dahulu di halaman 'Pelatihan Model'.")
            return
        
        model_trainer = st.session_state['model_trainer']
        feature_names = st.session_state['feature_names']
        
        st.subheader("Parameter Input")
        
        # Buat form input
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                adult_mortality = st.number_input("Kematian Dewasa", value=100.0, min_value=0.0)
                infant_deaths = st.number_input("Kematian Bayi", value=10.0, min_value=0.0)
                alcohol = st.number_input("Konsumsi Alkohol", value=5.0, min_value=0.0)
                percentage_expenditure = st.number_input("Persentase Pengeluaran", value=1000.0, min_value=0.0)
                hepatitis_b = st.number_input("Cakupan Hepatitis B", value=80.0, min_value=0.0, max_value=100.0)
                measles = st.number_input("Kasus Campak", value=100.0, min_value=0.0)
                bmi = st.number_input("BMI", value=25.0, min_value=10.0, max_value=50.0)
                under_five_deaths = st.number_input("Kematian Balita", value=15.0, min_value=0.0)
                polio = st.number_input("Cakupan Polio", value=80.0, min_value=0.0, max_value=100.0)
                total_expenditure = st.number_input("Total Pengeluaran", value=5.0, min_value=0.0)
            
            with col2:
                diphtheria = st.number_input("Cakupan Difteri", value=80.0, min_value=0.0, max_value=100.0)
                hiv_aids = st.number_input("HIV/AIDS", value=0.1, min_value=0.0)
                gdp = st.number_input("GDP", value=5000.0, min_value=0.0)
                population = st.number_input("Populasi", value=1000000.0, min_value=0.0)
                thinness_1_19 = st.number_input("Kekurusan 1-19 Tahun", value=5.0, min_value=0.0)
                thinness_5_9 = st.number_input("Kekurusan 5-9 Tahun", value=5.0, min_value=0.0)
                income_composition = st.number_input("Komposisi Pendapatan", value=0.5, min_value=0.0, max_value=1.0)
                schooling = st.number_input("Pendidikan", value=12.0, min_value=0.0, max_value=25.0)
                year = st.number_input("Tahun", value=2015, min_value=2000, max_value=2030)
                status = st.selectbox("Status", ["Developing", "Developed"])
                region = st.selectbox("Wilayah", ["Africa", "Asia", "Europe", "North America", "South America", "Oceania"])
            
            submitted = st.form_submit_button("🔮 Prediksi Harapan Hidup", type="primary")
        
        if submitted:
            try:
                # Buat DataFrame dengan data input
                input_data = pd.DataFrame({
                    'Adult Mortality': [adult_mortality],
                    'infant deaths': [infant_deaths],
                    'Alcohol': [alcohol],
                    'percentage expenditure': [percentage_expenditure],
                    'Hepatitis B': [hepatitis_b],
                    'Measles': [measles],
                    'BMI': [bmi],
                    'under-five deaths': [under_five_deaths],
                    'Polio': [polio],
                    'Total expenditure': [total_expenditure],
                    'Diphtheria': [diphtheria],
                    'HIV/AIDS': [hiv_aids],
                    'GDP': [gdp],
                    'Population': [population],
                    'thinness 1-19 years': [thinness_1_19],
                    'thinness 5-9 years': [thinness_5_9],
                    'Income composition of resources': [income_composition],
                    'Schooling': [schooling],
                    'Year': [year],
                    'Status': [status],
                    'Region': [region]
                })
                
                # Dapatkan preprocessor dari session state
                if 'preprocessor' in st.session_state:
                    preprocessor = st.session_state.preprocessor
                else:
                    # Jika preprocessor tidak ada di session state, buat yang baru
                    preprocessor = DataPreprocessor()
                
                # Terapkan pipeline preprocessing yang sama seperti saat training
                # Langkah 1: Feature engineering
                input_engineered = preprocessor.feature_engineering(input_data)
                
                # Langkah 2: Encode fitur kategorikal
                input_encoded = preprocessor.encode_categorical_features(input_engineered)
                
                # Langkah 3: Tangani nilai yang hilang (jika ada)
                input_imputed = preprocessor.handle_missing_values(input_encoded)
                
                # Langkah 4: Siapkan fitur untuk prediksi (sama seperti training)
                # Dapatkan kolom fitur yang digunakan saat training
                feature_columns = [col for col in input_imputed.columns if col not in 
                                  ['Country', 'Year', 'Status', 'Region', 'Life expectancy']]
                
                # Tambahkan fitur yang direkayasa yang mungkin hilang
                if 'is_developed' not in input_imputed.columns:
                    input_imputed['is_developed'] = (input_imputed['Status'] == 'Developed').astype(int)
                
                if 'year_normalized' not in input_imputed.columns:
                    input_imputed['year_normalized'] = (input_imputed['Year'] - 2000) / 15
                
                # Buat fitur komposit jika tidak ada
                if 'health_expenditure_per_capita' not in input_imputed.columns:
                    input_imputed['health_expenditure_per_capita'] = (
                        input_imputed['percentage expenditure'] / input_imputed['Population']
                    ).fillna(0)
                
                if 'total_mortality_rate' not in input_imputed.columns:
                    input_imputed['total_mortality_rate'] = (
                        input_imputed['Adult Mortality'] + input_imputed['infant deaths'] + 
                        input_imputed['under-five deaths']
                    )
                
                if 'vaccination_coverage' not in input_imputed.columns:
                    input_imputed['vaccination_coverage'] = (
                        input_imputed['Hepatitis B'] + input_imputed['Polio'] + input_imputed['Diphtheria']
                    ) / 3
                
                if 'thinness_composite' not in input_imputed.columns:
                    input_imputed['thinness_composite'] = (
                        input_imputed['thinness 1-19 years'] + input_imputed['thinness 5-9 years']
                    ) / 2
                
                if 'gdp_per_capita' not in input_imputed.columns:
                    input_imputed['gdp_per_capita'] = (
                        input_imputed['GDP'] / input_imputed['Population']
                    ).fillna(0)
                
                # Dapatkan kolom fitur akhir (tidak termasuk target dan kategorikal)
                final_feature_columns = [col for col in input_imputed.columns if col not in 
                                        ['Country', 'Year', 'Status', 'Region', 'Life expectancy']]
                
                # Pilih hanya fitur yang digunakan saat training
                X_input = input_imputed[final_feature_columns].select_dtypes(include=[np.number])
                
                # Scale fitur menggunakan scaler yang sama dari training
                if hasattr(preprocessor, 'scaler') and preprocessor.is_fitted:
                    X_scaled = preprocessor.scaler.transform(X_input)
                else:
                    # Jika scaler tidak fitted, gunakan yang dari model trainer
                    if hasattr(model_trainer, 'scaler'):
                        X_scaled = model_trainer.scaler.transform(X_input)
                    else:
                        # Fallback: gunakan fitur apa adanya
                        X_scaled = X_input.values
                
                # Buat prediksi menggunakan model terbaik
                best_model = model_trainer.best_model
                prediction = best_model.predict(X_scaled)[0]
                
                # Tampilkan informasi debug (bisa dihapus di production)
                with st.expander("🔧 Informasi Debug"):
                    st.write(f"**Model yang Digunakan**: {model_trainer.best_model_name}")
                    st.write(f"**Jumlah Fitur**: {X_scaled.shape[1]}")
                    st.write(f"**Nama Fitur**: {list(X_input.columns)}")
                    st.write(f"**Bentuk Fitur Scaled**: {X_scaled.shape}")
                
                # Tampilkan prediksi
                st.success(f"🎯 **Prediksi Harapan Hidup**: {prediction:.1f} tahun")
                
                # Tambahkan konteks
                if prediction < 60:
                    st.warning("⚠️ Harapan hidup rendah - mungkin menunjukkan tantangan kesehatan")
                elif prediction > 80:
                    st.success("✅ Harapan hidup tinggi - indikator kesehatan yang baik")
                else:
                    st.info("ℹ️ Harapan hidup sedang")
                
                # Tampilkan ringkasan input
                st.subheader("Ringkasan Input")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Indikator Kesehatan:**")
                    st.write(f"- Kematian Dewasa: {adult_mortality}")
                    st.write(f"- Kematian Bayi: {infant_deaths}")
                    st.write(f"- Kematian Balita: {under_five_deaths}")
                    st.write(f"- BMI: {bmi}")
                    st.write(f"- HIV/AIDS: {hiv_aids}")
                
                with col2:
                    st.write("**Ekonomi & Sosial:**")
                    st.write(f"- GDP: {gdp:,.0f}")
                    st.write(f"- Pendidikan: {schooling} tahun")
                    st.write(f"- Status: {status}")
                    st.write(f"- Wilayah: {region}")
                    st.write(f"- Tahun: {year}")
                
            except Exception as e:
                st.error(f"❌ Error dalam membuat prediksi: {e}")
                st.info("Pastikan model sudah dilatih terlebih dahulu dan semua nilai input valid.")
    
    # Tentang
    elif page == "📋 Tentang":
        st.markdown("## Tentang Aplikasi")
        
        st.markdown("""
        ### Ikhtisar
        Aplikasi Prediksi Harapan Hidup ini menggunakan machine learning untuk memprediksi harapan hidup 
        berdasarkan berbagai indikator kesehatan, ekonomi, dan sosial dari dataset Organisasi Kesehatan Dunia (WHO).
        
        ### Fitur yang Diimplementasikan
        
        #### 1. Preprocessing Data
        - **Penanganan Nilai Hilang**: Strategi imputasi (median, mean, most frequent)
        - **Pembersihan Data**: Menangani inkonsistensi, konversi tipe data
        - **Rekayasa Fitur**: Membuat fitur komposit, encoding variabel kategorikal
        - **Scaling**: Standardisasi fitur numerik
        
        #### 2. Deteksi Outlier Komprehensif
        - **Berbagai Metode Deteksi**: Z-score, IQR, Modified Z-score, Isolation Forest
        - **Analisis Multivariat**: Isolation Forest, Local Outlier Factor, Elliptic Envelope
        - **Penanganan Interaktif**: Pilih metode deteksi dan strategi penghapusan
        - **Analisis Dampak**: Menilai efek penghapusan outlier pada statistik data
        - **Visualisasi**: Plot komprehensif menunjukkan distribusi outlier dan dampak
        - **Informasi Detail**: Tampilkan data yang dikenali sebagai outlier
        
        #### 3. Analisis Komponen Utama (PCA)
        - **Reduksi Dimensi**: Mengurangi jumlah fitur sambil mempertahankan varians yang signifikan
        - **Analisis Varians**: Scree plot untuk menentukan jumlah komponen optimal
        - **Visualisasi Komponen**: Heatmap loadings dan plot 2D untuk interpretasi
        - **Seleksi Fitur**: Identifikasi fitur yang paling berkontribusi pada komponen utama
        - **Perbandingan Model**: Bandingkan performa model dengan fitur asli vs fitur PCA
        - **Interpretasi Komponen**: Analisis detail setiap komponen utama dan maknanya
        
        #### 4. Pelatihan Model
        - **Berbagai Algoritma**: Regresi Linear, Ridge, Lasso, Elastic Net, Random Forest, Gradient Boosting
        - **Validasi Silang**: 5-fold cross-validation untuk evaluasi yang robust
        - **Tuning Hyperparameter**: Grid search untuk parameter optimal
        - **Perbandingan Performa**: Evaluasi model side-by-side
        
        #### 5. Evaluasi Model
        - **Metrik Performa**: R², RMSE, MAE
        - **Perbandingan Model**: Analisis performa side-by-side
        - **Pentingnya Fitur**: Identifikasi prediktor kunci
        - **Analisis Residual**: Periksa asumsi model
        
        #### 6. Visualisasi
        - **Distribusi Data**: Histogram dan box plot
        - **Analisis Korelasi**: Heatmap dan scatter plot
        - **Analisis Outlier**: Visualisasi deteksi outlier komprehensif
        - **Performa Model**: Grafik perbandingan dan plot aktual vs prediksi
        - **Dashboard Interaktif**: Visualisasi berbasis Plotly
        
        ### Detail Teknis
        
        #### Dataset
        - **Sumber**: Organisasi Kesehatan Dunia (WHO)
        - **Ukuran**: ~2,900 record
        - **Fitur**: 22 variabel termasuk indikator kesehatan, ekonomi, dan sosial
        - **Periode Waktu**: 2000-2015
        - **Negara**: 193 negara
        
        #### Metode Deteksi Outlier
        1. **Z-score**: Deteksi berbasis standar deviasi
        2. **IQR**: Metode rentang interkuartil
        3. **Modified Z-score**: Robust terhadap nilai ekstrem
        4. **Isolation Forest**: Deteksi berbasis machine learning
        5. **Local Outlier Factor**: Deteksi berbasis densitas
        6. **Elliptic Envelope**: Deteksi outlier statistik
        
        #### Model yang Digunakan
        1. **Regresi Linear**: Model baseline
        2. **Ridge Regression**: Regularisasi L2 untuk multikolinearitas
        3. **Lasso Regression**: Regularisasi L1 untuk seleksi fitur
        4. **Elastic Net**: Kombinasi regularisasi L1 dan L2
        5. **Random Forest**: Metode ensemble untuk hubungan non-linear
        6. **Gradient Boosting**: Metode ensemble lanjutan
        
        #### Fitur Utama
        - **Kematian Dewasa**: Kematian per 1000 populasi
        - **Kematian Bayi**: Kematian per 1000 kelahiran hidup
        - **Alkohol**: Konsumsi alkohol per kapita
        - **GDP**: Produk Domestik Bruto per kapita
        - **Pendidikan**: Rata-rata tahun pendidikan
        - **BMI**: Indeks Massa Tubuh
        - **Cakupan Vaksinasi**: Hepatitis B, Polio, Difteri
        - **Indikator Ekonomi**: Komposisi pendapatan, pengeluaran
        
        ### Instruksi Penggunaan
        
        1. **Analisis Data**: Jelajahi dataset, periksa distribusi dan korelasi
        2. **Deteksi Outlier**: Gunakan deteksi outlier komprehensif dengan berbagai metode
        3. **Detail Outlier**: Lihat informasi detail tentang outlier yang terdeteksi
        4. **Analisis PCA**: Lakukan analisis komponen utama untuk reduksi dimensi dan seleksi fitur
        5. **Pelatihan Model**: Konfigurasi opsi preprocessing dan latih model
        6. **Hasil**: Lihat performa model dan pentingnya fitur
        7. **Prediksi**: Buat prediksi pada data baru (interface yang disederhanakan)
        
        ### Teknologi yang Digunakan
        - **Python**: Bahasa pemrograman utama
        - **Pandas**: Manipulasi dan analisis data
        - **NumPy**: Komputasi numerik
        - **Scikit-learn**: Algoritma machine learning
        - **Matplotlib/Seaborn**: Visualisasi statis
        - **Plotly**: Visualisasi interaktif
        - **Streamlit**: Framework aplikasi web
        
        ### Peningkatan Masa Depan
        - Update data real-time
        - Model yang lebih lanjutan (Neural Networks, XGBoost)
        - Visualisasi geografis
        - Analisis deret waktu
        - Integrasi API untuk prediksi dunia nyata
        
        ### Kontak
        Aplikasi ini dikembangkan sebagai proyek machine learning komprehensif untuk prediksi harapan hidup.
        """)

if __name__ == "__main__":
    main()