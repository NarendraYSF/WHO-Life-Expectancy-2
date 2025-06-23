# Aplikasi Prediksi Harapan Hidup

Aplikasi Python komprehensif untuk memprediksi harapan hidup berdasarkan berbagai indikator kesehatan, ekonomi, dan sosial menggunakan multiple machine learning models dan teknik analisis data tingkat lanjut.

## 🎯 Ikhtisar Proyek

Aplikasi ini menganalisis dataset Harapan Hidup Organisasi Kesehatan Dunia (WHO) untuk memprediksi harapan hidup menggunakan berbagai indikator kesehatan, ekonomi, dan sosial. Proyek ini mengimplementasikan pipeline machine learning lengkap dari preprocessing data hingga deployment model dengan antarmuka web interaktif.


### 🌍 **Visualisasi Peta Interaktif**
- **Marker Interaktif**: Marker negara yang dapat diklik dengan popup detail
- **Peta Choropleth**: Negara yang diberi kode warna berdasarkan nilai metrik
- **Heatmap**: Visualisasi kepadatan yang menunjukkan konsentrasi nilai
- **Statistik Geografis**: Analisis regional dan peringkat negara
- **Tren Temporal**: Analisis time series yang menunjukkan perubahan dari tahun ke tahun
- **Multiple Layer Peta**: Berbagai gaya tile (OpenStreetMap, CartoDB)

### 🔍 **Sistem Deteksi Outlier Komprehensif**
- **Multiple Metode Deteksi**: Z-score, IQR, Modified Z-score, Isolation Forest
- **Analisis Multivariat**: Local Outlier Factor, Elliptic Envelope
- **Penanganan Interaktif**: Pilih metode deteksi dan strategi penghapusan
- **Analisis Dampak**: Menilai efek penghapusan outlier pada statistik data
- **Informasi Detail**: Menunjukkan data apa yang dikenali sebagai outlier
- **Visualisasi**: Plot komprehensif yang menunjukkan distribusi outlier dan dampaknya

### 🤖 **Peningkatan Pelatihan & Prediksi Model**
- **Prediksi Real-time**: Prediksi akurat menggunakan model yang telah dilatih
- **Pipeline Preprocessing**: Transformasi data yang konsisten untuk prediksi
- **Persistence Model**: Simpan dan muat model yang telah dilatih
- **Feature Engineering**: Pembuatan dan seleksi fitur tingkat lanjut
- **Cross-validation**: Evaluasi model yang robust

### 📊 **Analisis Data Tingkat Lanjut**
- **Analisis Fitur vs Target**: Scatter plot interaktif dengan garis regresi kustom
- **Statistik Negara**: Analisis regional dan per negara
- **Analisis Korelasi**: Matriks korelasi komprehensif
- **Penilaian Kualitas Data**: Analisis missing value dan validasi data

### 🎨 **Peningkatan Antarmuka Pengguna**
- **Desain Modern**: Antarmuka yang bersih dan responsif dengan CSS kustom
- **Tab Navigasi**: Bagian yang terorganisir untuk berbagai fungsionalitas
- **Kontrol Interaktif**: Penyesuaian parameter dinamis
- **Update Real-time**: Visualisasi data dan analisis langsung
- **Responsif Mobile**: Berfungsi di desktop dan perangkat mobile

## 📁 Struktur Proyek

```
WHO Life Expectancy/
├── Dataset/
│   └── WHO Life Expectancy Descriptive Statistics - Raw Data.csv
├── __pycache__/                     # File cache Python
├── data_preprocessing.py            # Modul preprocessing data
├── model_training.py                # Pelatihan dan evaluasi model
├── visualization.py                 # Fungsi visualisasi
├── outlier_detection.py             # Deteksi outlier komprehensif
├── streamlit_app.py                # Aplikasi web Streamlit utama
├── Main.py                         # Aplikasi command-line
├── requirements.txt                # Dependensi Python
├── README.md                       # Dokumentasi proyek
└── Berbagai file test dan demo     # Utilitas tambahan
```

## 🚀 Instalasi

### Prasyarat
- Python 3.8 atau lebih tinggi
- pip (Python package installer)

### Langkah 1: Clone Repository
```bash
git clone <repository-url>
cd "WHO Life Expectancy"
```

### Langkah 2: Install Dependensi
```bash
pip install -r requirements.txt
```

### Langkah 3: Verifikasi Instalasi
```bash
python -c "import pandas, numpy, sklearn, streamlit, plotly, folium; print('Semua dependensi berhasil diinstal!')"
```

## 📖 Penggunaan

### Aplikasi Web (Direkomendasikan)

Jalankan aplikasi web Streamlit:
```bash
streamlit run streamlit_app.py
```

Aplikasi web menyediakan:

#### 🏠 **Halaman Beranda**
- Ikhtisar proyek dan fitur
- Informasi dataset dan statistik cepat
- Navigasi ke semua bagian

#### 📈 **Analisis Data**
- Ikhtisar dataset dan analisis missing value
- Analisis korelasi dengan heatmap interaktif
- Analisis fitur vs target dengan garis regresi kustom
- Statistik negara dan perbandingan regional

#### 🌍 **Visualisasi Peta** *(BARU)*
- **Marker Interaktif**: Klik negara untuk melihat informasi detail
- **Peta Choropleth**: Distribusi global dengan kode warna
- **Heatmap**: Visualisasi kepadatan metrik
- **Statistik Geografis**: Analisis regional dan peringkat
- **Tren Temporal**: Analisis time series berdasarkan region
- **Opsi Filtering**: Berdasarkan tahun, region, dan metrik

#### 🔍 **Deteksi Outlier**
- Multiple metode deteksi (Z-score, IQR, Isolation Forest, dll.)
- Penyesuaian parameter interaktif
- Analisis outlier komprehensif
- Penilaian dampak penghapusan outlier
- Identifikasi outlier visual

#### 📋 **Detail Outlier**
- Informasi detail tentang outlier yang terdeteksi
- Analisis outlier per negara
- Pemeriksaan outlier multivariat
- Analisis dampak penghapusan outlier

#### 🤖 **Pelatihan Model**
- Konfigurasi opsi preprocessing
- Latih multiple model dengan cross-validation
- Monitoring performa real-time
- Perbandingan dan seleksi model

#### 📊 **Hasil**
- Metrik performa model
- Plot actual vs predicted
- Analisis importance fitur
- Identifikasi model terbaik

#### 🔮 **Prediksi**
- Antarmuka prediksi interaktif
- Prediksi harapan hidup real-time
- Validasi input dan saran
- Konteks dan interpretasi prediksi

#### 📋 **Tentang**
- Dokumentasi proyek komprehensif
- Detail teknis dan metodologi
- Instruksi penggunaan dan troubleshooting

### Aplikasi Command Line

Jalankan aplikasi utama:
```bash
python Main.py
```

## 📊 Informasi Dataset

### Sumber
Dataset Harapan Hidup Organisasi Kesehatan Dunia (WHO)

### Fitur
- **22 Variabel**: Indikator kesehatan, ekonomi, dan sosial
- **2.929 Record**: Observasi negara-tahun
- **193 Negara**: Cakupan global
- **16 Tahun**: Periode waktu 2000-2015

### Variabel Utama
- **Target**: Harapan hidup (tahun)
- **Indikator Kesehatan**: Mortalitas dewasa, kematian bayi, BMI, cakupan vaksinasi
- **Indikator Ekonomi**: GDP, komposisi pendapatan, pengeluaran
- **Indikator Sosial**: Pendidikan, konsumsi alkohol, ketipisan
- **Variabel Kategorikal**: Negara, Status (Maju/Berkembang), Region

## 🔍 Pipeline Preprocessing Data

### 1. Pembersihan Data
- Menangani format angka Eropa (koma sebagai pemisah desimal)
- Konversi tipe data yang sesuai
- Menghapus atau menangani nilai yang tidak konsisten
- Penanganan kompatibilitas PyArrow

### 2. Imputasi Missing Value
- **Variabel Numerik**: Imputasi median (robust terhadap outlier)
- **Variabel Kategorikal**: Imputasi nilai paling sering
- **Persentase Missing**: Analisis dan pelaporan

### 3. Feature Engineering
- **Fitur Biner**: Konversi Status ke biner (Maju/Berkembang)
- **Fitur Komposit**: Pengeluaran kesehatan per kapita, total tingkat mortalitas
- **Fitur Normalisasi**: Normalisasi tahun, cakupan vaksinasi
- **Indikator Ekonomi**: GDP per kapita, komposit ketipisan

### 4. Encoding Kategorikal
- **Label Encoding**: Konversi Status dan Region ke nilai numerik
- **One-hot Encoding**: Pendekatan alternatif untuk variabel kategorikal

### 5. Scaling Fitur
- **StandardScaler**: Normalisasi fitur numerik ke mean nol dan variance unit
- **Robust Scaling**: Alternatif untuk scaling yang sensitif terhadap outlier

## 🤖 Proses Pelatihan Model

### 1. Pembagian Data
- **Train/Test Split**: Pembagian 80/20 untuk evaluasi model
- **Cross-Validation**: 5-fold CV untuk estimasi performa yang robust
- **Random State**: Tetap untuk reproduktibilitas

### 2. Pelatihan Model
- **Multiple Algoritma**: Latih 6 model berbeda
- **Hyperparameter Tuning**: Grid search untuk parameter optimal
- **Performance Tracking**: Monitor metrik pelatihan dan validasi

### 3. Evaluasi Model
- **Metrik**: R², RMSE, MAE untuk penilaian komprehensif
- **Cross-Validation**: Mean dan standard deviation dari skor CV
- **Perbandingan Model**: Analisis performa berdampingan

### 4. Importance Fitur
- **Model Linear**: Magnitude koefisien
- **Model Tree-based**: Skor importance fitur
- **Ranking**: Urutkan fitur berdasarkan importance

## 🌍 Fitur Visualisasi Peta

### 1. Peta Interaktif
- **Peta Berbasis Folium**: Peta web interaktif dengan multiple tile layer
- **Marker Clustering**: Mengelompokkan marker terdekat untuk performa lebih baik
- **Popup Kustom**: Informasi detail negara saat diklik
- **Kode Warna**: Warna berbeda untuk negara maju vs berkembang

### 2. Jenis Peta
- **Marker Interaktif**: Marker yang dapat diklik dengan informasi detail
- **Peta Choropleth**: Negara dengan kode warna berdasarkan nilai metrik
- **Heatmap**: Visualisasi kepadatan yang menunjukkan konsentrasi nilai

### 3. Analisis Geografis
- **Statistik Regional**: Rata-rata metrik berdasarkan region
- **Peringkat Negara**: Negara teratas dan terbawah berdasarkan metrik
- **Tren Temporal**: Perubahan dari waktu ke waktu berdasarkan region
- **Analisis Pertumbuhan**: Tingkat pertumbuhan tahunan berdasarkan region

### 4. Kontrol Interaktif
- **Pemilihan Metrik**: Pilih metrik numerik apapun dari dataset
- **Filtering Tahun**: Filter berdasarkan tahun tertentu atau lihat semua tahun
- **Filtering Region**: Fokus pada region tertentu atau lihat data global
- **Toggle Jenis Peta**: Beralih antara berbagai gaya visualisasi

## 🔍 Sistem Deteksi Outlier

### 1. Metode Deteksi
- **Z-score**: Deteksi berbasis standard deviation
- **IQR**: Metode interquartile range
- **Modified Z-score**: Robust terhadap nilai ekstrem
- **Isolation Forest**: Deteksi berbasis machine learning
- **Local Outlier Factor**: Deteksi berbasis kepadatan
- **Elliptic Envelope**: Deteksi outlier statistik

### 2. Fitur Analisis
- **Target Outliers**: Deteksi outlier harapan hidup
- **Feature Outliers**: Analisis outlier fitur individual
- **Multivariate Outliers**: Pola outlier kompleks
- **Analisis Dampak**: Efek penghapusan outlier pada statistik

### 3. Penanganan Interaktif
- **Pemilihan Metode**: Pilih metode deteksi
- **Penyesuaian Parameter**: Kustomisasi threshold dan parameter
- **Opsi Penghapusan**: Pilih strategi penanganan outlier
- **Analisis Visual**: Plot interaktif untuk identifikasi outlier

## 📈 Fitur Visualisasi

### 1. Eksplorasi Data
- **Plot Distribusi**: Histogram dan density plot untuk semua variabel
- **Matriks Korelasi**: Heatmap yang menunjukkan hubungan variabel
- **Scatter Plot**: Hubungan target vs fitur dengan garis regresi kustom
- **Analisis Negara**: Visualisasi regional dan per negara

### 2. Performa Model
- **Chart Perbandingan**: Bar plot yang membandingkan metrik model
- **Actual vs Predicted**: Scatter plot dengan garis prediksi sempurna
- **Analisis Residual**: Plot residual dan distribusi

### 3. Importance Fitur
- **Bar Chart**: Horizontal bar plot untuk ranking fitur
- **Plot Interaktif**: Visualisasi interaktif berbasis Plotly

### 4. Analisis Outlier
- **Visualisasi Deteksi**: Sorot outlier dalam distribusi
- **Analisis Dampak**: Perbandingan sebelum/sesudah penghapusan outlier
- **Laporan Detail**: Informasi outlier komprehensif

## 🎯 Performa Model

### Hasil yang Diharapkan
Berdasarkan karakteristik dataset, metrik performa tipikal meliputi:

- **Skor R²**: 0.75 - 0.85 (tergantung model dan preprocessing)
- **RMSE**: 3-5 tahun (root mean square error)
- **MAE**: 2-4 tahun (mean absolute error)

### Model Berperforma Terbaik
1. **Gradient Boosting**: Biasanya skor R² tertinggi
2. **Random Forest**: Keseimbangan performa dan interpretabilitas yang baik
3. **Elastic Net**: Terbaik di antara model linear dengan regularisasi

### Prediktor Utama
Fitur penting yang umum meliputi:
- Mortalitas Dewasa
- GDP per kapita
- Pendidikan
- BMI
- Cakupan vaksinasi
- Indikator ekonomi

## 🔧 Fitur Teknis

### 1. Kompatibilitas PyArrow
- **Environment Variables**: Nonaktifkan PyArrow untuk mencegah masalah kompatibilitas
- **Alternative Display**: Tabel HTML dan display DataFrame yang aman
- **Error Handling**: Penanganan error yang robust untuk masalah display data

### 2. Sanitasi Data
- **Safe Display Functions**: Mencegah error display
- **Data Type Handling**: Konversi dan validasi yang tepat
- **Missing Value Handling**: Manajemen missing data yang komprehensif

### 3. Manajemen Session State
- **Model Persistence**: Simpan model yang telah dilatih dalam session
- **Data Caching**: Loading dan caching data yang efisien
- **State Management**: Mempertahankan pilihan pengguna dan hasil

### 4. Desain Responsif
- **Custom CSS**: Styling dan layout modern
- **Mobile Friendly**: Berfungsi di berbagai ukuran layar
- **Interactive Elements**: Kontrol dan visualisasi dinamis

## 🚨 Troubleshooting

### Masalah Umum

1. **Dependensi Hilang**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **File Data Tidak Ditemukan**
   - Pastikan dataset ada di folder `Dataset/`
   - Periksa nama file: `WHO Life Expectancy Descriptive Statistics - Raw Data.csv`

3. **Error PyArrow**
   - Ini dapat diabaikan dengan aman - aplikasi menggunakan metode display alternatif
   - Periksa console untuk error kritis apapun

4. **Masalah Loading Peta**
   - Pastikan koneksi internet untuk map tiles
   - Periksa instalasi folium dan streamlit-folium

5. **Masalah Memori**
   - Kurangi ukuran dataset untuk testing
   - Gunakan model yang lebih kecil (estimator lebih sedikit untuk ensemble methods)

### Optimasi Performa

1. **Pelatihan Lebih Cepat**
   - Kurangi fold cross-validation
   - Gunakan parameter grid yang lebih kecil
   - Kurangi jumlah estimator dalam ensemble methods

2. **Optimasi Memori**
   - Proses data dalam chunks
   - Gunakan tipe data dengan footprint memori lebih kecil
   - Bersihkan variabel ketika tidak diperlukan

## 📝 Referensi API

### DataPreprocessor Class
```python
preprocessor = DataPreprocessor()
df_clean = preprocessor.load_and_clean_data(file_path)
X_scaled, y, features = preprocessor.prepare_final_dataset(df)
```

### LifeExpectancyModel Class
```python
model_trainer = LifeExpectancyModel()
results = model_trainer.train_models(X, y, feature_names)
importance = model_trainer.analyze_feature_importance()
```

### OutlierDetectionSystem Class
```python
outlier_system = OutlierDetectionSystem()
analysis = outlier_system.comprehensive_outlier_analysis(df, target_column='Life expectancy')
```

### Fungsi Visualisasi Peta
```python
map_obj = create_interactive_map(df, metric='Life expectancy', year=2015)
choropleth_fig = create_choropleth_map(df, metric='Life expectancy')
```

## 🤝 Kontribusi

1. Fork repository
2. Buat feature branch: `git checkout -b feature-name`
3. Commit perubahan: `git commit -am 'Add feature'`
4. Push ke branch: `git push origin feature-name`
5. Submit pull request

## 📄 Lisensi

Proyek ini dilisensikan di bawah MIT License - lihat file LICENSE untuk detail.

## 🙏 Ucapan Terima Kasih

- **Organisasi Kesehatan Dunia**: Untuk menyediakan dataset komprehensif
- **Scikit-learn**: Untuk library machine learning yang luar biasa
- **Streamlit**: Untuk framework aplikasi web
- **Plotly**: Untuk visualisasi interaktif
- **Folium**: Untuk visualisasi peta interaktif

---

**Catatan**: Aplikasi ini dirancang untuk tujuan pendidikan dan penelitian. Untuk penggunaan produksi, validasi tambahan, testing, dan pertimbangan deployment harus diimplementasikan. 